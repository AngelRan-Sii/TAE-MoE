# test_4to4.py
import argparse
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from TCNM.data.loader_training4to4 import data_loader
from TCNM.models_prior_unet4to4 import TrajectoryGenerator
from TCNM.losses import toNE, trajectory_displacement_error
from TCNM.utils import relative_to_abs, dic2cuda, get_dset_path


# =========================
# ✅ 默认参数（你也可以命令行覆盖）
# =========================
DEFAULT_MODEL_PATH = "/inspire/qb-ilm/project/urbanlowaltitude/fengzhaoran-253107020007/TropiCycloneNet1/scripts/model_save4to4/test_bst_divi10_train_val_test_inlcude15_new_gph_o4_p4/checkpoint_with_model_05000.pt"
DEFAULT_DATASET_ROOT = "/inspire/hdd/global_user/fengzhaoran-253107020007/TropiCycloneNet1/bst_divi10_train_val_test_inlcude15_2023_new"
DEFAULT_DSET_TYPE = "test"
DEFAULT_AREAS = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']
DEFAULT_BEST_K = 6
DEFAULT_GPU_NUM = "0"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_WHICH_STATE = "g_state"   # "g_state" / "g_best_state" / "g_best_nl_state"


def parse_areas_arg(x):
    """
    支持：
      --areas NA
      --areas EP NA WP
      --areas EP,NA,WP
    """
    if x is None:
        return DEFAULT_AREAS
    if isinstance(x, list):
        # argparse nargs='+' 会进这里
        if len(x) == 1 and isinstance(x[0], str) and (',' in x[0]):
            return [s.strip() for s in x[0].split(',') if s.strip()]
        return x
    if isinstance(x, str) and (',' in x):
        return [s.strip() for s in x.split(',') if s.strip()]
    return [x]


def ensure_bt(x, B, T):
    """把输出整理成 [B, T]，便于按步平均"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    if x.ndim == 1:
        if x.shape[0] == B:
            return x.reshape(B, 1)
        if x.shape[0] == T:
            return x.reshape(1, T)
        return x.reshape(-1, 1)

    if x.shape[0] == B and x.shape[1] == T:
        return x
    if x.shape[0] == T and x.shape[1] == B:
        return x.T

    return x.reshape(-1, x.shape[-1])


def get_cli_args():
    p = argparse.ArgumentParser()
    # 你想要的两个参数
    p.add_argument('--obs_len', type=int, required=True)
    p.add_argument('--pred_len', type=int, required=True)

    # 其他可选参数
    p.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument('--dset_type', type=str, default=DEFAULT_DSET_TYPE, choices=['train', 'val', 'test'])
    p.add_argument('--areas', nargs='+', default=None)
    p.add_argument('--best_k', type=int, default=DEFAULT_BEST_K)
    p.add_argument('--gpu_num', type=str, default=DEFAULT_GPU_NUM)
    p.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--loader_num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument('--which_state', type=str, default=DEFAULT_WHICH_STATE,
                   choices=['g_state', 'g_best_state', 'g_best_nl_state'])
    return p.parse_args()


def build_generator_from_ckpt(ckpt, device, which_state="g_state"):
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)

    gen = TrajectoryGenerator(
        obs_len=ckpt_args["obs_len"],
        pred_len=ckpt_args["pred_len"],
        embedding_dim=ckpt_args["embedding_dim"],
        encoder_h_dim=ckpt_args["encoder_h_dim_g"],
        decoder_h_dim=ckpt_args["decoder_h_dim_g"],
        mlp_dim=ckpt_args["mlp_dim"],
        num_layers=ckpt_args["num_layers"],
        noise_dim=tuple(ckpt_args["noise_dim"]) if isinstance(ckpt_args["noise_dim"], (list, tuple)) else ckpt_args["noise_dim"],
        noise_type=ckpt_args["noise_type"],
        noise_mix_type=ckpt_args["noise_mix_type"],
        pooling_type=ckpt_args.get("pooling_type", None),
        pool_every_timestep=ckpt_args.get("pool_every_timestep", False),
        dropout=ckpt_args.get("dropout", 0.0),
        bottleneck_dim=ckpt_args.get("bottleneck_dim", 16),
        neighborhood_size=ckpt_args.get("neighborhood_size", 2.0),
        grid_size=ckpt_args.get("grid_size", 8),
        batch_norm=ckpt_args.get("batch_norm", False),
    )

    state = ckpt.get(which_state, None)
    if state is None:
        raise KeyError(f"checkpoint 里没有 {which_state}，可用键：{list(ckpt.keys())}")

    gen.load_state_dict(state)
    gen.to(device).eval()
    return gen, ckpt_args


@torch.no_grad()
def main():
    args = get_cli_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ckpt = torch.load(args.model_path, map_location="cpu")

    # --- 读取 checkpoint 记录的 obs/pred，并校验你命令行传入的值 ---
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)

    ckpt_obs = int(ckpt_args["obs_len"])
    ckpt_pred = int(ckpt_args["pred_len"])

    if args.obs_len != ckpt_obs or args.pred_len != ckpt_pred:
        raise ValueError(
            f"[obs/pred 不匹配]\n"
            f"  你命令行传入: obs_len={args.obs_len}, pred_len={args.pred_len}\n"
            f"  但 checkpoint 记录: obs_len={ckpt_obs}, pred_len={ckpt_pred}\n"
            f"请换一个对应 obs/pred 的 checkpoint（或不要乱改 obs/pred）。"
        )

    generator, model_args = build_generator_from_ckpt(ckpt, device, which_state=args.which_state)

    # --- 用 ckpt 的训练参数作为 data_loader 的 base，然后覆盖数据相关项 ---
    model_args = dict(model_args)
    model_args["dataset_root"] = args.dataset_root
    model_args["areas"] = parse_areas_arg(args.areas)
    model_args["batch_size"] = args.batch_size
    model_args["loader_num_workers"] = args.loader_num_workers
    model_args["obs_len"] = args.obs_len
    model_args["pred_len"] = args.pred_len

    dset_path = get_dset_path(model_args["dataset_root"], args.dset_type)
    loader_args = argparse.Namespace(**model_args)
    _, loader = data_loader(loader_args, dset_path)

    pred_len = int(model_args["pred_len"])
    best_k = int(args.best_k)

    # running sums
    n_total = 0
    ade_sum = 0.0
    fde_sum = 0.0

    tde_sum = np.zeros(pred_len, dtype=np.float64)
    press_sum = np.zeros(pred_len, dtype=np.float64)
    wind_sum = np.zeros(pred_len, dtype=np.float64)

    print(f"Start evaluation | Best-of-{best_k} | obs={args.obs_len} pred={args.pred_len} | areas={model_args['areas']} | dset={args.dset_type}")

    for batch in loader:
        env_data = dic2cuda(batch[-2]) if device.type == "cuda" else batch[-2]
        batch_cuda = [t.to(device) for t in batch[:-2]]

        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end, obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
         obs_date_mask, pred_date_mask, image_obs, image_pre) = batch_cuda

        obs_all = torch.cat([obs_traj, obs_traj_Me], dim=2)          # [obs_len, B, 4]
        obs_rel_all = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
        gt_xy = pred_traj_gt                                         # [pred_len, B, 2]

        pred_rel_all, _, _, _ = generator(
            obs_all, obs_rel_all, seq_start_end, image_obs, env_data,
            num_samples=best_k, all_g_out=False
        )
        pred_abs_all = relative_to_abs(pred_rel_all, obs_all[-1])    # [T, K, B, 4]

        # best-of-k by ADE
        diff = pred_abs_all[:, :, :, :2] - gt_xy[:, None, :, :]
        dist = torch.norm(diff, p=2, dim=-1)                         # [T, K, B]
        ade_kb = dist.mean(0)                                        # [K, B]
        best_ade, best_idx = ade_kb.min(0)                           # [B], [B]

        fde_kb = torch.norm(pred_abs_all[-1, :, :, :2] - gt_xy[-1, None, :, :], p=2, dim=-1)
        best_fde = fde_kb.gather(0, best_idx.view(1, -1)).squeeze(0)

        B = int(gt_xy.size(1))
        T = pred_len

        ade_sum += best_ade.sum().item()
        fde_sum += best_fde.sum().item()
        n_total += B

        # gather best prediction: [T, B, 4]
        idx_expand = best_idx.view(1, 1, B, 1).expand(T, 1, B, 4)
        best_pred = pred_abs_all.gather(1, idx_expand).squeeze(1)

        # physical metrics
        real_pf_xy, real_pf_me = toNE(best_pred[:, :, :2].clone(), best_pred[:, :, 2:].clone())
        real_gt_xy, real_gt_me = toNE(pred_traj_gt.clone(), pred_traj_gt_Me.clone())

        tde = trajectory_displacement_error(real_pf_xy, real_gt_xy, mode="raw")
        tde_bt = ensure_bt(tde, B, T)

        press_err = torch.abs(real_pf_me[:, :, 0] - real_gt_me[:, :, 0])
        wind_err = torch.abs(real_pf_me[:, :, 1] - real_gt_me[:, :, 1])

        press_bt = ensure_bt(press_err, B, T)
        wind_bt = ensure_bt(wind_err, B, T)

        tde_sum += tde_bt.sum(0)
        press_sum += press_bt.sum(0)
        wind_sum += wind_bt.sum(0)

    overall_ade = ade_sum / max(n_total, 1)
    overall_fde = fde_sum / max(n_total, 1)

    avg_tde = tde_sum / max(n_total, 1)
    avg_press = press_sum / max(n_total, 1)
    avg_wind = wind_sum / max(n_total, 1)

    print("\n" + "=" * 64)
    print(f"Overall ADE: {overall_ade:.6f}")
    print(f"Overall FDE: {overall_fde:.6f}")
    print("-" * 64)
    print(f"{'Step':<10} | {'TDE (km)':<12} | {'Press (hPa)':<12} | {'Wind (m/s)':<10}")
    print("-" * 64)
    for i in range(pred_len):
        print(f"Step {i+1:<5} | {avg_tde[i]:<12.2f} | {avg_press[i]:<12.2f} | {avg_wind[i]:<10.2f}")
    print("-" * 64)
    print(f"{'AVG':<10} | {avg_tde.mean():<12.2f} | {avg_press.mean():<12.2f} | {avg_wind.mean():<10.2f}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
