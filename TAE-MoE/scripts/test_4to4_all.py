import argparse
import os
import sys
import torch
import numpy as np
import time

# =========================
# 路径配置
# =========================
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from TCNM.data.loader_training4to4 import data_loader
from TCNM.models_prior_unet4to4 import TrajectoryGenerator
from TCNM.losses import toNE, trajectory_displacement_error
from TCNM.utils import relative_to_abs, dic2cuda, get_dset_path

# 请根据你的实际路径修改这里
DEFAULT_MODEL_DIR = "/inspire/qb-ilm/project/urbanlowaltitude/fengzhaoran-253107020007/TropiCycloneNet1/scripts/model_seed=5008to4/test_bst_divi10_train_val_test_inlcude15_new_gph_o8_p4"
DEFAULT_DATASET_ROOT = "/inspire/hdd/global_user/fengzhaoran-253107020007/TropiCycloneNet1/bst_divi10_train_val_test_inlcude15_2023_new"

# 默认参数
DEFAULT_DSET_TYPE = "test"
DEFAULT_AREAS = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']
DEFAULT_BEST_K = 6
DEFAULT_GPU_NUM = "0"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_WHICH_STATE = "g_state"

# =========================
# 工具函数
# =========================
def parse_areas_arg(x):
    if x is None: return DEFAULT_AREAS
    if isinstance(x, list):
        if len(x) == 1 and isinstance(x[0], str) and (',' in x[0]):
            return [s.strip() for s in x[0].split(',') if s.strip()]
        return x
    if isinstance(x, str) and (',' in x):
        return [s.strip() for s in x.split(',') if s.strip()]
    return [x]

def ensure_bt(x, B, T):
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 1:
        if x.shape[0] == B: return x.reshape(B, 1)
        if x.shape[0] == T: return x.reshape(1, T)
        return x.reshape(-1, 1)
    if x.shape[0] == B and x.shape[1] == T: return x
    if x.shape[0] == T and x.shape[1] == B: return x.T
    return x.reshape(-1, x.shape[-1])

def get_cli_args():
    p = argparse.ArgumentParser()
    p.add_argument('--obs_len', type=int, required=True)
    p.add_argument('--pred_len', type=int, required=True)
    p.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR)
    p.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT)
    # 循环范围
    p.add_argument('--start_step', type=int, default=5000)
    p.add_argument('--end_step', type=int, default=70000)
    p.add_argument('--step_interval', type=int, default=5000)
    
    p.add_argument('--dset_type', type=str, default=DEFAULT_DSET_TYPE)
    p.add_argument('--areas', nargs='+', default=None)
    p.add_argument('--best_k', type=int, default=DEFAULT_BEST_K)
    p.add_argument('--gpu_num', type=str, default=DEFAULT_GPU_NUM)
    p.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--loader_num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument('--which_state', type=str, default=DEFAULT_WHICH_STATE)
    return p.parse_args()

def build_generator_from_ckpt(ckpt, device, which_state="g_state"):
    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict): ckpt_args = vars(ckpt_args)

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
    if state is None: raise KeyError(f"checkpoint missing {which_state}")
    gen.load_state_dict(state)
    gen.to(device).eval()
    return gen, ckpt_args

# =========================
# 核心：单模型评估
# =========================
def evaluate_single_model(model_path, args, device):
    """
    加载模型 -> 计算 -> 返回包含每一步详情的字典
    """
    try:
        ckpt = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return None

    ckpt_args = ckpt["args"]
    if not isinstance(ckpt_args, dict): ckpt_args = vars(ckpt_args)
    
    # 简单的参数校验
    if args.obs_len != int(ckpt_args["obs_len"]) or args.pred_len != int(ckpt_args["pred_len"]):
        print(f"⚠️  Skip mismatch: Req({args.obs_len},{args.pred_len}) vs Ckpt({ckpt_args['obs_len']},{ckpt_args['pred_len']})")
        return None

    generator, model_args = build_generator_from_ckpt(ckpt, device, which_state=args.which_state)

    # 覆盖 loader 参数
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

    # 统计器
    n_total = 0
    ade_sum = 0.0
    fde_sum = 0.0
    
    tde_sum = np.zeros(pred_len, dtype=np.float64)
    press_sum = np.zeros(pred_len, dtype=np.float64)
    wind_sum = np.zeros(pred_len, dtype=np.float64)

    with torch.no_grad():
        for batch in loader:
            env_data = dic2cuda(batch[-2]) if device.type == "cuda" else batch[-2]
            batch_cuda = [t.to(device) for t in batch[:-2]]
            
            (obs_traj, pred_traj_gt, obs_traj_rel, _, _, _, seq_start_end, 
             obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, _, _, _, image_obs, _) = batch_cuda

            obs_all = torch.cat([obs_traj, obs_traj_Me], dim=2)
            obs_rel_all = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            gt_xy = pred_traj_gt

            # Predict
            pred_rel_all, _, _, _ = generator(
                obs_all, obs_rel_all, seq_start_end, image_obs, env_data,
                num_samples=best_k, all_g_out=False
            )
            pred_abs_all = relative_to_abs(pred_rel_all, obs_all[-1])

            # ADE / FDE
            diff = pred_abs_all[:, :, :, :2] - gt_xy[:, None, :, :]
            dist = torch.norm(diff, p=2, dim=-1)
            ade_kb = dist.mean(0)
            best_ade, best_idx = ade_kb.min(0)
            
            fde_kb = torch.norm(pred_abs_all[-1, :, :, :2] - gt_xy[-1, None, :, :], p=2, dim=-1)
            best_fde = fde_kb.gather(0, best_idx.view(1, -1)).squeeze(0)

            B = int(gt_xy.size(1))
            T = pred_len

            ade_sum += best_ade.sum().item()
            fde_sum += best_fde.sum().item()
            n_total += B

            # Gather Best Pred for Physical Metrics
            idx_expand = best_idx.view(1, 1, B, 1).expand(T, 1, B, 4)
            best_pred = pred_abs_all.gather(1, idx_expand).squeeze(1)

            real_pf_xy, real_pf_me = toNE(best_pred[:, :, :2].clone(), best_pred[:, :, 2:].clone())
            real_gt_xy, real_gt_me = toNE(pred_traj_gt.clone(), pred_traj_gt_Me.clone())

            # Metrics
            tde = trajectory_displacement_error(real_pf_xy, real_gt_xy, mode="raw")
            tde_bt = ensure_bt(tde, B, T)

            press_err = torch.abs(real_pf_me[:, :, 0] - real_gt_me[:, :, 0])
            wind_err = torch.abs(real_pf_me[:, :, 1] - real_gt_me[:, :, 1])

            press_bt = ensure_bt(press_err, B, T)
            wind_bt = ensure_bt(wind_err, B, T)

            # Sum over Batch (Output shape: [T])
            # ensure_bt 输出可能是 [B, T] 或 [T, B]，需要确保按 Time 维叠加
            if tde_bt.shape == (B, T):
                tde_sum += tde_bt.sum(axis=0)
                press_sum += press_bt.sum(axis=0)
                wind_sum += wind_bt.sum(axis=0)
            else:
                tde_sum += tde_bt.sum(axis=1)
                press_sum += press_bt.sum(axis=1)
                wind_sum += wind_bt.sum(axis=1)

    del generator, batch_cuda, env_data
    torch.cuda.empty_cache()

    if n_total == 0: return None

    # 计算平均值
    res = {
        "ADE": ade_sum / n_total,
        "FDE": fde_sum / n_total,
        "TDE_steps": tde_sum / n_total,    # 数组
        "Press_steps": press_sum / n_total, # 数组
        "Wind_steps": wind_sum / n_total,   # 数组
    }
    # 整体平均
    res["TDE_mean"] = res["TDE_steps"].mean()
    res["Press_mean"] = res["Press_steps"].mean()
    res["Wind_mean"] = res["Wind_steps"].mean()
    
    return res

# =========================
# 🖨️ 打印详细表格的函数
# =========================
def print_detailed_table(res, step_num):
    print("\n" + "=" * 70)
    print(f"📌 CHECKPOINT STEP: {step_num}")
    print("=" * 70)
    print(f"Overall ADE (km): {res['ADE']:.6f}")
    print(f"Overall FDE (km): {res['FDE']:.6f}")
    print("-" * 70)
    print(f"{'Step':<10} | {'TDE (km)':<15} | {'Press (hPa)':<15} | {'Wind (m/s)':<15}")
    print("-" * 70)
    
    steps_count = len(res['TDE_steps'])
    for i in range(steps_count):
        print(f"Step {i+1:<5} | {res['TDE_steps'][i]:<15.2f} | {res['Press_steps'][i]:<15.2f} | {res['Wind_steps'][i]:<15.2f}")
    
    print("-" * 70)
    print(f"{'AVG':<10} | {res['TDE_mean']:<15.2f} | {res['Press_mean']:<15.2f} | {res['Wind_mean']:<15.2f}")
    print("-" * 70 + "\n")

# =========================
# 主循环
# =========================
def main():
    args = get_cli_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    steps = list(range(args.start_step, args.end_step + 1, args.step_interval))

    print(f"Starting Batch Evaluation...")
    print(f"Model Dir: {args.model_dir}")
    print(f"Steps: {steps}")
    
    for step in steps:
        filename = f"checkpoint_with_model_{step:05d}.pt"
        model_path = os.path.join(args.model_dir, filename)

        if not os.path.exists(model_path):
            print(f"⚠️  File not found: {filename}, skipping...")
            continue
        
        # 执行评估
        t0 = time.time()
        res = evaluate_single_model(model_path, args, device)
        t_cost = time.time() - t0

        if res is not None:
            # ✅ 直接输出你想要的详细表格
            print_detailed_table(res, step)
            print(f"✅ Evaluated Step {step} in {t_cost:.1f}s")
        else:
            print(f"❌ Failed to evaluate step {step}")

if __name__ == "__main__":
    main()