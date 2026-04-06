import argparse
import gc
import logging
import os
import sys
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))

from collections import defaultdict
# from visdom import Visdom
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F 

from TCNM.data.loader_training4to4 import data_loader
from TCNM.losses import gan_g_loss, gan_d_loss, l2_loss, TripletLoss,toNE,trajectory_displacement_error,value_error,trajectory_diff,value_diff
from TCNM.losses import displacement_error, final_displacement_error

# from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from TCNM.models_prior_unet4to4 import TrajectoryGenerator, TrajectoryDiscriminator
from TCNM.utils import int_tuple, bool_flag, get_total_norm,to_numpy,dic2cuda
from TCNM.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

# -------------------------
# 默认配置（可用命令行覆盖）
# -------------------------
AREAS_DEFAULT = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']
BATCH_SIZE_DEFAULT = 64
GPU_NUM_DEFAULT = '0'
DATASET_NAME_DEFAULT = 'bst_divi10_train_val_test_inlcude15_new'
DATASET_ROOT_DEFAULT = '/inspire/hdd/global_user/fengzhaoran-253107020007/TropiCycloneNet1/bst_divi10_train_val_test_inlcude15_2023_new'
MODAL_DEFAULT = 'gph'
PI_PRE_DEFAULT = True
PI_PRE_EPOCH_DEFAULT = 1

logger = None  # ✅ main 里初始化


def setup_logger(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    _logger = logging.getLogger(f"train::{output_dir}")
    _logger.setLevel(logging.INFO)
    _logger.propagate = False

    # 防止重复 addHandler
    if _logger.handlers:
        return _logger

    fmt = logging.Formatter('[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(os.path.join(output_dir, 'out.log'), mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    _logger.addHandler(sh)
    _logger.addHandler(fh)
    return _logger


def parse_areas_arg(x):
    """
    兼容两种写法：
    1) --areas EP NA WP
    2) --areas EP,NA,WP
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str) and (',' in x):
        return [s.strip() for s in x.split(',') if s.strip()]
    return [x]


parser = argparse.ArgumentParser()

# ---- Dataset / IO ----
parser.add_argument('--pinet_pre', default=PI_PRE_DEFAULT, type=bool_flag)     # ✅ 修复：用 bool_flag
parser.add_argument('--areas', default=AREAS_DEFAULT, nargs='+', type=str)     # ✅ 修复：list 参数别用 type=str
parser.add_argument('--other_modal', default=MODAL_DEFAULT, type=str)
parser.add_argument('--dataset_root', default=DATASET_ROOT_DEFAULT, type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)

# ✅ 这里就是你要的可变帧数入口：4/6/8 都行
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=4, type=int)

parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--finetune', default=True, type=bool_flag)

# ---- Optimization ----
parser.add_argument('--batch_size', default=BATCH_SIZE_DEFAULT, type=int)
parser.add_argument('--num_iterations', default=5000, type=int)
parser.add_argument('--num_epochs', default=100 + PI_PRE_EPOCH_DEFAULT, type=int)

# ---- Model ----
parser.add_argument('--embedding_dim', default=32, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=128, type=int)

# ---- Generator ----
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=64, type=int)
parser.add_argument('--noise_dim', default=(16,), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=1e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# ---- Pooling ----
parser.add_argument('--pooling_type', default=None)
parser.add_argument('--pool_every_timestep', default=False, type=bool_flag)
parser.add_argument('--bottleneck_dim', default=16, type=int)
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# ---- Discriminator ----
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=128, type=int)
parser.add_argument('--d_learning_rate', default=1e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=2.0, type=float)

# ---- Loss ----
parser.add_argument('--l2_loss_weight', default=1.0, type=float)
parser.add_argument('--best_k', default=6, type=int)

# ---- Output / Checkpoint ----
parser.add_argument('--output_dir', default=None, type=str)  # ✅ 改成运行时根据 obs/pred 自动生成
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=5000, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# ---- Misc ----
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default=GPU_NUM_DEFAULT, type=str)


def finetune_ini_weight(generator, discriminator):
    mmstn_path = 'pretrain_model/MMSTN_finetune.pt'
    checkpoint = torch.load(mmstn_path)

    g_model_dict = generator.state_dict()
    pre_g_state_dict = {k: v for k, v in checkpoint['g_state'].items() if k in g_model_dict.keys()}

    d_model_dict = discriminator.state_dict()
    pre_d_state_dict = {k: v for k, v in checkpoint['d_state'].items() if k in d_model_dict.keys()}

    g_model_dict.update(pre_g_state_dict)
    d_model_dict.update(pre_d_state_dict)

    generator.load_state_dict(g_model_dict)
    discriminator.load_state_dict(d_model_dict)
    return generator, discriminator


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    global logger

    # ✅ areas 统一成 list
    if isinstance(args.areas, str):
        args.areas = parse_areas_arg(args.areas)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    # ✅ output_dir 自动带上 obs/pred，避免不同设置互相覆盖
    if args.output_dir is None:
        dataset_name = DATASET_NAME_DEFAULT
        modal = args.other_modal
        args.output_dir = f"model_save{args.obs_len}to{args.pred_len}/test_{dataset_name}_{modal}_o{args.obs_len}_p{args.pred_len}"

    logger = setup_logger(args.output_dir)

    train_path = get_dset_path(args.dataset_root, 'train')
    val_path = get_dset_path(args.dataset_root, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info('There are {} iterations per epoch'.format(iterations_per_epoch))

    # -------------------------
    # Build models (obs/pred fully controlled by args)
    # -------------------------
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm
    )

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type
    )

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss
    tripletLoss = TripletLoss(1)

    optimizer_g = optim.AdamW(generator.parameters(), lr=args.g_learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.d_learning_rate, betas=(0.5, 0.999))

    lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, args.num_epochs, eta_min=0, last_epoch=-1
    )
    lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, args.num_epochs, eta_min=0, last_epoch=-1
    )

    # -------------------------
    # Restore checkpoint
    # -------------------------
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir, f'{args.checkpoint_name}_with_model.pt')

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {'t': None, 'epoch': None},
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }

    t0 = None
    pi_pre_epoch = PI_PRE_EPOCH_DEFAULT

    while t < args.num_iterations:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {},lr {}'.format(epoch, optimizer_g.state_dict()['param_groups'][0]['lr']))

        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            pi_end = False

            # ✅ pinet_pre 是 bool，现在逻辑才正确
            if epoch > pi_pre_epoch or (not args.pinet_pre):
                losses_d = discriminator_step(
                    args, batch, generator, discriminator, d_loss_fn, optimizer_d, tripletLoss
                )
                losses_g = generator_step(
                    args, batch, generator, discriminator, g_loss_fn, optimizer_g
                )
            if not pi_end:
                losses_netchooser = net_chooser_step(
                    args, batch, generator, discriminator, optimizer_g
                )

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(t - 1, time.time() - t0))
                t0 = time.time()

            # Print
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                if epoch > pi_pre_epoch or (not args.pinet_pre):
                    for k, v in sorted(losses_d.items()):
                        logger.info('  [D] {}: {:.3f}'.format(k, v))
                        checkpoint['D_losses'][k].append(v)
                    for k, v in sorted(losses_g.items()):
                        logger.info('  [G] {}: {:.3f}'.format(k, v))
                        checkpoint['G_losses'][k].append(v)
                if not pi_end:
                    for k, v in sorted(losses_netchooser.items()):
                        logger.info('  [NET] {}: {:.3f}'.format(k, v))
                checkpoint['losses_ts'].append(t)

            # Checkpoint
            if t > 0 and t % args.checkpoint_every == 0 and epoch > pi_pre_epoch:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(args, val_loader, generator, discriminator, d_loss_fn)

                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(args, train_loader, generator, discriminator, d_loss_fn, limit=True)

                for k, v in sorted(metrics_val.items()):
                    if k in ['traj_error', 'pv_error']:
                        logger.info('  [val] {}: {}'.format(k, v))
                        continue
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)

                for k, v in sorted(metrics_train.items()):
                    if k in ['traj_error', 'pv_error']:
                        logger.info('  [train] {}: {}'.format(k, v))
                        continue
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade']) if len(checkpoint['metrics_val']['ade']) else metrics_val['ade']
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl']) if len(checkpoint['metrics_val']['ade_nl']) else metrics_val['ade_nl']

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()

                os.makedirs(args.output_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model_%0.5d.pt' % (args.checkpoint_name, t)
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                checkpoint_path = os.path.join(args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {k: v for k, v in checkpoint.items() if k not in key_blacklist}
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            if t >= args.num_iterations:
                break

        lr_schedulerD.step()
        lr_schedulerG.step()


def net_chooser_step(args, batch, generator, discriminator, optimizer_g, weighting_target='ml'):
    env_data = dic2cuda(batch[-2])
    batch = [tensor.cuda() for tensor in batch[:-2]]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end, obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
     obs_date_mask, pred_date_mask, image_obs, image_pre) = batch

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
    gt_xy = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
    obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
    pred_traj_gt_rel = torch.cat([pred_traj_gt_rel, pred_traj_gt_rel_Me], dim=2)

    generator_out, image_out, net_chooser_weights, _ = generator(
        obs_traj, obs_traj_rel, seq_start_end, image_obs, env_data,
        num_samples=1, all_g_out=True
    )

    pred_traj_fake_rel = generator_out
    gen_out = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    if weighting_target == "l2":
        l2_dist = torch.norm(gen_out - gt_xy[:, None], p=2, dim=-1).mean(0)
        per_gen_dist = l2_dist.min(0)[0].transpose(0, 1)
        min_idx = torch.argmin(per_gen_dist, dim=1)
        loss = F.cross_entropy(net_chooser_weights, min_idx)

    elif weighting_target == "ml":
        out_probs = torch.softmax(net_chooser_weights, 1)
        log_prob = (
            torch.distributions.Normal(0, 1)
            .log_prob(gen_out - gt_xy[:, None])
            .sum([0, -1])
            .t()
        )
        gen_prob = torch.softmax(log_prob, 1)
        loss = -(gen_prob * out_probs.log()).sum(1).mean()

    elif weighting_target == "endpoint":
        l2_dist = torch.norm(gen_out[-1] - gt_xy[-1, None], p=2, dim=-1)
        per_gen_dist = l2_dist.min(0)[0].transpose(0, 1)
        min_idx = torch.argmin(per_gen_dist, dim=1)
        loss = F.cross_entropy(net_chooser_weights, min_idx)

    else:
        raise ValueError("Weighting target does not exist")

    losses["train_net_chooser_loss"] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()

    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
    optimizer_g.step()
    return losses


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d, tripletLoss):
    mse = nn.MSELoss()
    env_data = dic2cuda(batch[-2])
    batch = [tensor.cuda() for tensor in batch[:-2]]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end, obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
     obs_date_mask, pred_date_mask, image_obs, image_pre) = batch

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
    pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
    obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
    pred_traj_gt_rel = torch.cat([pred_traj_gt_rel, pred_traj_gt_rel_Me], dim=2)

    generator_out, image_out, _, _ = generator(
        obs_traj, obs_traj_rel, seq_start_end, image_obs, env_data,
        num_samples=1, all_g_out=False
    )

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake.squeeze()], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel.squeeze()], dim=0)

    img_real = torch.cat([image_obs, image_pre], dim=2)  # [b,c,obs+pred,h,w]
    img_fake = image_out                                 # [b,c,obs+pred,h,w]

    scores_fake, _ = discriminator(traj_fake, traj_fake_rel, seq_start_end, img_fake)
    scores_real, _ = discriminator(traj_real, traj_real_rel, seq_start_end, img_real)

    image_loss = mse(img_real[:, :, 1:], image_out[:, :, 1:])
    data_loss = d_loss_fn(scores_real, scores_fake)

    losses['D_data_loss'] = data_loss.item()
    losses['image_loss'] = image_loss.item()

    loss += data_loss
    loss += image_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()

    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g):
    mse = nn.MSELoss()
    env_data = dic2cuda(batch[-2])
    batch = [tensor.cuda() for tensor in batch[:-2]]

    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end, obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
     obs_date_mask, pred_date_mask, image_obs, image_pre) = batch

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    # ✅ pred 部分 mask，完全由 args.obs_len 控制（适配 4/6/8）
    loss_mask = loss_mask[:, args.obs_len:]

    obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
    pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
    obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
    pred_traj_gt_rel = torch.cat([pred_traj_gt_rel, pred_traj_gt_rel_Me], dim=2)

    generator_out, image_out, _, _ = generator(
        obs_traj, obs_traj_rel, seq_start_end, image_obs, env_data,
        num_samples=args.best_k, all_g_out=False, predrnn_img=None
    )

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    for sample_i in range(args.best_k):
        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(
                args.l2_loss_weight * l2_loss(
                    pred_traj_fake[:, sample_i].squeeze(),
                    pred_traj_gt,
                    loss_mask,
                    mode='raw'
                )
            )

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel

        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake[:, -1].squeeze()], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel[:, -1].squeeze()], dim=0)

    img_real = torch.cat([image_obs, image_pre], dim=2)
    img_fake = image_out

    scores_fake, _ = discriminator(traj_fake, traj_fake_rel, seq_start_end, img_fake)
    discriminator_loss = g_loss_fn(scores_fake)

    image_loss = mse(img_real, img_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()

    loss += image_loss
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()

    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
    optimizer_g.step()

    return losses


def check_accuracy(args, loader, generator, discriminator, d_loss_fn, limit=False):
    d_losses = []
    metrics = {}
    tde, ve = [], []
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0

    generator.eval()
    with torch.no_grad():
        for batch in loader:
            env_data = dic2cuda(batch[-2])
            batch = [tensor.cuda() for tensor in batch[:-2]]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end, obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
             obs_date_mask, pred_date_mask, image_obs, image_pre) = batch

            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
            pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
            obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            pred_traj_gt_rel = torch.cat([pred_traj_gt_rel, pred_traj_gt_rel_Me], dim=2)

            pred_traj_fake_rel, image_out, _, _ = generator(
                obs_traj, obs_traj_rel, seq_start_end, image_obs, env_data,
                num_samples=1, all_g_out=False
            )

            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1]).squeeze()
            pred_traj_fake_rel = pred_traj_fake_rel.squeeze()

            real_pred_traj_fake, real_pred_traj_fake_Me = toNE(
                torch.clone(pred_traj_fake[:, :, :2]), torch.clone(pred_traj_fake[:, :, 2:])
            )
            real_pred_traj_gt, real_pred_traj_gt_Me = toNE(
                torch.clone(pred_traj_gt[:, :, :2]), torch.clone(pred_traj_gt_Me)
            )

            tde.append(trajectory_displacement_error(
                real_pred_traj_fake, real_pred_traj_gt, mode='raw'
            ).data.cpu().numpy())
            ve.append(value_error(
                real_pred_traj_fake_Me, real_pred_traj_gt_Me, mode='raw'
            ).data.cpu().numpy())

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask
            )

            ade, ade_l, ade_nl = cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)
            fde, fde_l, fde_nl = cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            img_real = torch.cat([image_obs, image_pre], dim=2)
            img_fake = image_out

            scores_fake, _ = discriminator(traj_fake, traj_fake_rel, seq_start_end, img_fake)
            scores_real, _ = discriminator(traj_real, traj_real_rel, seq_start_end, img_real)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses) if len(d_losses) else 0.0
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum if loss_mask_sum else 0.0
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum if loss_mask_sum else 0.0

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len) if total_traj else 0.0
    metrics['fde'] = sum(f_disp_error) / total_traj if total_traj else 0.0

    metrics['traj_error'] = np.mean(np.concatenate(tde), axis=0) if len(tde) else None
    metrics['pv_error'] = np.mean(np.concatenate(ve), axis=0) if len(ve) else None

    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0

    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)