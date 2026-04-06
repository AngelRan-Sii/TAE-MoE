import logging
import os
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


# --- 核心改进：全自动特征对齐函数（兼容 obs_len=4/6/8/任意） ---
def env_data_processing(env_data, target_obs_len=None):
    """
    env_data: list[dict]
    return: dict[key] -> Tensor[B, obs_len, feat_dim]
    """
    if len(env_data) == 0:
        return {}

    # 1) obs_len 来源：优先用外部传入（最稳）
    if target_obs_len is None:
        first_item = env_data[0]
        probe_key = next((k for k in first_item.keys() if k != "location"), None)
        if probe_key is None:
            raise ValueError("env_data_processing: no feature keys except 'location'")
        v0 = first_item[probe_key]
        if isinstance(v0, list):
            target_obs_len = len(v0)
        else:
            t0 = torch.as_tensor(v0)
            if t0.dim() == 0:
                raise ValueError("env_data_processing: scalar feature but no target_obs_len")
            target_obs_len = int(t0.shape[0])
    target_obs_len = int(target_obs_len)

    # 2) 预扫描：找每个 key 的最大 feature_dim（按“单个时间步 flatten 后的长度”）
    max_feature_dims = {}
    for item in env_data:
        for key, value in item.items():
            if key == "location":
                continue

            # value 是 list：按时间步扫描每一个元素的维度
            if isinstance(value, list):
                if len(value) == 0:
                    f_size = 1
                else:
                    f_size = 1
                    for vv in value[:target_obs_len]:
                        tv = torch.as_tensor(vv)
                        f_size = max(f_size, int(tv.numel()))
            else:
                tv = torch.as_tensor(value)
                if tv.dim() == 0:
                    f_size = 1
                else:
                    # 默认认为第0维是时间，其他 flatten
                    f_size = int(tv.reshape(tv.shape[0], -1).shape[1])

            if key not in max_feature_dims or f_size > max_feature_dims[key]:
                max_feature_dims[key] = f_size

    env_data_merge = {k: [] for k in max_feature_dims}

    # 3) 构建对齐后的 [obs_len, feat_dim]
    for item in env_data:
        for key, max_dim in max_feature_dims.items():
            if key not in item:
                env_data_merge[key].append(torch.zeros((target_obs_len, max_dim), dtype=torch.float32))
                continue

            value = item[key]

            # --- 情况A：value 是 list（按时间步存的特征） ---
            if isinstance(value, list):
                out = torch.zeros((target_obs_len, max_dim), dtype=torch.float32)
                T = min(len(value), target_obs_len)
                for t in range(T):
                    tv = torch.as_tensor(value[t], dtype=torch.float32).reshape(-1)  # flatten
                    L = min(tv.numel(), max_dim)
                    out[t, :L] = tv[:L]
                env_data_merge[key].append(out)
                continue

            # --- 情况B：value 不是 list：认为它是 [T, ...] 或标量 ---
            tv = torch.as_tensor(value, dtype=torch.float32)
            if tv.dim() == 0:
                # 标量 -> 复制到每个时间步
                out = torch.zeros((target_obs_len, max_dim), dtype=torch.float32)
                out[:, 0] = tv
                env_data_merge[key].append(out)
                continue

            # 对齐时间轴
            if tv.shape[0] > target_obs_len:
                tv = tv[:target_obs_len]
            elif tv.shape[0] < target_obs_len:
                pad_shape = (target_obs_len - tv.shape[0], *tv.shape[1:])
                tv = torch.cat([tv, torch.zeros(pad_shape, dtype=torch.float32)], dim=0)

            tv = tv.reshape(target_obs_len, -1)
            out = torch.zeros((target_obs_len, max_dim), dtype=torch.float32)
            L = min(tv.shape[1], max_dim)
            out[:, :L] = tv[:, :L]
            env_data_merge[key].append(out)

    # 4) stack -> [B, obs_len, feat_dim]
    for key in env_data_merge:
        env_data_merge[key] = torch.stack(env_data_merge[key], dim=0)

    return env_data_merge


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
     obs_date_mask, pred_date_mask, image_obs, image_pre, env_data, tyID) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = torch.LongTensor([[s, e] for s, e in zip(cum_start_idx, cum_start_idx[1:])])

    # 维度转置以适配 LSTM/Transformer: [Seq, Batch, Feat]
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)         # [obs_len, B, 2]
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)       # [pred_len, B, 2]
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1) # [obs_len, B, 2]
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)

    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)

    obs_traj_Me = torch.cat(obs_traj_Me, dim=0).permute(2, 0, 1)
    pred_traj_Me = torch.cat(pred_traj_gt_Me, dim=0).permute(2, 0, 1)
    obs_traj_rel_Me = torch.cat(obs_traj_rel_Me, dim=0).permute(2, 0, 1)
    pred_traj_rel_Me = torch.cat(pred_traj_gt_rel_Me, dim=0).permute(2, 0, 1)

    obs_date_mask = torch.cat(obs_date_mask, dim=0).permute(2, 0, 1)
    pred_date_mask = torch.cat(pred_date_mask, dim=0).permute(2, 0, 1)

    # 图片数据 stack: [Batch, Channel, Seq, H, W]
    # 单样本 image_obs: [obs_len, H, W, 1]
    image_obs = torch.stack(image_obs, dim=0).permute(0, 4, 1, 2, 3).contiguous()
    image_pre = torch.stack(image_pre, dim=0).permute(0, 4, 1, 2, 3).contiguous()

    # ✅ 不再依赖 env_data['wind'] 来推断 obs_len，直接用 obs_traj 的长度
    env_data = env_data_processing(env_data, target_obs_len=obs_traj.size(0))

    return (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
            loss_mask, seq_start_end, obs_traj_Me, pred_traj_Me, obs_traj_rel_Me, pred_traj_rel_Me,
            obs_date_mask, pred_date_mask, image_obs, image_pre, env_data, tyID)


def read_file(_path, delim='\t'):
    data, add = [], []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            add.append(line[-2:])
            data.append([float(i) for i in line[:-2]])
    return {'main': np.asarray(data), 'addition': add}


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=4,         
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim='\t',
        other_modal='gph',
        areas=None,
        test_year=None
    ):
        super(TrajectoryDataset, self).__init__()
        self.obs_len, self.pred_len, self.skip = int(obs_len), int(pred_len), int(skip)
        self.seq_len = self.obs_len + self.pred_len
        self.modal_name = other_modal
        self.areas = areas if areas else ['WP']
        test_year = test_year if test_year else '.txt'

        all_files = []
        for area in self.areas:
            p = os.path.join(data_dir['root'], area, data_dir['type'])
            if os.path.exists(p):
                all_files += [os.path.join(p, f) for f in os.listdir(p) if test_year in f]

        self.seq_list, self.seq_list_rel, self.seq_list_date_mask = [], [], []
        self.loss_mask_list, self.non_linear_ped, self.tyID = [], [], []
        num_peds_in_seq = []

        for path in all_files:
            tyname = os.path.splitext(os.path.basename(path))[0]
            data_dict = read_file(path, delim)
            addinf, data = data_dict['addition'], data_dict['main']

            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[f == data[:, 0], :] for f in frames]

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                if idx + self.seq_len > len(frame_data):
                    break

                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_seq = np.unique(curr_seq_data[:, 1])

                c_seq = np.zeros((len(peds_in_seq), 4, self.seq_len))
                c_rel = np.zeros((len(peds_in_seq), 4, self.seq_len))
                c_mask = np.zeros((len(peds_in_seq), self.seq_len))
                c_date = np.zeros((len(peds_in_seq), 4, self.seq_len))

                num_peds = 0
                dates = [x[0] for x in addinf[idx:idx + self.seq_len]]  # ✅ 保证 dates 一定定义且长度=seq_len
                for pid in peds_in_seq:
                    p_seq = curr_seq_data[curr_seq_data[:, 1] == pid, :]
                    if len(p_seq) != self.seq_len:
                        continue

                    p_seq_feat = np.transpose(p_seq[:, 2:])  # [4, seq_len]
                    p_rel = np.zeros(p_seq_feat.shape)
                    p_rel[:, 1:] = p_seq_feat[:, 1:] - p_seq_feat[:, :-1]

                    c_seq[num_peds] = p_seq_feat
                    c_rel[num_peds] = p_rel
                    c_mask[num_peds] = 1

                    c_date[num_peds] = self.embed_time(dates)
                    self.non_linear_ped.append(1.0 if np.sum(np.var(p_seq_feat[:2], axis=1)) > threshold else 0.0)
                    num_peds += 1

                if num_peds >= min_ped:
                    num_peds_in_seq.append(num_peds)
                    self.loss_mask_list.append(c_mask[:num_peds])
                    self.seq_list.append(c_seq[:num_peds])
                    self.seq_list_rel.append(c_rel[:num_peds])
                    self.seq_list_date_mask.append(c_date[:num_peds])

                    # ✅ tyID 里的 tydate 必须长度=seq_len，且 obs/pred 用 self.obs_len 切
                    self.tyID.append({
                        'old': [tyname, idx],
                        'new': addinf[idx + self.obs_len - 1],
                        'tydate': dates
                    })

        self.num_seq = len(self.seq_list)
        if self.num_seq > 0:
            all_s = np.concatenate(self.seq_list, axis=0)
            all_r = np.concatenate(self.seq_list_rel, axis=0)
            all_d = np.concatenate(self.seq_list_date_mask, axis=0)

            # 注意：这里的 obs/pred 切分完全由 self.obs_len/self.pred_len 决定
            self.obs_traj = torch.from_numpy(all_s[:, :2, :self.obs_len]).float()
            self.pred_traj = torch.from_numpy(all_s[:, :2, self.obs_len:]).float()
            self.obs_traj_rel = torch.from_numpy(all_r[:, :2, :self.obs_len]).float()
            self.pred_traj_rel = torch.from_numpy(all_r[:, :2, self.obs_len:]).float()

            self.obs_traj_Me = torch.from_numpy(all_s[:, 2:, :self.obs_len]).float()
            self.pred_traj_Me = torch.from_numpy(all_s[:, 2:, self.obs_len:]).float()
            self.obs_traj_rel_Me = torch.from_numpy(all_r[:, 2:, :self.obs_len]).float()
            self.pred_traj_rel_Me = torch.from_numpy(all_r[:, 2:, self.obs_len:]).float()

            self.loss_mask = torch.from_numpy(np.concatenate(self.loss_mask_list)).float()
            self.non_linear_ped = torch.from_numpy(np.array(self.non_linear_ped)).float()

            self.obs_date_mask = torch.from_numpy(all_d[:, :, :self.obs_len]).float()
            self.pred_date_mask = torch.from_numpy(all_d[:, :, self.obs_len:]).float()

            cum = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [(cum[i], cum[i + 1]) for i in range(len(cum) - 1)]
        else:
            self.seq_start_end = []

    def __len__(self):
        return self.num_seq

    def embed_time(self, date_list):
        embeds = []
        for d in date_list:
            embeds.append([
                (float(d[:4]) - 1949) / 70 - 0.5,
                (float(d[4:6]) - 1) / 11 - 0.5,
                (float(d[6:8]) - 1) / 30 - 0.5,
                float(d[8:10]) / 18 - 0.5
            ])
        return np.array(embeds).transpose(1, 0)[np.newaxis, :, :]

    def get_img(self, ty_dic):
        tyname = ty_dic['new'][1]
        area, year, tydate = ty_dic['old'][0][:2], ty_dic['old'][0][2:6], ty_dic['tydate']

        root = r'/path/to/your/project_root'
        env_root = os.path.join(root, 'all_area_correct_location_includeunder15_2023')
        modal_root = os.path.join(root, 'all_ocean_gph500_2023')

        ty_dir = next(
            (v for v in [tyname, tyname.capitalize(), tyname.upper()]
             if os.path.exists(os.path.join(env_root, area, year, v))),
            tyname
        )

        def read_npy(p):
            if not os.path.exists(p):
                return np.zeros((64, 64, 1), dtype=np.float32)
            img = cv2.resize(np.load(p), (64, 64))
            img = np.expand_dims(np.clip((img - 44490) / (58768 - 44490), 0, 1), -1)
            return img.astype(np.float32)

        # ✅ obs/pre 图片严格按 self.obs_len/self.pred_len 切
        obs_imgs = [read_npy(os.path.join(modal_root, area, year, ty_dir, d + '.npy'))
                    for d in tydate[:self.obs_len]]
        pre_imgs = [read_npy(os.path.join(modal_root, area, year, ty_dir, d + '.npy'))
                    for d in tydate[self.obs_len:self.obs_len + self.pred_len]]

        # 处理环境特征：只取观测段（obs_len）
        env_feat = {}
        for d in tydate[:self.obs_len]:
            p = os.path.join(env_root, area, year, ty_dir, d + '.npy')
            data = np.load(p, allow_pickle=True).item() if os.path.exists(p) else {'wind': -1}
            for k, v in data.items():
                if k == 'location':
                    continue
                if k not in env_feat:
                    env_feat[k] = []
                env_feat[k].append(v)

        return {
            'obs': torch.tensor(np.array(obs_imgs), dtype=torch.float32),   # [obs_len, 64,64,1]
            'pre': torch.tensor(np.array(pre_imgs), dtype=torch.float32),   # [pred_len,64,64,1]
            'env': env_feat
        }

    def __getitem__(self, index):
        s, e = self.seq_start_end[index]
        img_data = self.get_img(self.tyID[index])
        return [
            self.obs_traj[s:e], self.pred_traj[s:e], self.obs_traj_rel[s:e], self.pred_traj_rel[s:e],
            self.non_linear_ped[s:e], self.loss_mask[s:e],
            self.obs_traj_Me[s:e], self.pred_traj_Me[s:e], self.obs_traj_rel_Me[s:e], self.pred_traj_rel_Me[s:e],
            self.obs_date_mask[s:e], self.pred_date_mask[s:e],
            img_data['obs'], img_data['pre'], img_data['env'], self.tyID[index]
        ]
