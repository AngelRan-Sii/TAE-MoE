import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

'''
the shape of Conv3d input is (batch size, channel, sequence, height, width)
'''

class Conv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(True),
        )

        self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
            Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, x1_in, x2_in, out_channel, kernel_size, stride, padding):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose3d(x1_in, x1_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )
        self.conv = Conv3d(x1_in + x2_in, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # x shape: (N, C, D, H, W)
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)

        # ✅ F.pad 的顺序是 (W_left, W_right, H_left, H_right, D_left, D_right)
        x1 = F.pad(
            x1,
            [
                diffW // 2, diffW - diffW // 2,
                diffH // 2, diffH - diffH // 2,
                diffD // 2, diffD - diffD // 2
            ]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel_list, out_channel, kernel_size, stride, padding):
        super(OutConv, self).__init__()

        self.up_list = []
        for i, channel in enumerate(in_channel_list):
            if i == len(in_channel_list) - 1:
                continue
            scale = int(np.power(2, (len(in_channel_list) - 1) - i))
            self.up_list.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        channel, channel,
                        kernel_size=[1, scale, scale],
                        stride=[1, scale, scale],
                        padding=padding,
                        bias=True
                    ),
                    nn.ReLU(),
                    nn.Conv3d(channel, in_channel_list[-1], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm3d(in_channel_list[-1]),
                    nn.ReLU(inplace=True)
                )
            )
        self.up_list = nn.ModuleList(self.up_list)

        self.conv = nn.Sequential(
            nn.Conv3d(in_channel_list[-1], out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x6, x7, x8, x9 = tuple(x)

        x6 = self.up_list[0](x6)
        x7 = self.up_list[1](x7)
        x8 = self.up_list[2](x8)

        # ✅ 四个块现在 channel 都会变成 in_channel_list[-1]（默认 16），沿 time(D) 维拼接
        x_last = torch.cat([x6, x7, x8, x9], dim=2)
        return self.conv(x_last)


class Unet3D(nn.Module):
    """
    ✅ 支持可变 obs_len / pred_len
    输入:  (B, 1, obs_len, H, W)
    输出:  (B, 1, obs_len+pred_len-1, H, W)
    """
    def __init__(self, in_channel, out_channel, obs_len=8, pred_len=4):
        super(Unet3D, self).__init__()

        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)

        self.inc = Conv3d(in_channel, 16, kernel_size=3, stride=1, padding=1)
        self.down1 = Down(16, 32, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down2 = Down(32, 64, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down3 = Down(64, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])
        self.down4 = Down(128, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])

        self.up1 = Up(128, 128, 64, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up2 = Up(64, 64, 32, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up3 = Up(32, 32, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)
        self.up4 = Up(16, 16, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)

        # ---- 动态计算 outc 的时间 kernel ----
        # x6 的 time = T4 = floor((T0-2)/2 + 1) （因为 down3 把 time 下采样一次）
        T0 = self.obs_len
        T4 = int(np.floor((T0 - 2) / 2 + 1))
        T_sum = T4 + 3 * T0                 # x6 + x7 + x8 + x9 沿 time 拼起来的长度

        out_T = self.obs_len + self.pred_len - 1  # 你期望 Unet 输出长度
        # conv3d valid 输出: out_T = T_sum - kT + 1  => kT = T_sum - out_T + 1
        kT = T_sum - out_T + 1

        if kT < 1:
            raise ValueError(f"[Unet3D] Invalid kernel_t={kT}. obs_len={self.obs_len}, pred_len={self.pred_len}, T_sum={T_sum}, out_T={out_T}")

        self.outc = OutConv([64, 32, 16, 16], out_channel, kernel_size=[kT, 1, 1], stride=[1, 1, 1], padding=0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        out = self.outc([x6, x7, x8, x9])
        return out


if __name__ == '__main__':
    # 示例：4->4
    x = torch.randn((2, 1, 4, 64, 64)).cuda()
    net = Unet3D(1, 1, obs_len=4, pred_len=4).cuda()
    out = net(x)
    print("out shape:", out.shape)  # (2,1,7,64,64)
