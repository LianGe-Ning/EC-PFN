import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM_SK(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_types=['avg', 'max']):
        super(CBAM_SK, self).__init__()

        # Spatial attention branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Channel attention branch
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

        # SK module
        self.sk_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=32)
        self.sk_bn = nn.BatchNorm2d(in_channels)
        self.sk_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels * 32, bias=False),
            nn.BatchNorm1d(in_channels * 32),
            nn.ReLU(inplace=True)
        )

        # Output Conv2d
        self.out_conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_channels)

        self.pool_types = pool_types

    def forward(self, x):
        # CBAM
        # Channel attention
        channel_avg = torch.mean(x, dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        channel_max, _ = torch.max(x, dim=-1, keepdim=True)
        channel_att = self.mlp(torch.cat([channel_avg, channel_max], dim=1).squeeze(-1).squeeze(-1)).unsqueeze(
            -1).unsqueeze(-1)

        # Spatial attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        spatial_att = self.sigmoid(avg_out + max_out)

        # Multiply attention maps with input
        att = channel_att * spatial_att
        x = x * att

        # SK module
        x = self.sk_conv(x)
        x = self.sk_bn(x)
        N, C, H, W = x.size()
        feat = F.avg_pool2d(x, (H, W))
        feat = feat.view(N, C)
        feat = self.sk_mlp(feat)
        feat = feat.view(N, 32, C // 32)
        attn = F.softmax(feat, dim=1)
        x = x.view(N, 32, C // 32, H, W)
        x = x * attn.unsqueeze(-1)

        # Concatenate CBAM output and SK output
        x = torch.cat([att * x, x], dim=1)

        # Output Conv2d
        x = self.out_bn(self.out_conv(x))

        return x
