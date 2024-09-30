import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Compute spatial attention weights
        weights = torch.mean(x, dim=[2, 3], keepdim=True)
        weights = self.conv1(weights)
        weights = F.relu(weights, inplace=True)
        weights = self.conv2(weights)
        weights = torch.sigmoid(weights)

        # Apply spatial attention weights to input tensor
        x = x * weights
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio * 2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Compute channel attention weights
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.conv1(avg_out)
        max_out = self.conv1(max_out)
        avg_out = F.relu(avg_out, inplace=True)
        max_out = F.relu(max_out, inplace=True)
        weights = torch.cat([avg_out, max_out], dim=1)
        weights = self.conv2(weights)
        weights = torch.sigmoid(weights)

        # Apply channel attention weights to input tensor
        x = x * weights
        return x


class CBAM_4_3(nn.Module):
    def __init__(self, in_channels, out_channels=None, reduction_ratio=16,):
        super(CBAM_4_3, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.spatial_att = SpatialAttention(in_channels, reduction_ratio)
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Compute attention weights
        x_sp = self.spatial_att(x)
        x_ch = self.channel_att(x)
        x_att = x_sp + x_ch

        # Apply attention weights and convolution
        x = x * x_att
        x = self.conv(x)
        return x
