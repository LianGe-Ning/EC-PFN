import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Elan (nn.Module):
    def __init__(self, in_channels, out_channels, n=1,  expand_ratio=0.5):
        super().__init__()
        mid_channels = int(in_channels * expand_ratio)
        self.initial_conv = Conv(in_channels, mid_channels, 1)
        self.main_conv = Conv(mid_channels, mid_channels, 3)
        self.final_conv = Conv(4*mid_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2_0 = self.initial_conv(x)
        x2_1 = self.main_conv(x2_0)
        x2_2 = self.main_conv(x2_1)
        x2_3 = self.main_conv(x2_2)
        x2_4 = self.main_conv(x2_3)
        y = torch.cat((x1, x2_0, x2_2, x2_4), dim=1)
        y = self.final_conv(y)
        return y


class Elan_2 (nn.Module):
    def __init__(self, in_channels, out_channels, n=1, expand_ratio=0.5):
        super().__init__()
        mid_channels = int(in_channels * expand_ratio)
        mid_channels_2 = int(mid_channels * expand_ratio)
        self.initial_conv = Conv(in_channels, mid_channels, 1)
        self.main_conv = Conv(mid_channels, mid_channels_2, 3)
        self.main_conv_2 = Conv(mid_channels_2, mid_channels_2, 3)
        self.final_conv = Conv(2 * mid_channels + 4 * mid_channels_2, out_channels, 1)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2_0 = self.initial_conv(x)
        x2_1 = self.main_conv(x2_0)
        x2_2 = self.main_conv_2(x2_1)
        x2_3 = self.main_conv_2(x2_2)
        x2_4 = self.main_conv_2(x2_3)
        y = torch.cat((x1, x2_0, x2_1, x2_2, x2_3, x2_4), dim=1)
        y = self.final_conv(y)
        return y

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(CAConv, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(oup),nn.ReLU())

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return self.conv(out)



class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=1 / 8):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel number of a convolution branch

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)

        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)

        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)

        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)

    def forward(self, x):
        # B, C, H, W = x.shape
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)

        return torch.cat((self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h), x_id), dim=1)


class RFCAConv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        super(RFCAConv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_conv1 = Conv(c1, 9 *c1, k=1, g=c1)
        self.group_conv2 = Conv(c1, 9 *c1, k=3, g=c1)
        self.group_conv3 = Conv(c1, 9 *c1, k=5, g=c1)

        self.softmax = nn.Softmax(dim=1)

        self.group_conv = Conv(c1, 9 * c1, k=3, g=c1)
        self.convDown = Conv(c1, c1, k=3, s=3)
        self.CA = CAConv(c1, c2, kernel_size, stride)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)

        group1 = self.softmax(self.group_conv1(y))
        group2 = self.softmax(self.group_conv2(y))
        group3 = self.softmax(self.group_conv3(y))
        # g1 =  torch.cat([group1, group2, group3], dim=1)

        g2 = self.group_conv(x)

        out1 = g2 * group1.expand_as(g2)
        out2 = g2 * group2.expand_as(g2)
        out3 = g2 * group3.expand_as(g2)

        out = sum([out1, out2, out3])
        # 获取输入特征图的形状
        batch_size, channels, height, width = out.shape

        # 计算输出特征图的通道数
        output_channels = channels // 9

        # 重塑并转置特征图以将通道数分成3x3个子通道并扩展高度和宽度
        out = out.view(batch_size, output_channels, 3, 3, height, width).permute(0, 1, 4, 2, 5,3).\
                                                reshape(batch_size, output_channels, 3 * height, 3 * width)
        out = self.convDown(out)
        out = self.CA(out)
        return out

class RFCAConv2(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        super(RFCAConv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_conv1 = Conv(c1, 3 *c1, k=1, g=c1)
        self.group_conv2 = Conv(c1, 3 *c1, k=3, g=c1)
        self.group_conv3 = Conv(c1, 3 *c1, k=5, g=c1)

        self.softmax = nn.Softmax(dim=1)

        self.group_conv = Conv(c1, 3 * c1, k=3, g=c1)
        self.convDown = Conv(c1, c1, k=3, s=3, g=c1)
        self.CA = CAConv(c1, c2, kernel_size, stride)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)

        group1 = self.softmax(self.group_conv1(y))
        group2 = self.softmax(self.group_conv2(y))
        group3 = self.softmax(self.group_conv3(y))
        # g1 =  torch.cat([group1, group2, group3], dim=1)

        g1 = self.group_conv(x)
        # g2 = self.group_conv(x)
        # g3 = self.group_conv(x)

        out1 = g1 * group1
        out2 = g1 * group2
        out3 = g1 * group3

        out =torch.cat([out1, out2, out3],dim=1)
        # 获取输入特征图的形状
        batch_size, channels, height, width = out.shape

        # 计算输出特征图的通道数
        output_channels = c

        # 重塑并转置特征图以将通道数分成3x3个子通道并扩展高度和宽度
        out = out.view(batch_size, output_channels, 3, 3, height, width).permute(0, 1, 4, 2, 5, 3).\
                                                reshape(batch_size, output_channels, 3 * height, 3 * width)
        # out = out.view(batch_size, output_channels, height*3, width*3)
        out = self.convDown(out)
        out = self.CA(out)
        return out

#---------------------------------------------------------------------------------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out



class BasicRFB_a(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out
#---------------------------------------------------------------------------------------------------------------------

