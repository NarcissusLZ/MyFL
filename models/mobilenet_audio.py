# models/mobilenet_audio.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Part 1: Audio / 2D Image Models (GroupNorm Version)
# ==========================================

class BlockV1(nn.Module):
    '''Depthwise conv + Pointwise conv (with GroupNorm)'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(BlockV1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        # 替换 BN 为 GN，设定 groups=8
        self.gn1 = nn.GroupNorm(8, in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(8, out_planes)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        return out


class MobileNetV1_Audio(nn.Module):
    def __init__(self, num_classes=35, input_channels=1):
        super(MobileNetV1_Audio, self).__init__()
        # 针对 40x81 这样的小尺寸输入
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 32)

        # 定义层结构：(输入通道, 输出通道, stride)
        self.layers = self._make_layers(in_planes=32)

        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        cfg = [
            (64, 1),
            (128, 2),  # 20x40
            (128, 1),
            (256, 2),  # 10x20
            (256, 1),
            (512, 2),  # 5x10
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),  # 3x5
            (1024, 1)
        ]

        for out_planes, stride in cfg:
            layers.append(BlockV1(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BlockV2(nn.Module):
    '''Inverted Residual Block (with GroupNorm)'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(BlockV2, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(8, out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(8, out_planes),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.stride == 1:
            if self.shortcut:
                out = out + self.shortcut(x)
            else:
                out = out + x
        return out


class MobileNetV2_Audio(nn.Module):
    def __init__(self, num_classes=35, input_channels=1):
        super(MobileNetV2_Audio, self).__init__()
        # NOTE: 针对 Log-Mel 40x81 输入的配置
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 32)

        # (expansion, out_planes, num_blocks, stride)
        self.cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 1),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(8, 1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for s in strides:
                layers.append(BlockV2(in_planes, out_planes, expansion, s))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.gn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ==========================================
# Part 2: IoT-23 / 1D Tabular Models (GroupNorm Version)
# ==========================================

class BlockV1_1D(nn.Module):
    '''Depthwise conv + Pointwise conv (1D version with GroupNorm)'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(BlockV1_1D, self).__init__()
        # 使用 Conv1d 处理序列特征
        self.conv1 = nn.Conv1d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        # GN 同样适用于 1D 数据 (N, C, L)
        self.gn1 = nn.GroupNorm(8, in_planes)
        self.conv2 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(8, out_planes)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        return out


class MobileNetV1_IoT23(nn.Module):
    '''Adapted MobileNet V1 for low-dimensional tabular data (with GroupNorm)'''

    def __init__(self, num_classes=5, input_dim=10):
        super(MobileNetV1_IoT23, self).__init__()

        self.input_dim = input_dim

        # Stem Layer: 1通道 -> 32通道
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 32)

        self.layers = self._make_layers(in_planes=32)

        self.linear = nn.Linear(512, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        cfg = [
            (64, 1),
            (128, 2),  # 10 -> 5
            (128, 1),
            (256, 1),  # 保持 5
            (256, 1),
            (512, 2),  # 5 -> 3
            (512, 1),
            (512, 1),
        ]

        for out_planes, stride in cfg:
            layers.append(BlockV1_1D(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, InputDim) -> 需要变成 (Batch, 1, InputDim)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layers(out)

        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BlockV2_1D(nn.Module):
    '''Inverted Residual Block (1D version with GroupNorm)'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(BlockV2_1D, self).__init__()
        self.stride = stride
        planes = expansion * in_planes

        # 1x1 升维
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)
        # 3x3 深度卷积
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)
        # 1x1 降维
        self.conv3 = nn.Conv1d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(8, out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(8, out_planes),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.stride == 1:
            if self.shortcut:
                out = out + self.shortcut(x)
            else:
                out = out + x
        return out


class MobileNetV2_IoT23(nn.Module):
    '''Adapted MobileNet V2 for low-dimensional tabular data (with GroupNorm)'''

    def __init__(self, num_classes=5, input_dim=10):
        super(MobileNetV2_IoT23, self).__init__()

        self.input_dim = input_dim
        # Stem
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 32)

        # Config: (expansion, out_planes, num_blocks, stride)
        self.cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 1),
            (6, 32, 2, 2),  # 10 -> 5
            (6, 64, 2, 1),
            (6, 96, 2, 1),
            (6, 160, 2, 2),  # 5 -> 3
            (6, 320, 1, 1),
        ]

        self.layers = self._make_layers(in_planes=32)

        # Final head
        self.conv2 = nn.Conv1d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(8, 1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for s in strides:
                layers.append(BlockV2_1D(in_planes, out_planes, expansion, s))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # 确保输入是 (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.gn2(self.conv2(out)))

        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

