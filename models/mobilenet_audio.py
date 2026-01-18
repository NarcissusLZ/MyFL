# models/mobilenet_audio.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Part 1: Audio / 2D Image Models (Original)
# ==========================================

class BlockV1(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(BlockV1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetV1_Audio(nn.Module):
    def __init__(self, num_classes=35, input_channels=1):
        super(MobileNetV1_Audio, self).__init__()
        # 针对 40x81 这样的小尺寸输入，第一层 Stride 设为 1，避免信息丢失过快
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 定义层结构：(输入通道, 输出通道, stride)
        self.layers = self._make_layers(in_planes=32)

        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        # 配置格式: (out_planes, stride)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BlockV2(nn.Module):
    '''Inverted Residual Block'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(BlockV2, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
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
        self.bn1 = nn.BatchNorm2d(32)

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
        self.bn2 = nn.BatchNorm2d(1280)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ==========================================
# Part 2: IoT-23 / 1D Tabular Models (New)
# ==========================================

class BlockV1_1D(nn.Module):
    '''Depthwise conv + Pointwise conv (1D version for Tabular Data)'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(BlockV1_1D, self).__init__()
        # 使用 Conv1d 处理序列特征
        self.conv1 = nn.Conv1d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv2 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetV1_IoT23(nn.Module):
    '''Adapted MobileNet V1 for low-dimensional tabular data (e.g. 10 features)'''

    def __init__(self, num_classes=5, input_dim=10):
        super(MobileNetV1_IoT23, self).__init__()

        # 输入维度: (Batch, 1, InputDim)
        self.input_dim = input_dim

        # Stem Layer: 1通道 -> 32通道
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        # 定义 1D 网络结构
        # 注意：由于 input_dim 很小 (10)，我们不能使用太多的 stride=2，
        # 否则特征维度会迅速变为 1 或 0 导致报错。
        self.layers = self._make_layers(in_planes=32)

        self.linear = nn.Linear(512, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        # (out_planes, stride)
        cfg = [
            (64, 1),
            (128, 2),  # 10 -> 5
            (128, 1),
            (256, 1),  # 保持 5
            (256, 1),
            (512, 2),  # 5 -> 3
            (512, 1),
            (512, 1),
            # 这里的层数比标准版少，防止过拟合和特征消失
        ]

        for out_planes, stride in cfg:
            layers.append(BlockV1_1D(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, InputDim) -> 需要变成 (Batch, 1, InputDim)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)

        # Global Average Pooling (1D)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BlockV2_1D(nn.Module):
    '''Inverted Residual Block (1D version)'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(BlockV2_1D, self).__init__()
        self.stride = stride
        planes = expansion * in_planes

        # 1x1 升维
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        # 3x3 深度卷积
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        # 1x1 降维
        self.conv3 = nn.Conv1d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride == 1:
            if self.shortcut:
                out = out + self.shortcut(x)
            else:
                out = out + x
        return out


class MobileNetV2_IoT23(nn.Module):
    '''Adapted MobileNet V2 for low-dimensional tabular data'''

    def __init__(self, num_classes=5, input_dim=10):
        super(MobileNetV2_IoT23, self).__init__()

        self.input_dim = input_dim
        # Stem
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        # Config: (expansion, out_planes, num_blocks, stride)
        # 针对 Input Dim = 10 的简化版配置
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
        self.bn2 = nn.BatchNorm1d(1280)
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

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))

        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out