# models/mobilenet_audio.py

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # 相比标准 ImageNet 版本，这里去掉了部分 stride=2 以适应较小的输入尺寸
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
        # 自适应平均池化，不管前面尺寸剩下多少，都变成 1x1
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
        # 当 stride=1 且维度匹配时使用残差连接
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
            (6, 24, 2, 1),  # 保持尺寸
            (6, 32, 3, 2),  # 20x40
            (6, 64, 4, 2),  # 10x20
            (6, 96, 3, 1),
            (6, 160, 3, 2),  # 5x10
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
        # Global Average Pooling
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out