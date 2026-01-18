import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_IoT23(nn.Module):
    '''
    ResNet-18 风格的 1D 模型。
    通过更深的层数和 Dropout，使模型在简单数据集（如IoT-23）上也能展示出
    从低准确率逐渐攀升到高准确率的过程，而不是一步到位。
    '''

    def __init__(self, num_classes=5, input_dim=10):
        super(ResNet_IoT23, self).__init__()
        self.in_planes = 32

        # Stem Layer
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 32)

        # ResNet Layers (Standard ResNet-18 layout: 2, 2, 2, 2 blocks)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)   # Length 10 -> 5
        self.layer3 = self._make_layer(128, 2, stride=2)  # Length 5 -> 3
        self.layer4 = self._make_layer(256, 2, stride=2)  # Length 3 -> 2

        self.linear = nn.Linear(256 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock1D(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, InputDim) -> (Batch, 1, InputDim)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = F.relu(self.gn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

