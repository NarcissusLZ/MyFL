import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_IoT23(nn.Module):
    def __init__(self, num_classes=5, input_dim=19):
        super(MLP_IoT23, self).__init__()

        # 第一层: 19 -> 256 (加宽)
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.GroupNorm(32, 256)

        # 第二层: 256 -> 512 (加宽)
        self.layer2 = nn.Linear(256, 512)
        self.bn2 = nn.GroupNorm(32, 512)

        # 第三层: 512 -> 256 (加宽)
        self.layer3 = nn.Linear(512, 256)
        self.bn3 = nn.GroupNorm(32, 256)

        # === 关键修改在这里 ===
        # 以前是 nn.Linear(128, num_classes)
        # 现在必须改成 256，因为 layer3 输出是 256
        self.head = nn.Linear(256, num_classes)
        # ======================

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.layer3(x)))
        # 这里不需要 dropout，通常最后一层前直接输出或者只在 ReLU 后接 dropout

        return self.head(x)
