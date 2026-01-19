import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_IoT23(nn.Module):
    # 修改默认 input_dim 为 20，适配最新的特征工程（12基础+2衍生+6History = 20维）
    def __init__(self, num_classes=5, input_dim=20):
        super(MLP_IoT23, self).__init__()

        # 对于表格数据，MLP 比 CNN 稳定得多
        self.layer1 = nn.Linear(input_dim, 128)
        # 将 BatchNorm1d 替换为 GroupNorm (32 groups, 128 channels)
        self.bn1 = nn.GroupNorm(32, 128)

        self.layer2 = nn.Linear(128, 256)
        # 将 BatchNorm1d 替换为 GroupNorm (32 groups, 256 channels)
        self.bn2 = nn.GroupNorm(32, 256)

        self.layer3 = nn.Linear(256, 128)
        # 将 BatchNorm1d 替换为 GroupNorm (32 groups, 128 channels)
        self.bn3 = nn.GroupNorm(32, 128)

        self.head = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        # 如果输入是 (Batch, 1, 10) 或 (Batch, 10)，统一展平
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.layer3(x)))

        return self.head(x)
