import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_IoT23(nn.Module):
    def __init__(self, num_classes=5, input_dim=10):
        super(MLP_IoT23, self).__init__()

        # 对于表格数据，MLP 比 CNN 稳定得多
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.layer2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

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