import torch
import torch.nn as nn
import torch.nn.functional as F


class IoT23_MLP(nn.Module):
    def __init__(self, input_dim=16, num_classes=2):
        """
        适用于 IoT-23 数据集的轻量级 MLP
        Args:
            input_dim: 输入特征数量 (需要在 getdata.py 中与预处理逻辑对应)
            num_classes: 分类数量 (例如 2: Malicious/Benign, 或更多: DDoS/Botnet/Normal)
        """
        super(IoT23_MLP, self).__init__()

        # 隐藏层设计：逐层压缩特征
        # IoT 设备计算能力弱，这里保持网络较浅但足够宽
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.classifier = nn.Linear(64, num_classes)

        # Dropout 用于防止过拟合，增强联邦学习中的鲁棒性
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 确保输入是 Flatten 的 (Batch_Size, Input_Dim)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.layer3(x)))

        out = self.classifier(x)
        return out


if __name__ == '__main__':
    # 测试模型结构
    model = IoT23_MLP(input_dim=16, num_classes=5)
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(32, 16)  # Batch size 32, 16 features
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")