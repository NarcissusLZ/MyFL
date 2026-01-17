import torch
import torch.nn as nn
import torch.nn.functional as F


class IoT23_MLP(nn.Module):
    def __init__(self, input_dim=10, num_classes=5):
        super(IoT23_MLP, self).__init__()

        # Layer 1
        self.layer1 = nn.Linear(input_dim, 64)
        # 修改：使用 GroupNorm (将 64 个通道分成 8 组进行归一化)
        # 或者直接注释掉 BatchNorm 也可以
        self.gn1 = nn.GroupNorm(8, 64)

        # Layer 2
        self.layer2 = nn.Linear(64, 128)
        self.gn2 = nn.GroupNorm(16, 128)

        # Layer 3
        self.layer3 = nn.Linear(128, 64)
        self.gn3 = nn.GroupNorm(8, 64)

        self.classifier = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Layer 1
        x = self.layer1(x)
        x = self.gn1(x)  # 使用 GroupNorm
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.layer2(x)
        x = self.gn2(x)  # 使用 GroupNorm
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.layer3(x)
        x = self.gn3(x)  # 使用 GroupNorm
        x = F.relu(x)

        out = self.classifier(x)
        return out


if __name__ == '__main__':
    # 测试模型结构
    model = IoT23_MLP(input_dim=10, num_classes=5)
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(32, 16)  # Batch size 32, 16 features
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")