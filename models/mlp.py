import torch
import torch.nn as nn
import torch.nn.functional as F


class IoT23_MLP(nn.Module):
    def __init__(self, input_dim=10, num_classes=5):
        super(IoT23_MLP, self).__init__()

        # 加宽网络： 64 -> 256
        self.layer1 = nn.Linear(input_dim, 256)
        self.gn1 = nn.GroupNorm(32, 256)  # GroupNorm

        self.layer2 = nn.Linear(256, 512)  # 中间层加宽到 512
        self.gn2 = nn.GroupNorm(64, 512)

        self.layer3 = nn.Linear(512, 256)
        self.gn3 = nn.GroupNorm(32, 256)

        self.layer4 = nn.Linear(256, 128)  # 增加一层
        self.gn4 = nn.GroupNorm(16, 128)

        self.classifier = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if x.dim() > 2: x = x.view(x.size(0), -1)

        x = F.relu(self.gn1(self.layer1(x)))
        x = self.dropout(x)

        x = F.relu(self.gn2(self.layer2(x)))
        x = self.dropout(x)

        x = F.relu(self.gn3(self.layer3(x)))
        x = self.dropout(x)

        x = F.relu(self.gn4(self.layer4(x)))

        return self.classifier(x)

if __name__ == '__main__':
    # 测试模型结构
    model = IoT23_MLP(input_dim=10, num_classes=5)
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(32, 16)  # Batch size 32, 16 features
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")