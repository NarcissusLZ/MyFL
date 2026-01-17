import torch
import torch.nn as nn
import torch.nn.functional as F


# === 残差块 ===
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()

        # 第一层线性变换
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.gn1 = nn.GroupNorm(8, hidden_dim)  # 使用 GroupNorm 适应 FL
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第二层线性变换
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gn2 = nn.GroupNorm(8, hidden_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x  # 保存输入用于跳跃连接

        out = self.linear1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.gn2(out)

        # === 核心：残差连接 ===
        # 将原始输入直接加到输出上，防止梯度消失
        out = out + identity

        out = self.act2(out)
        out = self.dropout2(out)

        return out


# === 主模型 ===
class IoT23_ResNet(nn.Module):
    def __init__(self, input_dim=10, num_classes=5, hidden_dim=128, num_blocks=3):
        super(IoT23_ResNet, self).__init__()

        # 1. 输入投影层 (将 10 维特征映射到高维空间)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU()
        )

        # 2. 堆叠残差块
        # num_blocks=3 意味着网络深度会有 3*2 + 2 = 8 层左右，比之前的 MLP 深
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate=0.2)
            for _ in range(num_blocks)
        ])

        # 3. 输出分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # 确保输入是平铺的
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        out = self.input_layer(x)

        for block in self.blocks:
            out = block(out)

        out = self.classifier(out)
        return out


if __name__ == '__main__':
    # 测试一下
    model = IoT23_ResNet(input_dim=10, num_classes=5)
    print(model)
    x = torch.randn(32, 10)
    y = model(x)
    print("Output shape:", y.shape)