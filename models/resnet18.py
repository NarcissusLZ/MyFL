import torch, yaml
import torch.nn as nn
import torch.nn.functional as F


# 根据配置和硬件支持自动选择设备
def select_device(device_config):
    if device_config == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_config == 'mps' and torch.backends.mps.is_available():  # macOS Metal
        return torch.device('mps')
    else:
        return torch.device('cpu')


class BasicBlock(nn.Module):
    """ResNet的基本块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFAR10_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_ResNet18, self).__init__()
        self.in_planes = 64

        # 针对CIFAR-10的32x32输入优化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet18层配置: [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CIFAR100_ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR100_ResNet18, self).__init__()
        self.in_planes = 64

        # 针对CIFAR-100的32x32输入优化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet18层配置: [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # CIFAR-100优化的分类器：添加dropout防止过拟合
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * BasicBlock.expansion, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class GoogleSpeech_ResNet18(nn.Module):
    def __init__(self, num_classes=35):
        """
        适用于 Google Speech Commands 的 ResNet-18
        :param num_classes: 默认为12 (10个核心命令 + Unknown + Silence)
        """
        super(GoogleSpeech_ResNet18, self).__init__()
        self.in_planes = 64

        # 关键修改：输入通道为 1 (单通道频谱图)，而不是 3 (RGB)
        # 尺寸保持 32x32 优化的配置 (kernel=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet18 标准层配置
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 全局平均池化 (自适应不同尺寸的频谱图输入)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 如果输入是 [Batch, Height, Width]，自动增加 Channel 维度
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 用于本地读取测试
if __name__ == '__main__':
    model = GoogleSpeech_ResNet18()
    # 获取配置参数
    with open("../config.yaml", 'r') as f:
        conf = yaml.safe_load(f)

    device_config = conf.get('device').lower()
    device = select_device(device_config)

    model = model.to(device)  # 统一迁移到目标设备
    print(model)
    print(next(model.parameters()).device)

    # Calculate and print parameters for each layer
    total_params = 0
    for name, param in GoogleSpeech_ResNet18().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_mb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_mb:.2f} MB")

    # 测试模型输出
    test_input = torch.randn(1, 1, 40, 98).to(device)
    with torch.no_grad():
        output = model(test_input)
        print(f"Output shape: {output.shape}")