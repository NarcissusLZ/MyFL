import torch, yaml
import torch.nn as nn


# 根据配置和硬件支持自动选择设备
def select_device(device_config):
    if device_config == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_config == 'mps' and torch.backends.mps.is_available():  # macOS Metal
        return torch.device('mps')
    else:
        return torch.device('cpu')


# 适配CIFAR-10的VGG配置：第一层96通道适合32x32输入
vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class CIFAR10_VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_VGG16, self).__init__()

        self.features = self._make_layers(vgg)
        # 添加自适应池化确保输出为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        # 使用传入的num_classes参数
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)  # 确保1x1输出
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = CIFAR10_VGG16()
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
    for name, param in CIFAR10_VGG16().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_kb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_kb:.2f} MB")