import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

class Client:
    def __init__(self, id, config, local_dataset):
        self.id = id
        self.config = config
        self.local_dataset = local_dataset
        self.device = self._select_device(config['device'])

        # 初始化模型
        self.model = self._create_model()
        self.model.to(self.device)

        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['lr'],
            momentum=config['momentum']
        )

        # 数据加载器
        self.train_loader = DataLoader(
            local_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True,  # 丢弃最后一个不完整的batch
            num_workers=config.get('num_workers', 0)
        )

        print(f"客户端 {self.id} 初始化完成, 设备: {self.device}, 数据量: {len(local_dataset)}")

    def _select_device(self, device_config):
        """选择设备"""
        if device_config == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_config == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _create_model(self):
        """创建与服务器相同结构的模型"""
        # 根据模型配置创建模型
        if self.config['model'] == 'CIFAR10_VGG16':
            from models.vgg16 import CIFAR10_VGG16
            return CIFAR10_VGG16()
        else:
            raise ValueError(f"未知模型: {self.config['model']}")

    def get_model(self, model_state_dict):
        """接收服务器下发的全局模型状态字典"""
        # 加载状态字典到本地模型
        self.model.load_state_dict(model_state_dict)
        # 确保模型在正确设备上
        self.model.to(self.device)
        print(f"客户端 {self.id} 已接收全局模型")

    def local_train(self):
        """在本地数据集上训练模型"""
        print(f"客户端 {self.id} 开始本地训练...")
        start_time = time.time()

        # 设置模型为训练模式
        self.model.train()

        # 训练多个epoch
        for epoch in range(self.config['local_epochs']):
            epoch_loss = 0.0

            # 创建进度条
            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"客户端 {self.id} | Epoch {epoch + 1}/{self.config['local_epochs']}",
                ncols=100,
                leave=True
            )

            for batch_idx, (data, target) in progress_bar:
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)

                # 反向传播
                loss.backward()

                # 梯度裁剪（如果配置）
                if 'max_grad_norm' in self.config:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['max_grad_norm']
                    )

                # 更新参数
                self.optimizer.step()

                epoch_loss += loss.item()

                # 更新进度条显示
                current_avg_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{current_avg_loss:.4f}'
                })

            # Epoch完成后的信息
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"客户端 {self.id} | Epoch {epoch + 1} 完成 | 平均损失: {avg_epoch_loss:.4f}")

        training_time = time.time() - start_time
        print(f"客户端 {self.id} 本地训练完成 | 耗时: {training_time:.2f}秒")

        # 返回更新后的模型状态字典
        return self.model.state_dict()

