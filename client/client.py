import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time,math
from tqdm import tqdm
import numpy as np
import json

class Client:

    def __init__(self, id, config, local_dataset, gpu_id=None):
        self.id = id
        self.config = config
        self.local_dataset = local_dataset
        self.gpu_id = gpu_id
        self.device = self._select_device(config['device'])
        self.bandwidth = 2e6
        self.distance = self.get_distance(self.id)

        self.packet_loss = self.get_plr(self.id)

        self.snr = self.get_snr(self.id)

        # 传输统计
        self.total_transmission_time = 0.0
        self.transmission_attempts = 0

        # 声明模型，等待下发
        self.model = None
        self.optimizer = None

        # 数据加载器
        self.train_loader = DataLoader(
            local_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True,  # 丢弃最后一个不完整的batch
            num_workers=config.get('num_workers', 0)
        )

        print(f"客户端 {self.id} 初始化完成, 设备: {self.device}, 数据量: {len(local_dataset)}")
        print(f"客户端 {self.id} 距离: {self.distance}m, 丢包率: {self.packet_loss:.3f}")
        print(f"客户端 {self.id} 等待服务器下发模型")

    def load_communication_data(self,filename="communication_results.json"):
        """加载JSON文件数据"""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_plr(self,device_id, filename="communication_results.json"):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for device in data['devices']:
            if device['id'] == device_id:
                return device['snr']
        return None

    def get_distance(self,device_id, filename="communication_results.json"):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for device in data['devices']:
            if device['id'] == device_id:
                return device['distance']
        return None

    def get_snr(self,device_id, filename="communication_results.json"):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for device in data['devices']:
            if device['id'] == device_id:
                return device['plr']
        return None

    def calculate_data_rate(self):
        """使用香农公式计算数据传输速度"""
        snr = self.snr
        # 香农公式: C = B * log2(1 + SNR)
        data_rate = self.bandwidth * math.log2(1 + snr)
        return data_rate

    def calculate_transmission_time(self, data_size_bytes):
        """计算单次传输时间（不考虑重传，由服务器端控制重传逻辑）"""
        data_rate = self.calculate_data_rate()  # bps
        transmission_time = (data_size_bytes * 8) / data_rate  # 转换为比特再除以速率
        return transmission_time, 1  # 返回传输时间和传输次数(1)

    def _select_device(self, device_config):
        """选择设备，支持指定GPU ID"""
        if device_config == 'cuda' and torch.cuda.is_available():
            if self.gpu_id is not None:
                return torch.device(f'cuda:{self.gpu_id}')
            return torch.device('cuda')
        elif device_config == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def receive_model(self, model_class, model_state_dict=None):
        """接收服务器下发的模型"""
        # 根据服务器指定的模型类创建模型
        self.model = model_class()
        self.model.to(self.device)

        # 如果提供了状态字典，则加载
        if model_state_dict is not None:
            # 确保状态字典在正确的设备上
            device_state_dict = {}
            for key, value in model_state_dict.items():
                device_state_dict[key] = value.to(self.device)
            self.model.load_state_dict(device_state_dict)

        # 初始化优化器（需要在模型创建后）
        self._initialize_optimizer()

        print(f"客户端 {self.id} 已接收模型: {self.model.__class__.__name__}")

    def _initialize_optimizer(self):
        """初始化优化器"""
        if self.model is None:
            raise ValueError(f"客户端 {self.id} 模型未初始化，无法创建优化器")

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['lr'],
            momentum=self.config['momentum']
        )

    def local_train(self):
        """在本地数据集上训练模型"""

        # 检查数据加载器是否为空或不足一个 batch
        if len(self.train_loader) == 0:
            print(f"客户端 {self.id} 的数据加载器为空，跳过训练")
            return None, 0
        print(f"客户端 {self.id} 开始本地训练")
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
                leave=False  # 修改为False，避免进度条累积
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

        # 返回更新后的模型状态字典和样本数量
        return self.model.state_dict(), len(self.local_dataset)
