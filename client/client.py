import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time,math
from tqdm import tqdm
import numpy as np

class Client:

    def __init__(self, id, config, local_dataset, gpu_id=None):
        self.id = id
        self.config = config
        self.local_dataset = local_dataset
        self.gpu_id = gpu_id
        self.device = self._select_device(config['device'])
        self.distance_seed = id * 100 + 42
        self.distance_generator = np.random.RandomState(self.distance_seed)
        self.distance = self.generate_distance()

        # 通信参数
        self.tx_power = 0.1  # 发射功率 100mW = 0.1W
        self.frequency = 2.4e9  # 频率 2.4GHz
        self.bandwidth = 20e6  # 带宽 20MHz
        self.noise_power = self._calculate_noise_power()
        
        # 环境相关指数（Open field - LOS）
        self.alpha = 0.644
        self.beta = 1.043

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
            drop_last=True,
            num_workers=config.get('num_workers', 0)
        )

        # 计算SNR
        snr = self._calculate_snr()
        snr_db = 10 * math.log10(snr) if snr > 0 else -float('inf')
        self.packet_loss = self.calculate_packet_loss_rate(16)
        
        print(f"客户端 {self.id} 初始化完成, 设备: {self.device}, 数据量: {len(local_dataset)}")
        print(f"客户端 {self.id} 距离: {self.distance}m, SNR: {snr_db:.2f}dB")
        print(f"客户端 {self.id} 等待服务器下发模型")


    def set_distance_seed(self, seed):
        """设置距离分配的随机种子"""
        self.distance_seed = seed
        self.distance_generator = np.random.RandomState(seed)
        print(f"客户端 {self.id} 距离分配种子设置为: {seed}")

    def generate_distance(self, mean=25, std=8.33):
        """
        生成符合正态分布的距离值
        mean: 均值，默认50米
        std: 标准差，默认8.33（使得99.7%的值在1-50范围内）
        """
        if self.distance_generator is None:
            # 如果没有设置种子，使用默认种子
            self.distance_generator = np.random.RandomState(42)

        # 生成正态分布的距离值
        distance = self.distance_generator.normal(mean, std)

        # 限制在合理范围内（1-50米）
        distance = max(1, min(50, distance))

        return round(distance, 2)

    def _calculate_noise_power(self):
        """计算噪声功率（加性高斯白噪声）"""
        # 热噪声功率 = k * T * B
        # k = 1.38e-23 (玻尔兹曼常数)
        # T = 290K (室温)
        # B = 带宽
        k_boltzmann = 1.38e-23
        temperature = 290  # K
        noise_power = k_boltzmann * temperature * self.bandwidth
        return noise_power

    def _calculate_path_loss(self):
        """计算自由空间路径损耗"""
        # 自由空间路径损耗公式: L = (4πdf/c)²
        # 其中 d是距离，f是频率，c是光速
        c = 3e8  # 光速
        path_loss_linear = (4 * math.pi * self.distance * self.frequency / c) ** 2
        return path_loss_linear

    def _calculate_received_power(self):
        """计算接收功率"""
        path_loss = self._calculate_path_loss()
        received_power = self.tx_power / path_loss
        return received_power

    def _calculate_snr(self):
        """计算信噪比"""
        received_power = self._calculate_received_power()
        snr = received_power / self.noise_power
        return snr

    def _calculate_data_rate(self):
        """使用香农公式计算数据传输速度"""
        snr = self._calculate_snr()
        # 香农公式: C = B * log2(1 + SNR)
        data_rate = self.bandwidth * math.log2(1 + snr)
        return data_rate
    
    def calculate_packet_loss_rate(self, data_size_bytes):
        """
        根据论文公式计算丢包率
        p_m = α * s_m * e^(-β * SNR)
        其中 s_m 是数据大小（以字节为单位），α=0.644, β=1.043
        """
        snr = self._calculate_snr()
        
        # 使用论文公式计算丢包率
        packet_loss = self.alpha * data_size_bytes * math.exp(-self.beta * snr)
        
        # 确保丢包率在[0, 1]范围内
        packet_loss = max(0.0, min(1.0, packet_loss))
        
        return packet_loss

    def calculate_transmission_time(self, data_size_bytes):
        """计算单次传输时间（不考虑重传，由服务器端控制重传逻辑）"""
        data_rate = self._calculate_data_rate()  # bps
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
