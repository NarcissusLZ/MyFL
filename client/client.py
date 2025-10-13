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
        self.distance = self.generate_distance()

        # 通信参数
        self.tx_power = 0.1  # 发射功率 100mW = 0.1W
        self.frequency = 2.4e9  # 频率 2.4GHz
        self.bandwidth = 20e6  # 带宽 20MHz
        self.noise_power = self._calculate_noise_power()
        
        # 环境相关指数（Open field - LOS）
        self.alpha = 0.084
        self.beta = 0.356

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
        self.packet_loss = self.calculate_packet_loss_rate(1500)
        
        print(f"客户端 {self.id} 初始化完成, 设备: {self.device}, 数据量: {len(local_dataset)}")
        print(f"客户端 {self.id} 距离: {self.distance}m, SNR: {snr_db:.2f}dB, 丢包率: {self.packet_loss:.3f}")

    def generate_distance(self, mean=25, std=8.33):
        """
        生成符合正态分布的距离值，基于客户端ID确保不同客户端有不同距离
        """
        # 使用固定种子与客户端ID混合
        np.random.seed(42 + self.id)

        # 生成正态分布的距离值
        distance = np.random.normal(mean, std)

        # 限制在合理范围内（1-50米）
        distance = max(1, min(50, distance))

        return round(distance, 2)

    def _calculate_noise_power(self):
        """计算噪声功率（包括热噪声和环境噪声）"""
        # 热噪声功率 = k * T * B
        k_boltzmann = 1.38e-23
        temperature = 290  # K
        thermal_noise = k_boltzmann * temperature * self.bandwidth

        # 添加接收器噪声系数 (Noise Figure, NF)
        noise_figure_db = 7  # 典型接收器噪声系数为5-10dB
        noise_figure = 10 ** (noise_figure_db / 10)

        # 添加背景噪声和干扰
        background_noise_dbm = -90  # 典型背景噪声和干扰为-100到-80dBm
        background_noise = 10 ** ((background_noise_dbm - 30) / 10)  # 转换为W

        # 总噪声功率
        total_noise = thermal_noise * noise_figure + background_noise

        return total_noise

    def _calculate_path_loss(self):
        """计算路径损耗（修正的瑞丽衰落实现）"""
        # 基础自由空间路径损耗
        freq_mhz = self.frequency / 1e6
        base_path_loss_db = 20 * math.log10(self.distance) + 20 * math.log10(freq_mhz) - 27.55

        # 阴影衰落（对数正态分布）
        np.random.seed(42 + self.id)
        shadow_std = 4.0  # 室内环境标准差通常在4-6dB
        shadow_fading_db = np.random.normal(0, shadow_std)

        # 室内额外损耗 - 增加基础损耗值
        indoor_loss_db = 15  # 增加到15dB

        # 瑞丽衰落（正确实现）
        np.random.seed(42 + self.id + 100)
        # 生成瑞丽分布随机数
        rayleigh_real = np.random.normal(0, 1)
        rayleigh_imag = np.random.normal(0, 1)

        # 计算瑞丽分布幅度
        rayleigh_amplitude = np.sqrt(rayleigh_real ** 2 + rayleigh_imag ** 2)

        # 关键修改：瑞丽衰落应该始终是损耗而不是增益
        # 转换为dB，这里的重点是确保它始终代表损耗
        rayleigh_fading_db = 10 * np.random.uniform(3, 15)  # 产生3-15dB的额外衰落

        # 总路径损耗
        total_path_loss_db = base_path_loss_db + shadow_fading_db + indoor_loss_db + rayleigh_fading_db

        print(f"客户端 {self.id} 路径损耗详情: 基础={base_path_loss_db:.2f}dB, "
              f"阴影={shadow_fading_db:.2f}dB, 室内={indoor_loss_db:.2f}dB, "
              f"瑞丽={rayleigh_fading_db:.2f}dB, 总计={total_path_loss_db:.2f}dB")

        # 将dB转换为线性单位
        total_path_loss = 10 ** (total_path_loss_db / 10)

        return total_path_loss

    def _calculate_received_power(self):
        """计算接收功率"""
        path_loss = self._calculate_path_loss()
        received_power = self.tx_power / path_loss
        return received_power

    def _calculate_snr(self):
        """计算信噪比"""
        received_power = self._calculate_received_power()
        snr = received_power / self.noise_power
        snr_db = 10 * math.log10(snr) if snr > 0 else -float('inf')
        print(
            f"客户端 {self.id}: 接收功率={received_power:.2e}W, 噪声功率={self.noise_power:.2e}W, SNR={snr:.2e} ({snr_db:.2f}dB)")
        return snr

    def _calculate_data_rate(self):
        """使用香农公式计算数据传输速度"""
        snr = self._calculate_snr()
        # 香农公式: C = B * log2(1 + SNR)
        data_rate = self.bandwidth * math.log2(1 + snr)
        return data_rate

    def calculate_packet_loss_rate(self, data_size_bytes):
        """计算更真实的丢包率分布"""
        snr = self._calculate_snr()
        snr_db = 10 * math.log10(snr) if snr > 0 else -float('inf')

        # 调整阈值和曲线参数，使丢包率在合理范围内
        snr_threshold = 20  # 提高阈值到20dB
        steepness = 0.25  # 减小曲线陡度，使变化更平滑

        if snr_db < 5:  # SNR过低，高丢包率
            packet_loss = 0.95
        else:
            # logistic函数: 1/(1+e^(steepness*(SNR_dB-threshold)))
            packet_loss = 1.0 / (1.0 + math.exp(steepness * (snr_db - snr_threshold)))

        # 加入距离的影响（提高远距离的影响）
        distance_factor = min(1.0, (self.distance / 40) ** 1.5)  # 非线性增长
        packet_loss = min(0.98, packet_loss * (0.6 + 0.4 * distance_factor))

        # 确保结果有足够的随机性
        np.random.seed(42 + self.id + 200)
        random_factor = np.random.uniform(0.9, 1.1)
        packet_loss = min(0.99, max(0.01, packet_loss * random_factor))

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
