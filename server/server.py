import torch
import numpy as np
from torch.utils.data import DataLoader
import copy
import torch.nn as nn
import random
import io
import math


class LayerMetricCalculator:
    """
    联邦学习版：用权重变化统计替代梯度敏感度
    """

    def __init__(self, model):
        print("初始化 LayerMetricCalculator: 正在备份初始权重...")
        self.prev_weights = {}
        self.movement_ema = {}  # 指数移动平均（用于平滑变化量）

        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                self.prev_weights[name] = param.data.clone().detach().cpu()
                self.movement_ema[name] = 0.0

    def update_prev_weights(self, model):
        """更新上一轮权重记录"""
        for name, param in model.named_parameters():
            if name in self.prev_weights:
                self.prev_weights[name] = param.data.clone().detach().cpu()

    def get_dual_metrics(self, model):
        """
        联邦学习版本：用权重变化统计替代梯度

        关键修改：
        - 因子1: 当前轮权重变化 (W_t - W_{t-1})
        - 因子2: 历史变化的EMA（替代梯度敏感度）
        """
        metrics_data = []

        for name, param in model.named_parameters():
            if 'weight' not in name or param.dim() <= 1:
                continue

            # 因子1: 当前轮次权重变化
            movement = 0.0
            if name in self.prev_weights:
                prev_w = self.prev_weights[name].to(param.device)
                movement = torch.norm(param.data - prev_w, p=2).item()

            # 因子2: 用EMA平滑的权重变化替代梯度
            # 物理意义：持续变化大 → 对损失敏感 → 类似高梯度
            self.movement_ema[name] = (
                    0.8 * self.movement_ema[name] + 0.2 * movement
            )
            grad_approx = self.movement_ema[name]

            metrics_data.append({
                'name': name,
                'movement': movement,  # 当前变化
                'grad': grad_approx  # ← 用平滑变化量替代梯度
            })

        return metrics_data


class Server:
    def __init__(self, config, test_dataset):
        self.config = config
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else self._select_device(config['device'])

        # 设置随机种子以确保实验可重复
        self.random_seed = config.get('random_seed', 42)
        self._set_random_seeds()

        # 初始化全局模型
        self.global_model = self.init_model()
        self.global_model.to(self.device)

        # 准备测试数据集
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=False
        )

        # 添加通信量统计
        # self.total_down_communication = 0  # 下行通信总量（字节）
        self.total_up_communication = 0  # 上行通信总量（字节）
        # self.round_down_communication = 0  # 当前轮次下行通信量
        self.round_up_communication = 0  # 当前轮次上行通信量

        # 添加分层流量统计
        self.total_robust_communication = 0  # 鲁棒层总通信量（字节）
        self.total_critical_communication = 0  # 关键层总通信量（字节）
        self.round_robust_communication = 0  # 当前轮次鲁棒层通信量
        self.round_critical_communication = 0  # 当前轮次关键层通信量

        self.communication_history = []  # 每轮通信量记录

        # 添加传输时间统计
        self.round_transmission_times = {}  # 当前轮次各客户端传输时间
        self.max_transmission_times = []  # 每轮最大传输时间记录

        # 添加传输次数统计
        self.total_robust_transmission_count = 0  # 鲁棒层总传输次数
        self.total_critical_transmission_count = 0  # 关键层总传输次数
        self.round_robust_transmission_count = 0  # 当前轮次鲁棒层传输次数
        self.round_critical_transmission_count = 0  # 当前轮次关键层传输次数

        # 记录聚合权重
        self.client_weights = {}

        # Gilbert-Elliott模型参数初始化
        self.gilbert_elliott_states = {}  # 每个客户端的网络状态
        self.client_random_generators = {}  # 每个客户端的独立随机数生成器 - 在这里初始化
        self._init_gilbert_elliott_params()

        # 添加动态分层配置参数
        self.use_dynamic_layer_classification = config.get('use_dynamic_layer_classification', False)
        self.critical_ratio = config.get('critical_ratio', 0.5)
        self.grad_beta = config.get('grad_beta', 1.0)

        # 初始化指标计算器
        self.use_dynamic_layer_classification = config.get('use_dynamic_layer_classification', False)
        if self.use_dynamic_layer_classification:
            self.metric_calculator = LayerMetricCalculator(self.global_model)
        else:
            self.metric_calculator = None

        print(f"服务器初始化完成, 设备: {self.device}")

    def _set_random_seeds(self):
        """设置随机种子以确保实验可重复性"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        print(f"已设置随机种子: {self.random_seed}")

    def _init_gilbert_elliott_params(self):
        """初始化Gilbert-Elliott模型状态字典"""
        # 只初始化状态字典，不再设置全局丢包率参数
        self.gilbert_elliott_states = {}  # 每个客户端的网络状态
        self.client_random_generators = {}  # 每个客户端的独立随机数生成器
        print("已初始化Gilbert-Elliott模型状态字典，将使用客户端各自的丢包率")

    def _get_client_random_generator(self, client_id):
        """为每个客户端获取或创建独立的随机数生成器"""
        if client_id not in self.client_random_generators:
            # 基于随机种子和客户端ID创建确定性的种子
            client_seed = self.random_seed + hash(str(client_id)) % 10000
            self.client_random_generators[client_id] = random.Random(client_seed)
            print(f"为客户端 {client_id} 创建随机数生成器，种子: {client_seed}")
        return self.client_random_generators[client_id]

    def get_model_size(self):
        """计算模型参数大小（字节）"""
        size_bytes = 0
        for param in self.global_model.parameters():
            # 每个参数元素占用4字节（float32）
            size_bytes += param.nelement() * 4
        return size_bytes

    def init_model(self):
        print("服务器开始初始化模型")
        if self.config['model'] == 'VGG16' and self.config['dataset'] == 'cifar10':
            from models.vgg16 import CIFAR10_VGG16
            return CIFAR10_VGG16(num_classes=10)
        elif self.config['model'] == 'VGG16' and self.config['dataset'] == 'cifar100':
            from models.vgg16 import CIFAR100_VGG16
            return CIFAR100_VGG16(num_classes=100)
        elif self.config['model'] == 'RESNET18' and self.config['dataset'] == 'cifar10':
            from models.resnet18 import CIFAR10_ResNet18
            return CIFAR10_ResNet18(num_classes=10)
        elif self.config['model'] == 'RESNET18' and self.config['dataset'] == 'cifar100':
            from models.resnet18 import CIFAR100_ResNet18
            return CIFAR100_ResNet18(num_classes=100)
        elif self.config['model'] == 'RESNET50' and self.config['dataset'] == 'cifar10':
            from models.resnet50 import CIFAR10_ResNet50
            return CIFAR10_ResNet50(num_classes=10)
        elif self.config['model'] == 'RESNET50' and self.config['dataset'] == 'cifar100':
            from models.resnet50 import CIFAR100_ResNet50
            return CIFAR100_ResNet50(num_classes=100)
        else:
            raise ValueError(f"未知模型: {self.config['model']}")

    def _select_device(self, device_config):
        """选择设备"""
        if device_config == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_config == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def broadcast_model(self, selected_clients):
        """向选中的客户端广播最新的全局模型参数"""
        print("服务器广播全局模型参数")

        global_state_dict = copy.deepcopy(self.global_model.state_dict())
        model_class = self.global_model.__class__  # 获取模型类

        for client in selected_clients:
            client.receive_model(model_class, global_state_dict)

        print(f"已向 {len(selected_clients)} 个客户端广播模型参数")

    def select_clients(self, all_clients, fraction=0.1):
        """选择参与本轮训练的客户端"""
        num_selected = max(1, int(fraction * len(all_clients)))
        selected_clients = np.random.choice(
            all_clients,
            num_selected,
            replace=False
        )
        print(f"本轮选中 {num_selected} 个客户端")
        return selected_clients

    def _gilbert_elliott_packet_loss(self, client):
        """使用Gilbert-Elliott模型判断单个数据包是否丢包"""
        client_id = client.id
        client_loss_rate = client.packet_loss

        if client_loss_rate == 0:
            return False  # 没有丢包率，不丢包

        client_rng = self._get_client_random_generator(client_id)

        if client_id not in self.gilbert_elliott_states:
            self.gilbert_elliott_states[client_id] = 0 if client_rng.random() > client_loss_rate else 1

        current_state = self.gilbert_elliott_states[client_id]
        good_to_bad = 0.5 * client_loss_rate
        bad_to_good = 0.5 - good_to_bad
        rand = client_rng.random()

        if current_state == 0:  # 好状态
            if rand <= good_to_bad:
                self.gilbert_elliott_states[client_id] = 1
                return True  # 丢包
            return False  # 不丢包
        else:  # 坏状态
            if rand <= 1 - bad_to_good:
                return True  # 丢包
            self.gilbert_elliott_states[client_id] = 0
            return False  # 不丢包

    def get_ltq_strategy_phase(self, current_round, current_accuracy=None):
        """
        确定LTQ当前应该采用的策略阶段
        返回: 'early', 'middle', 'late'
        """
        total_rounds = self.config['num_rounds']

        # 方案1: 纯轮次划分
        if self.config['ltq_phase_method'] == 'rounds':
            # 从配置文件读取轮次比例
            early_ratio = self.config['ltq_early_ratio']
            middle_ratio = self.config['ltq_middle_ratio']
            # 计算最小中期和后期轮次
            min_middle_round = int(total_rounds * early_ratio)
            min_late_round = int(total_rounds * middle_ratio)

            if current_round <= min_middle_round:
                return 'early'
            elif current_round <= min_late_round:
                return 'middle'
            else:
                return 'late'

        # 方案2: 基于精度的自适应划分
        elif self.config['ltq_phase_method'] == 'accuracy':
            if current_accuracy is None:
                return 'early'  # 如果没有精度信息，默认早期

            early_acc_threshold = self.config['ltq_early_acc_threshold']
            middle_acc_threshold = self.config['ltq_middle_acc_threshold']

            if current_accuracy < early_acc_threshold:
                return 'early'
            elif current_accuracy < middle_acc_threshold:
                return 'middle'
            else:
                return 'late'

        # 方案3: 混合策略（轮次 + 精度）
        else:  # 'hybrid'
            # 从配置文件读取轮次比例
            early_ratio = self.config['ltq_early_ratio']
            middle_ratio = self.config['ltq_middle_ratio']
            # 计算最小中期和后期轮次
            min_middle_round = int(total_rounds * early_ratio)
            min_late_round = int(total_rounds * middle_ratio)

            # 从配置文件读取精度阈值
            early_acc_threshold = self.config['ltq_early_acc_threshold']
            middle_acc_threshold = self.config['ltq_middle_acc_threshold']

            if current_round < min_middle_round or (
                    current_accuracy is not None and current_accuracy < early_acc_threshold):
                return 'early'
            elif current_round < min_late_round or (
                    current_accuracy is not None and current_accuracy < middle_acc_threshold):
                return 'middle'
            else:
                return 'late'

    def _serialize_and_packetize(self, state_dict, packet_size=1500):
        """将模型状态字典序列化并切分为数据包"""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data_bytes = buffer.getvalue()

        packets = []
        for i in range(0, len(data_bytes), packet_size):
            packets.append(data_bytes[i:i + packet_size])

        return packets, data_bytes

    def _get_packet_layer_type_map(self, state_dict, total_size, packet_size=1500):
        """创建从数据包索引到层类型的映射（使用动态分层）"""
        # 获取动态分层结果
        critical_layers, robust_layers = self.classify_layers_dual_factor()

        packet_map = {}
        current_pos = 0

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        ordered_state_dict = torch.load(buffer)

        temp_buffer = io.BytesIO()
        for key, param in ordered_state_dict.items():
            # 判断层类型
            is_robust = key in robust_layers
            layer_type = 'robust' if is_robust else 'critical'

            torch.save(param, temp_buffer)
            param_size = temp_buffer.tell()
            temp_buffer.seek(0)
            temp_buffer.truncate(0)

            start_packet = current_pos // packet_size
            end_packet = (current_pos + param_size - 1) // packet_size

            for i in range(start_packet, end_packet + 1):
                packet_map[i] = layer_type

            current_pos += param_size

        num_packets = math.ceil(total_size / packet_size)
        for i in range(num_packets):
            if i not in packet_map:
                packet_map[i] = 'critical'

        return packet_map

    def receive_local_model(self, client, model_state_dict, num_samples, current_round=None, current_accuracy=None):
        """接收客户端上传的模型更新，并基于数据包模拟丢包和重传"""
        if model_state_dict is None:
            print(f"客户端 {client.id} 上传的模型状态字典为空，跳过更新")
            return False

        client_id = client.id
        transport_type = self.config.get('Transport', 'TCP')
        packet_size = 1500  # MTU大小
        max_retries = 16

        # 1. 序列化并分包
        packets, original_data_bytes = self._serialize_and_packetize(model_state_dict, packet_size)
        num_packets = len(packets)
        total_size = len(original_data_bytes)

        # 2. 获取旧的全局模型数据作为回退
        _, fallback_data_bytes = self._serialize_and_packetize(self.global_model.state_dict(), packet_size)

        # 3. 确定每个包的类型（用于LTQ中期策略）
        packet_layer_type_map = {}
        if transport_type == 'LTQ':
            packet_layer_type_map = self._get_packet_layer_type_map(model_state_dict, total_size, packet_size)

        # 4. 模拟传输过程
        received_packets = {}  # {packet_idx: data}
        total_transmission_time = 0.0
        robust_bytes_transmitted = 0
        critical_bytes_transmitted = 0
        robust_transmissions = 0
        critical_transmissions = 0
        initial_losses = 0
        retransmissions = 0
        total_transmissions = 0

        for i, packet_data in enumerate(packets):
            current_packet_size = len(packet_data)
            is_robust_packet = packet_layer_type_map.get(i) == 'robust'

            # 确定是否需要重传
            should_retransmit_on_loss = False
            if transport_type == 'TCP':
                should_retransmit_on_loss = True
            elif transport_type == 'LTQ':
                ltq_phase = self.get_ltq_strategy_phase(current_round, current_accuracy)
                if ltq_phase == 'early':
                    should_retransmit_on_loss = True
                elif ltq_phase == 'middle' and not is_robust_packet:
                    should_retransmit_on_loss = True

            # 模拟传输
            attempts = 0
            packet_successfully_sent = False
            while attempts <= max_retries:
                attempts += 1
                total_transmissions += 1
                time_for_packet, _ = client.calculate_transmission_time(current_packet_size)
                total_transmission_time += time_for_packet

                if is_robust_packet:
                    robust_bytes_transmitted += current_packet_size
                    robust_transmissions += 1
                else:
                    critical_bytes_transmitted += current_packet_size
                    critical_transmissions += 1

                if not self._gilbert_elliott_packet_loss(client):
                    received_packets[i] = packet_data
                    packet_successfully_sent = True
                    break  # 成功接收
                else:
                    # 仅在第一次尝试失败时计为初始丢包
                    if attempts == 1:
                        initial_losses += 1

                # 如果不需要重传，则直接跳出循环
                if not should_retransmit_on_loss:
                    break

                if attempts > max_retries:
                   # print(f"客户端 {client.id} 的数据包 {i + 1}/{num_packets} 重传失败")
                    break

        retransmissions = total_transmissions - num_packets

        # 5. 重组模型
        reconstructed_bytes = bytearray(total_size)
        lost_packet_count = 0
        for i in range(num_packets):
            start = i * packet_size
            end = start + len(packets[i])
            if i in received_packets:
                reconstructed_bytes[start:end] = received_packets[i]
            else:
                # 从旧的全局模型中填充丢失的部分
                reconstructed_bytes[start:end] = fallback_data_bytes[start:end]
                lost_packet_count += 1

        # 6. 反序列化为状态字典
        buffer = io.BytesIO(reconstructed_bytes)
        reconstructed_state_dict = torch.load(buffer)

        # 7. 更新服务器端的权重记录
        self.client_weights[client_id] = {
            'state_dict': reconstructed_state_dict,
            'num_samples': num_samples
        }

        # 8. 更新统计数据
        actual_received_size = robust_bytes_transmitted + critical_bytes_transmitted
        self.round_transmission_times[client_id] = total_transmission_time
        self.total_up_communication += actual_received_size
        self.round_up_communication += actual_received_size
        self.total_robust_communication += robust_bytes_transmitted
        self.round_robust_communication += robust_bytes_transmitted
        self.total_critical_communication += critical_bytes_transmitted
        self.round_critical_communication += critical_bytes_transmitted
        self.total_robust_transmission_count += robust_transmissions
        self.round_robust_transmission_count += robust_transmissions
        self.total_critical_transmission_count += critical_transmissions
        self.round_critical_transmission_count += critical_transmissions

        print(f"服务器已接收客户端 {client.id} 的更新:")
        print(
            f"  总数据包: {num_packets}, 初始丢包: {initial_losses}, 重传次数: {retransmissions}, 最终丢失: {lost_packet_count}")
        print(f"  总传输流量: {actual_received_size / 1024 / 1024:.3f} MB, 总传输时间: {total_transmission_time:.2f}s")

        return True

    def classify_layers_dual_factor(self):
        """基于双因子动态分层（联邦学习版）"""
        if not self.use_dynamic_layer_classification:
            return [], []

        # 获取指标（包含近似梯度）
        raw_data = self.metric_calculator.get_dual_metrics(self.global_model)

        # 归一化
        movements = [x['movement'] for x in raw_data]
        grads = [x['grad'] for x in raw_data]  # ← 这里是EMA近似值

        max_mov = max(movements) if movements and max(movements) > 0 else 1.0
        max_grad = max(grads) if grads and max(grads) > 0 else 1.0

        final_scores = []
        for item in raw_data:
            norm_mov = item['movement'] / max_mov
            norm_grad = item['grad'] / max_grad

            combined_score = norm_mov + self.grad_beta * norm_grad

            final_scores.append({
                'name': item['name'],
                'score': combined_score
            })

        # 排序和切分
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        num_critical = max(1, int(len(final_scores) * self.critical_ratio))

        critical_names = [x['name'] for x in final_scores[:num_critical]]
        robust_names = [x['name'] for x in final_scores[num_critical:]]

        return critical_names, robust_names

    def finalize_round_transmission_time(self):
        """完成本轮传输，记录最大传输时间"""
        if self.round_transmission_times:
            max_time = max(self.round_transmission_times.values())
            max_client = max(self.round_transmission_times, key=self.round_transmission_times.get)
            self.max_transmission_times.append(max_time)

            # 格式化每个客户端的传输时间为两位小数
            formatted_times = {k: f"{v:.2f}" for k, v in self.round_transmission_times.items()}
            print(f"本轮传输完成，最慢客户端: {max_client}，传输时间: {max_time:.2f}s")
            print(f"所有客户端传输时间: {formatted_times}")
        else:
            self.max_transmission_times.append(0.0)

    def next_round(self):
        """准备下一轮训练"""
        # 更新权重记录（用于下一轮计算变化量）
        if self.use_dynamic_layer_classification and self.metric_calculator:
            self.metric_calculator.update_prev_weights(self.global_model)
            print("已更新权重记录用于下一轮计算")

        # 记录本轮通信量
        self.communication_history.append({
            'round': len(self.communication_history) + 1,
            'up_communication': self.round_up_communication,
            'robust_communication': self.round_robust_communication,
            'critical_communication': self.round_critical_communication,
            'robust_transmissions': self.round_robust_transmission_count,
            'critical_transmissions': self.round_critical_transmission_count
        })

        # 重置统计
        self.round_up_communication = 0
        self.round_robust_communication = 0
        self.round_critical_communication = 0
        self.round_robust_transmission_count = 0
        self.round_critical_transmission_count = 0
        self.round_transmission_times = {}
        self.client_weights = {}

    def get_communication_stats(self):
        """获取通信统计信息"""
        return {
            "总上行通信量(MB)": self.total_up_communication / (1024 * 1024),
            "总鲁棒层通信量(MB)": self.total_robust_communication / (1024 * 1024),
            "总关键层通信量(MB)": self.total_critical_communication / (1024 * 1024),
            "总鲁棒层传输次数": self.total_robust_transmission_count,
            "总关键层传输次数": self.total_critical_transmission_count,
            "每轮通信量记录": self.communication_history,
            "每轮最大传输时间": self.max_transmission_times  # 新增传输时间记录
        }

    def test_model(self):
        """在测试集上评估全局模型"""
        print("开始模型测试")
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)

                # 计算损失
                test_loss += nn.CrossEntropyLoss()(output, target).item()

                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total

        print(f"测试结果 | 损失: {avg_loss:.4f} | 准确率: {accuracy:.2f}%")
        return {'loss': avg_loss, 'accuracy': accuracy}
