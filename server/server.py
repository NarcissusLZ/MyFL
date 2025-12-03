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
        联邦学习版本：获取双因子指标
        """
        metrics_data = []

        for name, param in model.named_parameters():
            if 'weight' not in name or param.dim() <= 1:
                continue

            # 因子1: 当前轮次权重变化 (W_t - W_{t-1})
            movement = 0.0
            if name in self.prev_weights:
                prev_w = self.prev_weights[name].to(param.device)
                movement = torch.norm(param.data - prev_w, p=2).item()

            # 因子2: 用EMA平滑的权重变化替代梯度
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
        self.total_up_communication = 0  # 上行通信总量（字节）
        self.round_up_communication = 0  # 当前轮次上行通信量

        # 添加分层流量统计
        self.total_robust_communication = 0
        self.total_critical_communication = 0
        self.round_robust_communication = 0
        self.round_critical_communication = 0

        self.communication_history = []  # 每轮通信量记录

        # 添加传输时间统计
        self.round_transmission_times = {}  # 当前轮次各客户端传输时间
        self.max_transmission_times = []  # 每轮最大传输时间记录

        # 添加传输次数统计
        self.total_robust_transmission_count = 0
        self.total_critical_transmission_count = 0
        self.round_robust_transmission_count = 0
        self.round_critical_transmission_count = 0

        # 记录聚合权重
        self.client_weights = {}

        # Gilbert-Elliott模型参数初始化
        self.gilbert_elliott_states = {}
        self.client_random_generators = {}
        self._init_gilbert_elliott_params()

        # 添加动态分层配置参数
        self.use_dynamic_layer_classification = config.get('use_dynamic_layer_classification', False)
        self.critical_ratio = config.get('critical_ratio', 0.5)
        self.grad_beta = config.get('grad_beta', 1.0)

        # === 修复核心：初始化缓存列表 ===
        self.metric_calculator = None
        self.cached_critical_layers = []
        self.cached_robust_layers = []

        if self.use_dynamic_layer_classification:
            self.metric_calculator = LayerMetricCalculator(self.global_model)
            # 预先进行一次默认分层，防止第0轮出错
            self._init_default_layers()
        else:
            # 如果不使用动态，则读取配置文件的固定分层
            self._init_fixed_layers()

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
        self.gilbert_elliott_states = {}
        self.client_random_generators = {}
        print("已初始化Gilbert-Elliott模型状态字典")

    def _get_client_random_generator(self, client_id):
        """为每个客户端获取或创建独立的随机数生成器"""
        if client_id not in self.client_random_generators:
            client_seed = self.random_seed + hash(str(client_id)) % 10000
            self.client_random_generators[client_id] = random.Random(client_seed)
            print(f"为客户端 {client_id} 创建随机数生成器，种子: {client_seed}")
        return self.client_random_generators[client_id]

    def _init_default_layers(self):
        """初始化默认分层（仅用于第0轮，按顺序切分）"""
        all_layers = [name for name, _ in self.global_model.named_parameters() if 'weight' in name]
        split_idx = int(len(all_layers) * self.critical_ratio)
        self.cached_critical_layers = all_layers[:split_idx]
        self.cached_robust_layers = all_layers[split_idx:]
        print(f"已初始化默认分层: 关键层 {len(self.cached_critical_layers)}, 鲁棒层 {len(self.cached_robust_layers)}")

    def _init_fixed_layers(self):
        """从配置文件初始化固定分层"""
        layers_to_drop = self.config.get('layers_to_drop', [])
        all_layers = [name for name, _ in self.global_model.named_parameters() if 'weight' in name]

        self.cached_robust_layers = [
            layer for layer in all_layers
            if any(pattern in layer for pattern in layers_to_drop)
        ]
        self.cached_critical_layers = [
            layer for layer in all_layers
            if layer not in self.cached_robust_layers
        ]
        print(f"使用固定配置分层: 关键层 {len(self.cached_critical_layers)}, 鲁棒层 {len(self.cached_robust_layers)}")

    def get_model_size(self):
        """计算模型参数大小（字节）"""
        size_bytes = 0
        for param in self.global_model.parameters():
            size_bytes += param.nelement() * 4
        return size_bytes

    def init_model(self):
        print("服务器开始初始化模型")
        model_name = self.config['model']
        dataset_name = self.config['dataset']

        if model_name == 'VGG16':
            from models.vgg16 import CIFAR10_VGG16, CIFAR100_VGG16
            return CIFAR10_VGG16(num_classes=10) if dataset_name == 'cifar10' else CIFAR100_VGG16(num_classes=100)
        elif model_name == 'RESNET18':
            from models.resnet18 import CIFAR10_ResNet18, CIFAR100_ResNet18
            return CIFAR10_ResNet18(num_classes=10) if dataset_name == 'cifar10' else CIFAR100_ResNet18(num_classes=100)
        elif model_name == 'RESNET50':
            from models.resnet50 import CIFAR10_ResNet50, CIFAR100_ResNet50
            return CIFAR10_ResNet50(num_classes=10) if dataset_name == 'cifar10' else CIFAR100_ResNet50(num_classes=100)
        else:
            raise ValueError(f"未知模型: {model_name}")

    def _select_device(self, device_config):
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
        model_class = self.global_model.__class__

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
            return False

        client_rng = self._get_client_random_generator(client_id)

        if client_id not in self.gilbert_elliott_states:
            self.gilbert_elliott_states[client_id] = 0 if client_rng.random() > client_loss_rate else 1

        current_state = self.gilbert_elliott_states[client_id]
        good_to_bad = 0.5 * client_loss_rate
        bad_to_good = 0.5 - good_to_bad
        rand = client_rng.random()

        if current_state == 0:  # Good
            if rand <= good_to_bad:
                self.gilbert_elliott_states[client_id] = 1
                return True
            return False
        else:  # Bad
            if rand <= 1 - bad_to_good:
                return True
            self.gilbert_elliott_states[client_id] = 0
            return False

    def get_ltq_strategy_phase(self, current_round, current_accuracy=None):
        """确定LTQ当前应该采用的策略阶段"""
        total_rounds = self.config['num_rounds']
        method = self.config.get('ltq_phase_method', 'rounds')

        early_ratio = self.config.get('ltq_early_ratio', 0.2)
        middle_ratio = self.config.get('ltq_middle_ratio', 0.6)
        min_middle = int(total_rounds * early_ratio)
        min_late = int(total_rounds * middle_ratio)

        if method == 'rounds':
            if current_round <= min_middle:
                return 'early'
            elif current_round <= min_late:
                return 'middle'
            else:
                return 'late'
        elif method == 'accuracy':
            if current_accuracy is None: return 'early'
            if current_accuracy < self.config.get('ltq_early_acc_threshold', 40):
                return 'early'
            elif current_accuracy < self.config.get('ltq_middle_acc_threshold', 70):
                return 'middle'
            else:
                return 'late'
        else:  # hybrid
            if current_round < min_middle or (
                    current_accuracy is not None and current_accuracy < self.config.get('ltq_early_acc_threshold', 40)):
                return 'early'
            elif current_round < min_late or (
                    current_accuracy is not None and current_accuracy < self.config.get('ltq_middle_acc_threshold',
                                                                                        70)):
                return 'middle'
            else:
                return 'late'

    def _serialize_and_packetize(self, state_dict, packet_size=1500):
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data_bytes = buffer.getvalue()
        packets = []
        for i in range(0, len(data_bytes), packet_size):
            packets.append(data_bytes[i:i + packet_size])
        return packets, data_bytes

    def _get_packet_layer_type_map(self, state_dict, total_size, packet_size=1500):
        """创建从数据包索引到层类型的映射（使用缓存的分层结果）"""

        # === 修复核心：直接使用缓存的列表，不实时计算 ===
        critical_layers = self.cached_critical_layers
        robust_layers = self.cached_robust_layers
        # ==========================================

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
        """接收客户端上传的模型更新"""
        if model_state_dict is None:
            print(f"客户端 {client.id} 上传的模型状态字典为空，跳过更新")
            return False

        client_id = client.id
        transport_type = self.config.get('Transport', 'TCP')
        packet_size = 1500
        max_retries = 16

        packets, original_data_bytes = self._serialize_and_packetize(model_state_dict, packet_size)
        num_packets = len(packets)
        total_size = len(original_data_bytes)
        _, fallback_data_bytes = self._serialize_and_packetize(self.global_model.state_dict(), packet_size)

        packet_layer_type_map = {}
        if transport_type == 'LTQ':
            packet_layer_type_map = self._get_packet_layer_type_map(model_state_dict, total_size, packet_size)

        received_packets = {}
        total_transmission_time = 0.0
        robust_bytes_transmitted = 0
        critical_bytes_transmitted = 0
        robust_transmissions = 0
        critical_transmissions = 0
        initial_losses = 0
        total_transmissions = 0

        for i, packet_data in enumerate(packets):
            current_packet_size = len(packet_data)
            is_robust_packet = packet_layer_type_map.get(i) == 'robust'

            should_retransmit_on_loss = False
            if transport_type == 'TCP':
                should_retransmit_on_loss = True
            elif transport_type == 'LTQ':
                ltq_phase = self.get_ltq_strategy_phase(current_round, current_accuracy)
                if ltq_phase == 'early':
                    should_retransmit_on_loss = True
                elif ltq_phase == 'middle' and not is_robust_packet:
                    should_retransmit_on_loss = True

            attempts = 0
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
                    break
                else:
                    if attempts == 1: initial_losses += 1

                if not should_retransmit_on_loss: break
                if attempts > max_retries: break

        retransmissions = total_transmissions - num_packets

        # 重组
        reconstructed_bytes = bytearray(total_size)
        lost_packet_count = 0
        for i in range(num_packets):
            start = i * packet_size
            end = start + len(packets[i])
            if i in received_packets:
                reconstructed_bytes[start:end] = received_packets[i]
            else:
                reconstructed_bytes[start:end] = fallback_data_bytes[start:end]
                lost_packet_count += 1

        buffer = io.BytesIO(reconstructed_bytes)
        try:
            reconstructed_state_dict = torch.load(buffer)
        except Exception as e:
            print(f"反序列化失败: {e}")
            return False

        self.client_weights[client_id] = {
            'state_dict': reconstructed_state_dict,
            'num_samples': num_samples
        }

        # 统计更新
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

        # === 恢复输出：详细的接收日志 ===
        print(f"服务器已接收客户端 {client.id} 的更新:")
        print(
            f"  总数据包: {num_packets}, 初始丢包: {initial_losses}, 重传次数: {retransmissions}, 最终丢失: {lost_packet_count}")
        print(f"  总传输流量: {actual_received_size / 1024 / 1024:.3f} MB, 总传输时间: {total_transmission_time:.2f}s")
        # ================================

        return True

    def classify_layers_dual_factor(self):
        """
        基于双因子动态分层（计算并更新缓存）
        调用时机：每轮聚合完成后，prepare next round 之前
        """
        if not self.use_dynamic_layer_classification or self.metric_calculator is None:
            return self.cached_critical_layers, self.cached_robust_layers

        # 获取指标 (Current Global - Prev Global)
        raw_data = self.metric_calculator.get_dual_metrics(self.global_model)
        if not raw_data:
            print("警告: 无法获取层指标，保持原样")
            return self.cached_critical_layers, self.cached_robust_layers

        movements = [x['movement'] for x in raw_data]
        grads = [x['grad'] for x in raw_data]

        max_mov = max(movements) if movements else 1.0
        max_grad = max(grads) if grads else 1.0

        # === 核心保护：如果模型没有变化，跳过更新 ===
        if max_mov == 0 and max_grad == 0:
            print("注意：模型权重未变化（Score=0），保持上一轮分层策略。")
            return self.cached_critical_layers, self.cached_robust_layers

        final_scores = []
        for item in raw_data:
            norm_mov = item['movement'] / max_mov if max_mov > 0 else 0
            norm_grad = item['grad'] / max_grad if max_grad > 0 else 0
            combined_score = norm_mov + self.grad_beta * norm_grad

            final_scores.append({
                'name': item['name'],
                'score': combined_score
            })

        # 排序
        final_scores.sort(key=lambda x: x['score'], reverse=True)

        # 切分
        num_critical = max(1, int(len(final_scores) * self.critical_ratio))

        critical_names = [x['name'] for x in final_scores[:num_critical]]
        robust_names = [x['name'] for x in final_scores[num_critical:]]

        # 更新缓存
        self.cached_critical_layers = critical_names
        self.cached_robust_layers = robust_names

        # === 恢复输出：详细的分层信息 ===
        print(f"\n动态分层结果 (阈值: {final_scores[num_critical - 1]['score']:.4f}):")
        print(f"  关键层 ({len(critical_names)}): {', '.join(critical_names[:8])}...")
        print(f"  鲁棒层 ({len(robust_names)}): {', '.join(robust_names[:9])}...")
        # ==============================

        return self.cached_critical_layers, self.cached_robust_layers

    def finalize_round_transmission_time(self):
        """完成本轮传输，记录最大传输时间"""
        if self.round_transmission_times:
            max_time = max(self.round_transmission_times.values())
            max_client = max(self.round_transmission_times, key=self.round_transmission_times.get)
            self.max_transmission_times.append(max_time)

            # === 恢复输出：详细的时间统计 ===
            formatted_times = {k: f"{v:.2f}" for k, v in self.round_transmission_times.items()}
            print(f"本轮传输完成，最慢客户端: {max_client}，传输时间: {max_time:.2f}s")
            print(f"所有客户端传输时间: {formatted_times}")
            # ==============================
        else:
            self.max_transmission_times.append(0.0)

    def next_round(self):
        """准备下一轮训练"""

        # === 修复核心：1. 先基于本轮聚合后的模型计算新策略 ===
        if self.use_dynamic_layer_classification:
            print("正在计算下一轮传输策略...")
            self.classify_layers_dual_factor()

        # === 修复核心：2. 再更新基准权重（将当前模型设为W_prev） ===
        if self.use_dynamic_layer_classification and self.metric_calculator:
            self.metric_calculator.update_prev_weights(self.global_model)
            print("已更新基准权重 (W_prev)")

            # 记录本轮通信量
        self.communication_history.append({
            'round': len(self.communication_history) + 1,
            'up_communication': self.round_up_communication,
            # === 修改下面两行键名 ===
            'robust_layer_communication': self.round_robust_communication,  # 改为 robust_layer_communication
            'critical_layer_communication': self.round_critical_communication,  # 改为 critical_layer_communication
            # ======================
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
        return {
            "总上行通信量(MB)": self.total_up_communication / (1024 * 1024),
            "总鲁棒层通信量(MB)": self.total_robust_communication / (1024 * 1024),
            "总关键层通信量(MB)": self.total_critical_communication / (1024 * 1024),
            "总鲁棒层传输次数": self.total_robust_transmission_count,
            "总关键层传输次数": self.total_critical_transmission_count,
            "每轮通信量记录": self.communication_history,
            "每轮最大传输时间": self.max_transmission_times
        }

    def test_model(self):
        print("开始模型测试")
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += nn.CrossEntropyLoss()(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        print(f"测试结果 | 损失: {avg_loss:.4f} | 准确率: {accuracy:.2f}%")
        return {'loss': avg_loss, 'accuracy': accuracy}