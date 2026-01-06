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
    联邦学习版：基于开题报告 2.3.2 的双因子层重要性评估
    """

    def __init__(self, model):
        print("初始化 LayerMetricCalculator: 正在备份初始权重...")
        self.prev_weights = {}
        self.movement_ema = {}  # 因子2: 历史趋势 (H_t)

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
        获取双因子指标：
        1. 单位参数平均变化量 (公式 12)
        2. 指数移动平均趋势 (公式 13)
        """
        metrics_data = []

        for name, param in model.named_parameters():
            if 'weight' not in name or param.dim() <= 1:
                continue

            movement = 0.0
            num_params = param.numel()  # N_l: 该层的参数总数

            if name in self.prev_weights:
                prev_w = self.prev_weights[name].to(param.device)

                # === 修改点 1: 实现公式 (12) ===
                # V_l^t = ||W_t - W_{t-1}|| / sqrt(N_l)
                # 消除层规模差异带来的偏差
                l2_norm = torch.norm(param.data - prev_w, p=2).item()
                if num_params > 0:
                    movement = l2_norm / math.sqrt(num_params)
                else:
                    movement = l2_norm

            # === 修改点 2: 实现公式 (13) ===
            # H_l^t = lambda * H_{t-1} + (1-lambda) * V_l^t
            # 平滑衰减系数 lambda 取 0.8
            self.movement_ema[name] = (
                    0.8 * self.movement_ema[name] + 0.2 * movement
            )
            grad_approx = self.movement_ema[name]

            metrics_data.append({
                'name': name,
                'movement': movement,  # V_l^t
                'grad': grad_approx  # H_l^t
            })

        return metrics_data


class Server:
    def __init__(self, config, test_dataset):
        self.config = config
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else self._select_device(config['device'])

        # 设置随机种子
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

        # 通信量统计
        self.total_up_communication = 0
        self.round_up_communication = 0
        self.total_robust_communication = 0
        self.total_critical_communication = 0
        self.round_robust_communication = 0
        self.round_critical_communication = 0
        self.communication_history = []
        self.round_transmission_times = {}
        self.max_transmission_times = []
        self.total_robust_transmission_count = 0
        self.total_critical_transmission_count = 0
        self.round_robust_transmission_count = 0
        self.round_critical_transmission_count = 0

        self.client_weights = {}

        # Gilbert-Elliott模型参数
        self.gilbert_elliott_states = {}
        self.client_random_generators = {}
        self._init_gilbert_elliott_params()

        # 分层配置
        self.use_dynamic_layer_classification = config.get('use_dynamic_layer_classification', False)
        self.grad_beta = config.get('grad_beta', 1.0)  # 调节因子 gamma (公式 14)

        # === 新增: Loss 自适应相关状态 ===
        self.prev_test_loss = None  # 记录上一轮 Loss (L_{t-1})
        self.min_critical_ratio = 0.1  # δ_min
        self.max_critical_ratio = 0.5  # δ_max
        self.sigmoid_k = config.get('sigmoid_k', 20.0)  # Sigmoid 斜率 k
        self.sigmoid_tau = config.get('sigmoid_tau', 0.05)  # 中心偏移 tau (假设loss变化率一般在这个量级)

        self.metric_calculator = None
        self.cached_critical_layers = []
        self.cached_robust_layers = []

        if self.use_dynamic_layer_classification:
            self.metric_calculator = LayerMetricCalculator(self.global_model)
            self._init_default_layers()
        else:
            self._init_fixed_layers()

        print(f"服务器初始化完成, 设备: {self.device}")

    def _set_random_seeds(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def _init_gilbert_elliott_params(self):
        self.gilbert_elliott_states = {}
        self.client_random_generators = {}

    def _get_client_random_generator(self, client_id):
        if client_id not in self.client_random_generators:
            client_seed = self.random_seed + hash(str(client_id)) % 10000
            self.client_random_generators[client_id] = random.Random(client_seed)
        return self.client_random_generators[client_id]

    def _init_default_layers(self):
        """初始化默认分层（第0轮，100%关键层）"""
        all_layers = [name for name, _ in self.global_model.named_parameters() if 'weight' in name]
        self.cached_critical_layers = all_layers
        self.cached_robust_layers = []
        print(f"初始分层: 100% 关键层 (TCP)")

    def _init_fixed_layers(self):
        """非动态模式下的固定分层"""
        all_layers = [name for name, _ in self.global_model.named_parameters() if 'weight' in name]
        cutoff = int(len(all_layers) * self.min_critical_ratio)
        self.cached_critical_layers = all_layers[:cutoff]
        self.cached_robust_layers = all_layers[cutoff:]

    def get_model_size(self):
        size_bytes = 0
        for param in self.global_model.parameters():
            size_bytes += param.nelement() * 4
        return size_bytes

    def init_model(self):
        model_name = self.config['model']
        dataset_name = self.config['dataset']
        # (保持原有的模型初始化逻辑不变)
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
        global_state_dict = copy.deepcopy(self.global_model.state_dict())
        model_class = self.global_model.__class__
        for client in selected_clients:
            client.receive_model(model_class, global_state_dict)

    def select_clients(self, all_clients, fraction=0.1):
        num_selected = max(1, int(fraction * len(all_clients)))
        selected_clients = np.random.choice(all_clients, num_selected, replace=False)
        return selected_clients

    def _gilbert_elliott_packet_loss(self, client):
        # (保持原有的丢包模拟逻辑不变)
        client_id = client.id
        client_loss_rate = client.packet_loss
        if client_loss_rate == 0: return False
        client_rng = self._get_client_random_generator(client_id)
        if client_id not in self.gilbert_elliott_states:
            self.gilbert_elliott_states[client_id] = 0 if client_rng.random() > client_loss_rate else 1
        current_state = self.gilbert_elliott_states[client_id]
        good_to_bad = 0.5 * client_loss_rate
        bad_to_good = 0.5 - good_to_bad
        rand = client_rng.random()
        if current_state == 0:
            if rand <= good_to_bad:
                self.gilbert_elliott_states[client_id] = 1
                return True
            return False
        else:
            if rand <= 1 - bad_to_good: return True
            self.gilbert_elliott_states[client_id] = 0
            return False

    def _serialize_and_packetize(self, state_dict, packet_size=1500):
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data_bytes = buffer.getvalue()
        packets = []
        for i in range(0, len(data_bytes), packet_size):
            packets.append(data_bytes[i:i + packet_size])
        return packets, data_bytes

    def get_adaptive_critical_ratio(self, current_loss):
        """
        基于开题报告公式 (15) 和 (16) 的自适应比例计算
        根据 Loss 变化率动态调整关键层比例
        """
        if self.prev_test_loss is None:
            # 第一轮没有上一轮Loss，默认高保护
            self.prev_test_loss = current_loss
            return 1.0

        # === 公式 (15): Loss 相对变化率 r_t ===
        epsilon = 1e-6
        # r_t < 0 表示 Loss 下降。训练初期下降快，r_t 为较大的负数。
        r_t = (current_loss - self.prev_test_loss) / (self.prev_test_loss + epsilon)

        # 为了符合 Sigmoid 的直觉（输入越大，输出越大），我们取变化率的“幅度”或者反转符号
        # 报告意图：训练初期(变化大/下降快) -> 高TCP比例。训练后期(稳定) -> 低TCP比例。
        # 如果 r_t = -0.5 (下降快), 我们希望 Sigmoid 输出接近 1。
        # 如果 r_t = -0.01 (稳定), 我们希望 Sigmoid 输出接近 0。
        # 因此，我们可以使用 -r_t (即下降的幅度) 作为输入指标。
        input_metric = -r_t

        # === 公式 (16): Sigmoid 映射 ===
        # delta_t = min + (max - min) * Sigmoid(k * (input - tau))
        # k: 斜率，控制敏感度
        # tau: 阈值偏移
        try:
            sigmoid_val = 1 / (1 + math.exp(-self.sigmoid_k * (input_metric - self.sigmoid_tau)))
        except OverflowError:
            sigmoid_val = 1.0 if (input_metric - self.sigmoid_tau) > 0 else 0.0

        adaptive_ratio = self.min_critical_ratio + (self.max_critical_ratio - self.min_critical_ratio) * sigmoid_val

        # 更新上一轮 Loss
        self.prev_test_loss = current_loss

        print(
            f"  [自适应比例] Loss变化率 r_t: {r_t:.4f}, 输入Sigmoid: {input_metric:.4f}, 计算比例: {adaptive_ratio * 100:.1f}%")
        return adaptive_ratio

    def classify_layers_dual_factor(self, current_loss=None):
        """
        基于双因子动态分层（公式 14）
        现在接收 current_loss 来计算自适应比例
        """
        if not self.use_dynamic_layer_classification or self.metric_calculator is None:
            return self.cached_critical_layers, self.cached_robust_layers

        # === 步骤 1: 计算自适应关键层比例 ===
        if current_loss is not None:
            effective_ratio = self.get_adaptive_critical_ratio(current_loss)
        else:
            effective_ratio = 1.0  # 默认安全策略
        # =================================

        # === 步骤 2: 获取双因子指标 (公式 12 & 13) ===
        raw_data = self.metric_calculator.get_dual_metrics(self.global_model)
        if not raw_data:
            return self.cached_critical_layers, self.cached_robust_layers

        # 提取指标用于归一化
        movements = [x['movement'] for x in raw_data]  # V_l
        grads = [x['grad'] for x in raw_data]  # H_l

        max_mov = max(movements) if movements and max(movements) > 0 else 1.0
        max_grad = max(grads) if grads and max(grads) > 0 else 1.0

        # === 步骤 3: 综合打分与排序 (公式 14) ===
        # S_l = V_l/max(V) + gamma * H_l/max(H)
        final_scores = []
        for item in raw_data:
            norm_mov = item['movement'] / max_mov
            norm_grad = item['grad'] / max_grad
            # self.grad_beta 对应公式中的 gamma
            combined_score = norm_mov + self.grad_beta * norm_grad
            final_scores.append({'name': item['name'], 'score': combined_score})

        final_scores.sort(key=lambda x: x['score'], reverse=True)

        # === 步骤 4: 划分层 ===
        num_critical = max(1, int(len(final_scores) * effective_ratio))

        critical_names = [x['name'] for x in final_scores[:num_critical]]
        robust_names = [x['name'] for x in final_scores[num_critical:]]

        self.cached_critical_layers = critical_names
        self.cached_robust_layers = robust_names

        print(
            f"动态分层完成: 关键层 {len(critical_names)} (Top {effective_ratio * 100:.0f}%), 鲁棒层 {len(robust_names)}")

        return self.cached_critical_layers, self.cached_robust_layers

    def _get_packet_layer_type_map(self, state_dict, total_size, packet_size=1500):
        # (保持原有的映射逻辑不变)
        critical_layers = self.cached_critical_layers
        robust_layers = self.cached_robust_layers
        packet_map = {}
        current_pos = 0
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        ordered_state_dict = torch.load(buffer)
        temp_buffer = io.BytesIO()
        for key, param in ordered_state_dict.items():
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
            if i not in packet_map: packet_map[i] = 'critical'
        return packet_map

    def receive_local_model(self, client, model_state_dict, num_samples, current_round=None, current_accuracy=None):
        # (保持原有的接收逻辑不变，包含TCP/UDP重传判断)
        if model_state_dict is None: return False
        client_id = client.id
        transport_type = self.config.get('Transport', 'TCP')
        packet_size = 1500
        max_retries = 40
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

            if transport_type == 'TCP':
                should_retransmit_on_loss = True
            elif transport_type == 'UDP':
                should_retransmit_on_loss = False
            elif transport_type == 'LTQ':
                should_retransmit_on_loss = not is_robust_packet
            else:
                should_retransmit_on_loss = False

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
        reconstructed_bytes = bytearray(total_size)
        for i in range(num_packets):
            start = i * packet_size
            end = start + len(packets[i])
            if i in received_packets:
                reconstructed_bytes[start:end] = received_packets[i]
            else:
                reconstructed_bytes[start:end] = fallback_data_bytes[start:end]

        buffer = io.BytesIO(reconstructed_bytes)
        try:
            reconstructed_state_dict = torch.load(buffer)
        except Exception:
            return False

        self.client_weights[client_id] = {
            'state_dict': reconstructed_state_dict,
            'num_samples': num_samples
        }
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

        return True

    def finalize_round_transmission_time(self):
        if self.round_transmission_times:
            max_time = max(self.round_transmission_times.values())
            self.max_transmission_times.append(max_time)
        else:
            self.max_transmission_times.append(0.0)

    def next_round(self, current_loss=None):
        """
        准备下一轮训练
        修改：接收 current_loss 用于计算下一轮的自适应比例
        """
        # 1. 动态分层计算 (传入当前的 Loss)
        if self.use_dynamic_layer_classification:
            print(f"正在基于 Loss ({current_loss:.4f}) 计算下一轮传输策略...")
            self.classify_layers_dual_factor(current_loss)

        # 2. 更新权重变化量的基准 (W_{t-1} -> W_t)
        if self.use_dynamic_layer_classification and self.metric_calculator:
            self.metric_calculator.update_prev_weights(self.global_model)

        # 3. 记录日志
        self.communication_history.append({
            'round': len(self.communication_history) + 1,
            'up_communication': self.round_up_communication,
            'robust_layer_communication': self.round_robust_communication,
            'critical_layer_communication': self.round_critical_communication,
            'robust_transmissions': self.round_robust_transmission_count,
            'critical_transmissions': self.round_critical_transmission_count
        })

        # 4. 重置本轮计数器
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