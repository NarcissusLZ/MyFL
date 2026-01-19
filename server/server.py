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
                l2_norm = torch.norm(param.data - prev_w, p=2).item()
                if num_params > 0:
                    movement = l2_norm / math.sqrt(num_params)
                else:
                    movement = l2_norm

            self.movement_ema[name] = (
                    0.8 * self.movement_ema[name] + 0.2 * movement
            )
            grad_approx = self.movement_ema[name]

            metrics_data.append({
                'name': name,
                'movement': movement,
                'grad': grad_approx
            })

        return metrics_data


class Server:
    def __init__(self, config, test_dataset):
        self.config = config
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else self._select_device(config['device'])

        # 设置随机种���
        self.random_seed = config.get('random_seed', 42)
        self._set_random_seeds()

        # 初始化全局模型
        self.global_model = self.init_model()
        self.global_model.to(self.device)

        # 准备测试数据集
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=False,
            num_workers=16,  # <--- 关键修改：建议设置 4 或 8
            pin_memory=True  # <--- 关键修改：加速 CUDA 传输
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
        self.grad_beta = config.get('grad_beta', 1.0)

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
        all_layers = [name for name, _ in self.global_model.named_parameters() if 'weight' in name]
        self.cached_critical_layers = all_layers
        self.cached_robust_layers = []
        print(f"初始分层: 100% 关键层 (TCP)")

    def _init_fixed_layers(self):
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
        if model_name == 'VGG16':
            from models.vgg16 import CIFAR10_VGG16, CIFAR100_VGG16
            if dataset_name == 'cifar10':
                return CIFAR10_VGG16(num_classes=10)
            elif dataset_name == 'cifar100':
                return CIFAR100_VGG16(num_classes=100)
        elif model_name == 'RESNET18':
            from models.resnet18 import CIFAR10_ResNet18, CIFAR100_ResNet18, GoogleSpeech_ResNet18
            if dataset_name == 'cifar10':
                return CIFAR10_ResNet18(num_classes=10)
            elif dataset_name == 'cifar100':
                return CIFAR100_ResNet18(num_classes=100)
            elif dataset_name == 'googlespeech':
                return GoogleSpeech_ResNet18(num_classes=35)
            elif dataset_name == 'iot23':
                # 调用新创建的 ResNet-1D 模型 (在此处导入以避免未使用的引用)
                from models.resnet_iot23 import ResNet_IoT23
                return ResNet_IoT23(num_classes=5, input_dim=20)

        elif model_name == 'RESNET50':
            from models.resnet50 import CIFAR10_ResNet50, CIFAR100_ResNet50, ImageNet_ResNet50
            if dataset_name == 'cifar10':
                return CIFAR10_ResNet50(num_classes=10)
            elif dataset_name == 'cifar100':
                return CIFAR100_ResNet50(num_classes=100)
            elif dataset_name == 'imagenet':
                return ImageNet_ResNet50(num_classes=1000)


        elif model_name == 'MOBILENET_V1':
            from models.mobilenet_audio import MobileNetV1_Audio, MobileNetV1_IoT23
            if dataset_name == 'googlespeech':
                return MobileNetV1_Audio(num_classes=35, input_channels=1)
            elif dataset_name == 'iot23':
                # 使用新添加的 1D MobileNet
                return MobileNetV1_IoT23(num_classes=5, input_dim=20)


        elif model_name == 'MOBILENET_V2':
            from models.mobilenet_audio import MobileNetV2_Audio, MobileNetV2_IoT23
            if dataset_name == 'googlespeech':
                return MobileNetV2_Audio(num_classes=35, input_channels=1)
            elif dataset_name == 'iot23':
                # 使用新添加的 1D MobileNet
                return MobileNetV2_IoT23(num_classes=5, input_dim=20)

        # === 新增: MLP 支持 (IoT-23) ===
        elif model_name == 'MLP':
            from models.MLP import MLP_IoT23
            if dataset_name == 'iot23':
                return MLP_IoT23(num_classes=5, input_dim=20)

        raise ValueError(f"不支持的模型与数据集组合: {model_name} + {dataset_name}")

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
        if self.prev_test_loss is None:
            self.prev_test_loss = current_loss
            return 1.0
        epsilon = 1e-6
        r_t = (current_loss - self.prev_test_loss) / (self.prev_test_loss + epsilon)
        input_metric = -r_t
        try:
            sigmoid_val = 1 / (1 + math.exp(-self.sigmoid_k * (input_metric - self.sigmoid_tau)))
        except OverflowError:
            sigmoid_val = 1.0 if (input_metric - self.sigmoid_tau) > 0 else 0.0
        adaptive_ratio = self.min_critical_ratio + (self.max_critical_ratio - self.min_critical_ratio) * sigmoid_val
        self.prev_test_loss = current_loss
        print(
            f"  [自适应比例] Loss变化率 r_t: {r_t:.4f}, 输入Sigmoid: {input_metric:.4f}, 计算比例: {adaptive_ratio * 100:.1f}%")
        return adaptive_ratio

    def classify_layers_dual_factor(self, current_loss=None):
        if not self.use_dynamic_layer_classification or self.metric_calculator is None:
            return self.cached_critical_layers, self.cached_robust_layers
        if current_loss is not None:
            effective_ratio = self.get_adaptive_critical_ratio(current_loss)
        else:
            effective_ratio = 1.0
        raw_data = self.metric_calculator.get_dual_metrics(self.global_model)
        if not raw_data:
            return self.cached_critical_layers, self.cached_robust_layers
        movements = [x['movement'] for x in raw_data]
        grads = [x['grad'] for x in raw_data]
        max_mov = max(movements) if movements and max(movements) > 0 else 1.0
        max_grad = max(grads) if grads and max(grads) > 0 else 1.0
        final_scores = []
        for item in raw_data:
            norm_mov = item['movement'] / max_mov
            norm_grad = item['grad'] / max_grad
            combined_score = norm_mov + self.grad_beta * norm_grad
            final_scores.append({'name': item['name'], 'score': combined_score})
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        num_critical = max(1, int(len(final_scores) * effective_ratio))
        critical_names = [x['name'] for x in final_scores[:num_critical]]
        robust_names = [x['name'] for x in final_scores[num_critical:]]
        self.cached_critical_layers = critical_names
        self.cached_robust_layers = robust_names
        print(
            f"动态分层完成: 关键层 {len(critical_names)} (Top {effective_ratio * 100:.0f}%), 鲁棒层 {len(robust_names)}")
        return self.cached_critical_layers, self.cached_robust_layers

    def _get_packet_layer_type_map(self, state_dict, total_size, packet_size=1500):
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

    def simulate_transmission(self, client, model_state_dict, num_samples):
        """
        核心方法：模拟传输并返回结果，但不立即更新服务器的统计信息。
        用于在 Top-K 模式下预先计算每个客户端的耗时和结果。
        """
        if model_state_dict is None: return None
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
        total_transmissions = 0
        initial_losses = 0

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
            return None

        # 返回模拟结果包
        return {
            'client_id': client_id,
            'state_dict': reconstructed_state_dict,
            'num_samples': num_samples,
            'transmission_time': total_transmission_time,
            'metrics': {
                'actual_received_size': robust_bytes_transmitted + critical_bytes_transmitted,
                'robust_bytes': robust_bytes_transmitted,
                'critical_bytes': critical_bytes_transmitted,
                'robust_count': robust_transmissions,
                'critical_count': critical_transmissions
            }
        }

    def receive_local_model(self, client, model_state_dict, num_samples, current_round=None, current_accuracy=None):
        """兼容旧方法的 Wrapper"""
        result = self.simulate_transmission(client, model_state_dict, num_samples)
        if result:
            self.update_server_stats([result])
            return True
        return False

    def update_server_stats(self, results):
        """
        批量接受客户端结果（Top-K 策略调用此方法）
        """
        for res in results:
            client_id = res['client_id']
            metrics = res['metrics']

            self.client_weights[client_id] = {
                'state_dict': res['state_dict'],
                'num_samples': res['num_samples']
            }
            self.round_transmission_times[client_id] = res['transmission_time']

            self.total_up_communication += metrics['actual_received_size']
            self.round_up_communication += metrics['actual_received_size']
            self.total_robust_communication += metrics['robust_bytes']
            self.round_robust_communication += metrics['robust_bytes']
            self.total_critical_communication += metrics['critical_bytes']
            self.round_critical_communication += metrics['critical_bytes']
            self.total_robust_transmission_count += metrics['robust_count']
            self.round_robust_transmission_count += metrics['robust_count']
            self.total_critical_transmission_count += metrics['critical_count']
            self.round_critical_transmission_count += metrics['critical_count']

    def finalize_round_transmission_time(self):
        if self.round_transmission_times:
            max_time = max(self.round_transmission_times.values())
            self.max_transmission_times.append(max_time)
        else:
            self.max_transmission_times.append(0.0)

    def next_round(self, current_loss=None):
        if self.use_dynamic_layer_classification:
            print(f"正在基于 Loss ({current_loss:.4f}) 计算下一轮传输策略...")
            self.classify_layers_dual_factor(current_loss)
        if self.use_dynamic_layer_classification and self.metric_calculator:
            self.metric_calculator.update_prev_weights(self.global_model)
        self.communication_history.append({
            'round': len(self.communication_history) + 1,
            'up_communication': self.round_up_communication,
            'robust_layer_communication': self.round_robust_communication,
            'critical_layer_communication': self.round_critical_communication,
            'robust_transmissions': self.round_robust_transmission_count,
            'critical_transmissions': self.round_critical_transmission_count
        })
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
        print("开始模型测试...")
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        # === 动态确定类别数量与名称 ===
        dataset_name = self.config.get('dataset', 'cifar10').lower()

        # 预定义常见数据集标签
        if dataset_name == 'cifar10':
            num_classes = 10
            classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        elif dataset_name == 'cifar100':
            num_classes = 100
            classes = [str(i) for i in range(100)]
        elif 'googlespeech' in dataset_name:
            num_classes = 35
            # Google Speech Commands V2 (35类)
            classes = [str(i) for i in range(35)]
        elif 'iot23' in dataset_name:
            num_classes = 5
            classes = ['Benign', 'DDoS', 'PortScan', 'C&C', 'Malware']
        elif 'imagenet' in dataset_name:
            num_classes = 1000
            classes = [str(i) for i in range(1000)]
        else:
            # 尝试通过模型最后一层推断，若无法推断则默认 10
            num_classes = 10
            if hasattr(self.global_model, 'linear'):
                num_classes = self.global_model.linear.out_features
            elif hasattr(self.global_model, 'fc'):
                num_classes = self.global_model.fc.out_features
            classes = [f"Class_{i}" for i in range(num_classes)]
            print(f"提示: 未知数据集 {dataset_name}, 自动推断为 {num_classes} 类")

        class_correct = [0.0] * num_classes
        class_total = [0.0] * num_classes

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += nn.CrossEntropyLoss()(output, target).item()

                # 获取预测结果
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                # === 统计每类的正确率 ===
                # 将 target 和 pred 转为 1D 列表以便遍历统计
                t_flat = target.view(-1)
                p_flat = pred.view(-1)

                for i in range(len(t_flat)):
                    label = t_flat[i].item()
                    if 0 <= label < num_classes:
                        class_total[label] += 1
                        if t_flat[i] == p_flat[i]:
                            class_correct[label] += 1

        avg_loss = 0.0
        accuracy = 0.0
        if len(self.test_loader) > 0 and total > 0:
            avg_loss = test_loss / len(self.test_loader)
            accuracy = 100. * correct / total

        print(
            f"\n测试结果 | Dataset: {dataset_name} | 总Loss: {avg_loss:.4f} | 总Acc: {accuracy:.2f}% | 总样本数: {total}")
        print("-" * 75)
        print(f"{'ID':<4} | {'类别名称':<12} | {'准确率':<10} | {'样本数':<8} | {'占比 (分布)':<12}")
        print("-" * 75)

        printed_count = 0
        for i in range(num_classes):
            if class_total[i] > 0:
                acc_rate = 100 * class_correct[i] / class_total[i]
                # 计算该类样本占总测试集样本的比例
                dist_rate = 100 * class_total[i] / total

                cls_name = classes[i] if i < len(classes) else str(i)
                # 截断过长名称
                if len(cls_name) > 12: cls_name = cls_name[:10] + ".."

                print(f"{i:<4} | {cls_name:<12} | {acc_rate:6.2f}%    | {int(class_total[i]):<8} | {dist_rate:6.2f}%")
                printed_count += 1
            else:
                # 只有当样本数很少时才打印空类别，避免很多空类刷屏
                if num_classes <= 10:
                    cls_name = classes[i] if i < len(classes) else str(i)
                    print(f"{i:<4} | {cls_name:<12} | {'N/A':<10} | {0:<8} | 0.00%")

            # 防止 CIFAR100/ImageNet 刷屏，限制显示行数 (比如显示前 50 个非空类别)
            if printed_count > 50:
                print(f"... (剩余 {num_classes - i - 1} 个类别已省略) ...")
                break

        print("-" * 75)

        return {'loss': avg_loss, 'accuracy': accuracy}
