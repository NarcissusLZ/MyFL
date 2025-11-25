import torch
import numpy as np
from torch.utils.data import DataLoader
import copy
import torch.nn as nn
import random

# 定义数据包大小常量（1500字节）
PACKET_SIZE = 1500

# 定义GE状态常量
GE_GOOD_STATE = 0  # 好状态 (低丢包率)
GE_BAD_STATE = 1  # 坏状态 (高丢包率)


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
        # self.gilbert_elliott_states 现在记录的是客户端当前的 GE 状态 (0 或 1)
        self.gilbert_elliott_states = {}
        self.client_random_generators = {}  # 每个客户端的独立随机数生成器
        self._init_gilbert_elliott_params()

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
        self.gilbert_elliott_states = {}  # 每个客户端的网络状态 (初始状态在首次使用时确定)
        self.client_random_generators = {}
        print("已初始化Gilbert-Elliott模型状态字典，将使用客户端各自的丢包率")

    def _get_client_random_generator(self, client_id):
        """为每个客户端获取或创建独立的随机数生成器"""
        if client_id not in self.client_random_generators:
            client_seed = self.random_seed + hash(str(client_id)) % 10000
            self.client_random_generators[client_id] = random.Random(client_seed)
        return self.client_random_generators[client_id]

    def get_model_size(self):
        """计算模型参数大小（字节）"""
        size_bytes = 0
        for param in self.global_model.parameters():
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

    def _gilbert_elliott_state_transition(self, client_id, client_loss_rate):
        """
        Gilbert-Elliott 模型状态转移函数。
        根据当前状态和转移概率，更新状态并返回新状态。
        """
        if client_loss_rate == 0:
            # 如果没有丢包率，则强制保持好状态
            self.gilbert_elliott_states[client_id] = GE_GOOD_STATE
            return GE_GOOD_STATE

        client_rng = self._get_client_random_generator(client_id)

        # 初始化客户端状态（如果是第一次）
        if client_id not in self.gilbert_elliott_states:
            # 初始状态基于平均丢包率随机分配 (0: 好状态, 1: 坏状态)
            self.gilbert_elliott_states[
                client_id] = GE_BAD_STATE if client_rng.random() < client_loss_rate else GE_GOOD_STATE

        current_state = self.gilbert_elliott_states[client_id]

        # 转移概率计算 (基于 P_avg = G / (G + B) 的简化)
        good_to_bad = client_loss_rate * 0.5
        bad_to_good = (1 - client_loss_rate) * 0.5

        # 确保转移概率在 [0, 1] 范围内
        good_to_bad = max(0.0, min(1.0, good_to_bad))
        bad_to_good = max(0.0, min(1.0, bad_to_good))

        rand = client_rng.random()
        new_state = current_state

        if current_state == GE_GOOD_STATE:
            if rand < good_to_bad:
                new_state = GE_BAD_STATE
        else:  # 当前是坏状态
            if rand < bad_to_good:
                new_state = GE_GOOD_STATE
            # 否则保持坏状态

        self.gilbert_elliott_states[client_id] = new_state
        return new_state

    def _is_packet_lost_in_state(self, state, client_rng):
        """根据 Gilbert-Elliott 状态判断一个数据包是否丢失。"""
        # 假设：
        # - 好状态 (GE_GOOD_STATE=0) 丢包率 P_g = 0.01 (极低)
        # - 坏状态 (GE_BAD_STATE=1) 丢包率 P_b = 0.8 (极高)
        P_g = 0.01
        P_b = 0.8

        if state == GE_GOOD_STATE:
            return client_rng.random() < P_g
        else:
            return client_rng.random() < P_b

    def _simulate_and_get_lost_percentage(self, client, client_id, layer_size, layer_name, should_retransmit,
                                          max_retries):
        """
        运行包级模拟，并返回总传输统计和最终未接收包的百分比。
        修改自原 _simulate_packet_transmission

        返回:
            total_time: 传输消耗的总时间
            total_size: 传输消耗的总字节数
            transmission_count: 总传输次数 (数据包数量)
            lost_percentage: 最终未成功接收包的百分比
        """
        if layer_size == 0:
            return 0.0, 0, 0, 0.0

        # 计算数据包数量
        num_packets = max(1, (layer_size + PACKET_SIZE - 1) // PACKET_SIZE)

        # 记录每个数据包的丢包状态 (True: 丢失, False: 成功接收)
        packet_lost_status = [False] * num_packets
        current_lost_count = 0
        total_time = 0.0
        total_size = 0
        transmission_count = 0
        client_rng = self._get_client_random_generator(client_id)

        # 计算单个数据包的传输时间
        time_per_packet, _ = client.calculate_transmission_time(PACKET_SIZE)

        # --- 1. 初始传输 (所有数据包) ---
        for i in range(num_packets):
            transmission_count += 1
            total_time += time_per_packet
            total_size += PACKET_SIZE

            # 状态转移和丢包判断
            new_state = self._gilbert_elliott_state_transition(client_id, client.packet_loss)
            is_lost = self._is_packet_lost_in_state(new_state, client_rng)

            if is_lost:
                packet_lost_status[i] = True
                current_lost_count += 1

        retries = 0

        # --- 2. 重传阶段 ---
        while current_lost_count > 0 and should_retransmit and retries < max_retries:
            retries += 1
            packets_to_retransmit = [i for i, lost in enumerate(packet_lost_status) if lost]

            # 对每个丢失的包进行重传
            for i in packets_to_retransmit:
                # 记录重传包的传输时间和大小
                total_time += time_per_packet
                total_size += PACKET_SIZE
                transmission_count += 1

                # 状态转移和丢包判断 (重传也经历状态转移)
                new_state = self._gilbert_elliott_state_transition(client_id, client.packet_loss)
                is_lost = self._is_packet_lost_in_state(new_state, client_rng)

                if not is_lost:
                    # 重传成功
                    packet_lost_status[i] = False
                    current_lost_count -= 1

            # 打印重传信息
            print(
                f"  客户端{client_id}的{layer_name}重传 {len(packets_to_retransmit)} 个包 ({retries}/{max_retries})，累计丢失: {current_lost_count}")

        lost_percentage = current_lost_count / num_packets if num_packets > 0 else 0.0

        if current_lost_count > 0:
            print(f"  客户端{client_id}的{layer_name}最终丢失 {current_lost_count} 个包 ({lost_percentage * 100:.2f}%)")
        elif not should_retransmit and current_lost_count > 0:
            print(f"  客户端{client_id}的{layer_name}不重传，最终丢失 {current_lost_count} 个包，将应用部分替换。")

        # 注意：这里不再返回 is_successful，而是返回 lost_percentage
        return total_time, total_size, transmission_count, lost_percentage

    # 新增核心方法：应用部分替换逻辑
    def _apply_partial_replacement(self, client_id, layers_dict, lost_percentage):
        """
        根据丢失的包百分比，对客户端上传的层进行参数替换。

        逻辑：将该层总参数量的 (lost_percentage) 部分替换为全局模型的旧参数。
        简化假设：丢失的参数块位于张量的末尾部分。
        """
        if not layers_dict:
            return

        if lost_percentage == 0.0:
            # 如果没有丢失，则完全使用客户端更新
            for key, param in layers_dict.items():
                self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
            return

        # 获取全局模型的旧参数，用于替代
        global_state_dict = self.global_model.state_dict()

        for key, client_param in layers_dict.items():

            # 获取参数张量
            client_tensor = client_param.to(self.device)
            global_tensor = global_state_dict[key].to(self.device)

            # 确定替换点 (替换百分比对应的元素数量)
            num_elements = client_tensor.numel()
            num_elements_to_replace = int(num_elements * lost_percentage)

            if num_elements_to_replace == 0:
                # 丢失数量小于一个元素，仍全部接收
                self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(client_tensor)
                continue

            # --- 执行替换操作 ---

            # 1. 将张量扁平化 (方便按索引替换)
            client_flat = client_tensor.flatten()
            global_flat = global_tensor.flatten()

            # 2. 确定替换范围
            start_index = num_elements - num_elements_to_replace

            # 3. 用全局模型的旧参数替换客户端更新的对应部分
            client_flat[start_index:] = global_flat[start_index:].clone()

            # 4. 将扁平化后的张量重塑回原始形状
            replaced_tensor = client_flat.view_as(client_tensor)

            # 5. 更新到客户端权重字典
            self.client_weights[client_id]['state_dict'][key] = replaced_tensor

        print(
            f"  客户端 {client_id} 的层 {list(layers_dict.keys())[0]}... (等) 已应用 {lost_percentage * 100:.2f}% 的旧参数替换。")

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

    def receive_local_model(self, client, model_state_dict, num_samples, current_round=None, current_accuracy=None):
        """接收客户端上传的模型更新，应用部分替换逻辑"""
        if model_state_dict is None:
            print(f"客户端 {client.id} 上传的模型状态字典为空，跳过更新")
            return False

        client_id = client.id
        transport_type = self.config.get('Transport', 'TCP')
        layers_to_drop = self.config.get('layers_to_drop', [])  # 被定义为鲁棒层

        # 初始化客户端的权重记录
        if client_id not in self.client_weights:
            self.client_weights[client_id] = {
                'state_dict': copy.deepcopy(self.global_model.state_dict()),
                'num_samples': num_samples
            }
        else:
            self.client_weights[client_id]['num_samples'] = num_samples

        # 将模型层分类
        robust_layers = {}
        critical_layers = {}

        for key, param in model_state_dict.items():
            is_robust_layer = False
            for layer_pattern in layers_to_drop:
                if layer_pattern in key:
                    is_robust_layer = True
                    break

            if is_robust_layer:
                robust_layers[key] = param
            else:
                critical_layers[key] = param

        # 计算两部分的大小
        robust_layer_size_bytes = sum(param.nelement() * 4 for param in robust_layers.values())
        critical_layer_size_bytes = sum(param.nelement() * 4 for param in critical_layers.values())

        # 初始化统计变量
        total_transmission_time = 0.0
        actual_received_size = 0
        robust_layer_received_size = 0
        critical_layer_received_size = 0
        robust_transmission_count = 0
        critical_transmission_count = 0

        # --- 确定传输策略 ---
        if transport_type == 'LTQ':
            ltq_phase = self.get_ltq_strategy_phase(current_round, current_accuracy)
            should_r_retransmit = (ltq_phase == 'early')
            should_c_retransmit = (ltq_phase != 'late')
            max_r_retries = 16 if should_r_retransmit else 0
            max_c_retries = 16 if should_c_retransmit else 0
        elif transport_type == 'TCP':
            should_r_retransmit, should_c_retransmit = True, True
            max_r_retries, max_c_retries = 16, 16
        else:  # UDP
            should_r_retransmit, should_c_retransmit = False, False
            max_r_retries, max_c_retries = 0, 0

        # --- 运行包级传输模拟并获取丢失百分比 ---

        # 鲁棒层模拟
        r_time, r_size, r_count, r_lost_percentage = self._simulate_and_get_lost_percentage(
            client, client_id, robust_layer_size_bytes, '鲁棒层', should_r_retransmit, max_r_retries)

        # 关键层模拟
        c_time, c_size, c_count, c_lost_percentage = self._simulate_and_get_lost_percentage(
            client, client_id, critical_layer_size_bytes, '关键层', should_c_retransmit, max_c_retries)

        # 汇总统计
        total_transmission_time = r_time + c_time
        actual_received_size = r_size + c_size
        robust_layer_received_size = r_size
        critical_layer_received_size = c_size
        robust_transmission_count = r_count
        critical_transmission_count = c_count

        # --- 应用部分替换逻辑 ---

        # 1. 处理鲁棒层
        self._apply_partial_replacement(client_id, robust_layers, r_lost_percentage)

        # 2. 处理关键层
        self._apply_partial_replacement(client_id, critical_layers, c_lost_percentage)

        # 记录统计信息
        self.round_transmission_times[client_id] = total_transmission_time
        self.total_up_communication += actual_received_size
        self.round_up_communication += actual_received_size
        self.total_robust_communication += robust_layer_received_size
        self.total_critical_communication += critical_layer_received_size
        self.round_robust_communication += robust_layer_received_size
        self.round_critical_communication += critical_layer_received_size
        self.total_robust_transmission_count += robust_transmission_count
        self.total_critical_transmission_count += critical_transmission_count
        self.round_robust_transmission_count += robust_transmission_count
        self.round_critical_transmission_count += critical_transmission_count

        # 打印输出
        print(f"服务器已接收客户端 {client_id} 的更新:")
        print(f"  实际传输数据量: {actual_received_size / 1024 / 1024:.2f} MB")
        print(f"  鲁棒层流量: {robust_layer_received_size / 1024 / 1024:.2f} MB")
        print(f"  关键层流量: {critical_layer_received_size / 1024 / 1024:.2f} MB")
        print(f"  总传输时间: {total_transmission_time:.2f}s")

        return True

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
        # 记录本轮通信量
        self.communication_history.append({
            'up_communication': self.round_up_communication / (1024 * 1024),  # MB
            'robust_layer_communication': self.round_robust_communication / (1024 * 1024),  # MB
            'critical_layer_communication': self.round_critical_communication / (1024 * 1024),  # MB
        })

        # 重置本轮统计
        self.round_up_communication = 0
        self.round_robust_communication = 0
        self.round_critical_communication = 0
        self.round_robust_transmission_count = 0
        self.round_critical_transmission_count = 0
        self.round_transmission_times = {}  # 清空本轮传输时间记录
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