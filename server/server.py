import torch
import numpy as np
from torch.utils.data import DataLoader
import copy
import torch.nn as nn
import random

class Server:
    def __init__(self, config, test_dataset):
        self.config = config
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else self._select_device(config['device'])

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
        self.communication_history = []  # 每轮通信量记录

        # 添加传输时间统计
        self.round_transmission_times = {}  # 当前轮次各客户端传输时间
        self.max_transmission_times = []  # 每轮最大传输时间记录
        
        # 记录聚合权重
        self.client_weights = {}
        
        # Gilbert-Elliott模型参数初始化
        self.gilbert_elliott_states = {}  # 每个客户端的网络状态
        self._init_gilbert_elliott_params()
        
        print(f"服务器初始化完成, 设备: {self.device}")

    def _init_gilbert_elliott_params(self):
        """初始化Gilbert-Elliott模型状态字典"""
        # 只初始化状态字典，不再设置全局丢包率参数
        self.gilbert_elliott_states = {}  # 每个客户端的网络状态
        print("已初始化Gilbert-Elliott模型状态字典，将使用客户端各自的丢包率")

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
        # 计算模型大小和总下行通信量
        model_size = self.get_model_size()
        # round_down_communication = model_size * len(selected_clients)
        # self.total_down_communication += round_down_communication
        # self.round_down_communication = round_down_communication
        # print(f"模型大小: {model_size / 1024 / 1024:.2f} MB")
        # print(f"本轮下行通信量: {round_down_communication / 1024 / 1024:.2f} MB")
        
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

    def _gilbert_elliott_packet_loss(self, client, layers_to_check):
        """使用Gilbert-Elliott模型判断是否丢包"""

        client_id = client.id
        client_loss_rate = client.packet_loss

        if client_loss_rate == 0:
            return False  # 没有丢包率，不丢包

        # 初始化客户端状态（如果是第一次）
        if client_id not in self.gilbert_elliott_states:
            # 随机初始化为好状态(0)或坏状态(1)
            self.gilbert_elliott_states[client_id] = 0 if random.random() > client_loss_rate else 1
        
        current_state = self.gilbert_elliott_states[client_id]

        # 根据客户端的丢包率计算转移概率
        good_to_bad = 0.5 * client_loss_rate
        bad_to_good = 0.5 - good_to_bad

        # 生成随机数用于状态转移判断
        rand = random.random()

        # 根据当前状态和转移概率决定是否丢包
        if current_state == 0:  # 当前是好状态
            if rand <= good_to_bad:
                # 转为坏状态
                self.gilbert_elliott_states[client_id] = 1
                should_drop = True
            else:
                # 保持好状态
                should_drop = False
        else:  # 当前是坏状态
            if rand <= 1 - bad_to_good:
                # 保持坏状态
                should_drop = True
            else:
                # 转为好状态
                self.gilbert_elliott_states[client_id] = 0
                should_drop = False

        return should_drop

    def receive_local_model(self, client, model_state_dict, num_samples):
        """接收客户端上传的模型更新，将模型分为两部分整体处理丢包"""
        if model_state_dict is None:
            print(f"客户端 {client.id} 上传的模型状态字典为空，跳过更新")
            return False

        client_id = client.id
        # 获取传输协议类型
        transport_type = self.config.get('Transport', 'TCP')
        # 获取需要丢弃的层名称列表
        layers_to_drop = self.config.get('layers_to_drop', [])

        # 初始化客户端的权重记录
        if client_id not in self.client_weights:
            self.client_weights[client_id] = {
                'state_dict': {},
                'num_samples': num_samples
            }
        
        # 将模型层分类
        drop_list_layers = {}
        normal_layers = {}
        
        # 先对层进行分类
        for key, param in model_state_dict.items():
            is_in_drop_list = False
            for layer_pattern in layers_to_drop:
                if layer_pattern in key:
                    is_in_drop_list = True
                    break
            
            if is_in_drop_list:
                drop_list_layers[key] = param
            else:
                normal_layers[key] = param
        
        # 计算两部分的大小
        drop_list_size = sum(param.nelement() * 4 for param in drop_list_layers.values())  # float32 = 4字节
        normal_size = sum(param.nelement() * 4 for param in normal_layers.values())  # float32 = 4字节

        # 初始化传输时间和实际接收的流量
        total_transmission_time = 0.0
        actual_received_size = 0
        
        # 为两部分分别决定是否丢包
        is_drop_list_lost = self._gilbert_elliott_packet_loss(client, ["drop_list_layers"])
        is_normal_lost = self._gilbert_elliott_packet_loss(client, ["normal_layers"])
        
        # 根据传输协议处理丢包和重传
        if transport_type == 'TCP':
            # 处理drop_list中的层
            if drop_list_layers:
                # 计算初次传输时间
                transmission_time, _ = client.calculate_transmission_time(drop_list_size)
                total_transmission_time += transmission_time
                actual_received_size += drop_list_size

                retries_drop = 0
                max_retries = 16

                while is_drop_list_lost and retries_drop < max_retries:
                    retries_drop += 1
                    # 每次重传都要重新计算传输时间
                    retrans_time, _ = client.calculate_transmission_time(drop_list_size)
                    total_transmission_time += retrans_time
                    actual_received_size += drop_list_size
                    print(f"TCP模式：客户端{client_id}的鲁棒层丢包，尝试重传 ({retries_drop}/{max_retries})，累计传输时间: {total_transmission_time:.4f}s")
                    is_drop_list_lost = self._gilbert_elliott_packet_loss(client, ["drop_list_layers"])

                if not is_drop_list_lost:
                    for key, param in drop_list_layers.items():
                        self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(
                        f"TCP模式：客户端{client_id}的鲁棒层成功接收，总传输次数：{retries_drop + 1}，总传输时间：{total_transmission_time:.4f}s")

            # 处理关键层
            if normal_layers:
            # 计算初次传输时间
                transmission_time, _ = client.calculate_transmission_time(normal_size)
                total_transmission_time += transmission_time
                actual_received_size += normal_size

                retries_normal = 0
                max_retries = 16

                while is_normal_lost and retries_normal < max_retries:
                    retries_normal += 1
                    # 每次重传都要重新计算传输时间
                    retrans_time, _ = client.calculate_transmission_time(normal_size)
                    total_transmission_time += retrans_time
                    actual_received_size += normal_size
                    print(f"TCP模式：客户端{client_id}的关键层丢包，尝试重传 ({retries_normal}/{max_retries})，累计传输时间: {total_transmission_time:.4f}s")
                    is_normal_lost = self._gilbert_elliott_packet_loss(client, ["normal_layers"])

                if not is_normal_lost:
                    for key, param in normal_layers.items():
                        self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(f"TCP模式：客户端{client_id}的关键层成功接收，总传输次数：{retries_normal + 1}，总传输时间：{total_transmission_time:.4f}s")



        elif transport_type == 'UDP':
            # UDP模式：只传输一次
            if drop_list_layers:
                transmission_time, _ = client.calculate_transmission_time(drop_list_size)
                total_transmission_time += transmission_time
                if not is_drop_list_lost:
                    actual_received_size += drop_list_size
                    for key, param in drop_list_layers.items():
                        self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(f"UDP模式：客户端{client_id}的鲁棒层成功接收，传输时间: {transmission_time:.4f}s")
                else:
                    print(f"UDP模式：客户端{client_id}的鲁棒层丢包，不重传")

            if normal_layers:
                transmission_time, _ = client.calculate_transmission_time(normal_size)
                total_transmission_time += transmission_time
                if not is_normal_lost:
                    actual_received_size += normal_size
                    for key, param in normal_layers.items():
                        self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(f"UDP模式：客户端{client_id}的关键层成功接收，传输时间: {transmission_time:.4f}s")
                else:
                    print(f"UDP模式：客户端{client_id}的关键层丢包，不重传")


        elif transport_type == 'LTQ':
            # LTQ模式：鲁棒层不重传，关键层重传
            if drop_list_layers:
                transmission_time, _ = client.calculate_transmission_time(drop_list_size)
                total_transmission_time += transmission_time
                actual_received_size += drop_list_size
                if not is_drop_list_lost:
                    for key, param in drop_list_layers.items():
                        self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(f"LTQ模式：客户端{client_id}的鲁棒层成功接收，传输时间: {transmission_time:.4f}s")
                else:
                    print(f"LTQ模式：客户端{client_id}的鲁棒层丢包，不重传")

            if normal_layers:
                # 计算初次传输时间
                transmission_time, _ = client.calculate_transmission_time(normal_size)
                total_transmission_time += transmission_time
                actual_received_size += normal_size
                retries_normal = 0
                max_retries = 16

                while is_normal_lost and retries_normal < max_retries:
                    retries_normal += 1
                    # 每次重传都要重新计算传输时间
                    retrans_time, _ = client.calculate_transmission_time(normal_size)
                    total_transmission_time += retrans_time
                    actual_received_size += normal_size
                    print(f"LTQ模式：客户端{client_id}的关键层丢包，尝试重传 ({retries_normal}/{max_retries})，累计传输时间: {total_transmission_time:.4f}s")
                    is_normal_lost = self._gilbert_elliott_packet_loss(client, ["normal_layers"])

                if not is_normal_lost:
                    for key, param in normal_layers.items():
                        self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(f"LTQ模式：客户端{client_id}的关键层成功接收，总传输次数：{retries_normal + 1}，总传输时间：{total_transmission_time:.4f}s")

        # 记录该客户端的传输时间
        self.round_transmission_times[client_id] = total_transmission_time

        # 更新通信量统计（只计算实际传输的字节数）
        self.total_up_communication += actual_received_size
        self.round_up_communication += actual_received_size

        if not self.client_weights[client_id]['state_dict']:
            print(f"客户端 {client_id} 的所有层传输失败，无法使用该客户端的更新")
            del self.client_weights[client_id]
            if client_id in self.round_transmission_times:
                del self.round_transmission_times[client_id]
            return False

        print(f"服务器已接收客户端 {client_id} 的更新:")
        print(f"  实际接收数据量: {actual_received_size / 1024 / 1024:.2f} MB")
        print(f"  总传输时间: {total_transmission_time:.4f}s")
        return True

    def finalize_round_transmission_time(self):
        """完成本轮传输，记录最大传输时间"""
        if self.round_transmission_times:
            max_time = max(self.round_transmission_times.values())
            max_client = max(self.round_transmission_times, key=self.round_transmission_times.get)
            self.max_transmission_times.append(max_time)

            print(f"本轮传输完成，最慢客户端: {max_client}，传输时间: {max_time:.4f}s")
            print(f"所有客户端传输时间: {self.round_transmission_times}")
        else:
            self.max_transmission_times.append(0.0)

    def next_round(self):
        """准备下一轮训练"""
        # 记录本轮通信量
        self.communication_history.append({
            'up_communication': self.round_up_communication / (1024 * 1024),  # MB
        })

        # 重置本轮统计
        self.round_up_communication = 0
        self.round_transmission_times = {}  # 清空本轮传输时间记录
        self.client_weights = {}

    def get_communication_stats(self):
        """获取通信统计信息"""
        return {
            "总上行通信量(MB)": self.total_up_communication / (1024 * 1024),
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
