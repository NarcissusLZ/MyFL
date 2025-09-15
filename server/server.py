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
        """接收客户端上传的模型更新，使用Gilbert-Elliott模型模拟丢包并根据传输协议处理重传"""
        if model_state_dict is None:
            raise ValueError(f"客户端 {client.id} 上传的模型状态字典为空，无法接收更新")

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

        # 计算上行通信量
        received_model_size = 0

        # 遍历模型的每一层
        for key, param in model_state_dict.items():
            # 计算当前层的数据大小
            layer_size = param.nelement() * 4  # float32 = 4字节

            # 检查当前层是否在特殊处理列表中
            is_in_drop_list = False
            for layer_pattern in layers_to_drop:
                if layer_pattern in key:
                    is_in_drop_list = True
                    break

            # 使用Gilbert-Elliott模型决定是否丢包
            is_packet_lost = self._gilbert_elliott_packet_loss(client, [key])

            # 初始传输计入通信量
            received_model_size += layer_size

            # 根据传输协议处理丢包和重传
            if transport_type == 'TCP':
                # TCP模式：丢包后重传，最多重传16次
                retries = 0
                max_retries = 16

                while is_packet_lost and retries < max_retries:
                    # 重传
                    retries += 1
                    print(f"TCP模式：客户端{client_id}的层{key}丢包，尝试重传 ({retries}/{max_retries})")

                    # 每次重传都计入通信流量
                    received_model_size += layer_size

                    # 重传可能再次丢包，继续使用Gilbert-Elliott模型
                    is_packet_lost = self._gilbert_elliott_packet_loss(client, [key])

                if not is_packet_lost:
                    # 传输成功
                    self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(f"TCP模式：客户端{client_id}的层{key}成功接收，总传输次数：{retries + 1}")
                else:
                    print(f"TCP模式：客户端{client_id}的层{key}传输失败，达到最大重传次数")

            elif transport_type == 'UDP':
                # UDP模式：丢包后不重传
                if not is_packet_lost:
                    # 传输成功
                    self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(f"UDP模式：客户端{client_id}的层{key}成功接收")
                else:
                    print(f"UDP模式：客户端{client_id}的层{key}丢包，不重传")

            elif transport_type == 'LTQ':
                # LTQ模式：只对非layers_to_drop的层进行重传
                if is_packet_lost:
                    if not is_in_drop_list:
                        # 非drop列表中的层需要重传
                        retries = 0
                        max_retries = 16

                        while is_packet_lost and retries < max_retries:
                            # 重传计入通信流量
                            received_model_size += layer_size
                            retries += 1
                            print(
                                f"LTQ模式：客户端{client_id}的层{key}丢包(非drop列表)，尝试重传 ({retries}/{max_retries})")

                            # 重传可能再次丢包，继续使用Gilbert-Elliott模型
                            is_packet_lost = self._gilbert_elliott_packet_loss(client, [key])

                        if not is_packet_lost:
                            # 重传成功
                            self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                            print(f"LTQ模式：客户端{client_id}的层{key}成功接收，总传输次数：{retries + 1}")
                        else:
                            print(f"LTQ模式：客户端{client_id}的层{key}传输失败，达到最大重传次数")
                    else:
                        # 在drop列表中的层不重传
                        print(f"LTQ模式：客户端{client_id}的层{key}在丢弃列表中且丢包，不重传")
                else:
                    # 首次传输成功
                    self.client_weights[client_id]['state_dict'][key] = copy.deepcopy(param.to(self.device))
                    print(f"LTQ模式：客户端{client_id}的层{key}成功接收")

        # 更新通信量统计
        self.total_up_communication += received_model_size
        self.round_up_communication += received_model_size

        # 检查是否有任何层被成功接收
        if not self.client_weights[client_id]['state_dict']:
            print(f"客户端 {client_id} 的所有层传输失败，无法使用该客户端的更新")
            del self.client_weights[client_id]
            return False

        print(f"服务器已接收客户端 {client_id} 的更新，接收数据量: {received_model_size / 1024 / 1024:.2f} MB")
        return True

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

    def next_round(self):
        """准备下一轮训练"""
        # 记录本轮通信量
        self.communication_history.append({
            #'down_communication': self.round_down_communication / (1024 * 1024),  # MB
            'up_communication': self.round_up_communication / (1024 * 1024),  # MB
            #'total_communication': (self.round_down_communication + self.round_up_communication) / (1024 * 1024)  # MB
        })
        
        # 重置本轮通信量统计
        # self.round_down_communication = 0
        self.round_up_communication = 0
        
        # 清除客户端权重
        self.client_weights = {}

    def get_communication_stats(self):
        """获取通信统计信息"""
        return {
            # "总下行通信量(MB)": self.total_down_communication / (1024 * 1024),
            "总上行通信量(MB)": self.total_up_communication / (1024 * 1024),
            # "总通信量(MB)": (self.total_down_communication + self.total_up_communication) / (1024 * 1024),
            "每轮通信量记录": self.communication_history
        }