import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import time
import copy
import torch.nn as nn


class Server:
    def __init__(self, config, test_dataset):
        self.config = config
        self.device = self._select_device(config['server_device'])

        # 初始化全局模型
        self.global_model = self._create_model()
        self.global_model.to(self.device)

        # 准备测试数据集
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=False
        )

        # 记录聚合权重
        self.client_weights = {}

        print(f"服务器初始化完成, 设备: {self.device}")

    def _select_device(self, device_config):
        """选择设备"""
        if device_config == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_config == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _create_model(self):
        """创建模型"""
        # 根据您的模型配置创建模型
        if self.config['model'] == 'CIFAR10_VGG16':
            from models.vgg16 import CIFAR10_VGG16
            return CIFAR10_VGG16()
        else:
            raise ValueError(f"未知模型: {self.config['model']}")

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

    def get_local_model(self, client_id, model_state_dict, num_samples):
        """接收客户端上传的模型更新"""
        # 存储模型状态字典和样本数量
        self.client_weights[client_id] = {
            'state_dict': copy.deepcopy(model_state_dict),
            'num_samples': num_samples
        }
        print(f"服务器已接收客户端 {client_id} 的更新")

    def fed_avg(self):
        """执行FedAvg聚合算法"""
        print("开始模型聚合...")

        # 计算总样本数
        total_samples = sum(w['num_samples'] for w in self.client_weights.values())

        # 初始化全局模型参数
        global_state = self.global_model.state_dict()

        # 重置所有参数为零（跳过不需要聚合的参数）
        for key in global_state:
            if 'num_batches_tracked' not in key:
                global_state[key] = torch.zeros_like(global_state[key])

        # 加权平均
        for client_id, client_data in self.client_weights.items():
            client_weight = client_data['num_samples'] / total_samples
            client_state = client_data['state_dict']

            for key in global_state:
                # 跳过BatchNorm的num_batches_tracked参数
                if 'num_batches_tracked' in key:
                    # 对于num_batches_tracked，直接使用第一个客户端的值
                    if client_id == list(self.client_weights.keys())[0]:
                        global_state[key] = client_state[key].clone()
                    continue

                # 确保设备一致
                if client_state[key].device != global_state[key].device:
                    client_state[key] = client_state[key].to(global_state[key].device)

                # 加权累加
                global_state[key] += client_weight * client_state[key]

        # 更新全局模型
        self.global_model.load_state_dict(global_state)
        print("模型聚合完成")

    def test_model(self):
        """在测试集上评估全局模型"""
        print("开始模型测试...")
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
        self.client_weights = {}

        # 释放MPS内存（如果使用Apple Silicon）
        if self.device.type == 'mps':
            torch.mps.empty_cache()
