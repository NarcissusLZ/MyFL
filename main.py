import torch
import yaml
import os
import time
import numpy as np
import copy
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# 导入自定义模块
from data.getdata import get_dataset
from models.vgg16 import CIFAR10_VGG16
from client.client import Client
from server.server import Server


def load_config(config_path='config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


#该函数应单独置于data文件夹，而非在main中
def split_dataset_to_clients(dataset, num_clients, non_iid=False, alpha=0.5, seed=42):
    """
    将数据集划分给多个客户端
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    client_data = {}
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    if non_iid:
        # 非独立同分布划分 (狄利克雷分布)
        class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients), num_classes)

        client_idxs = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idxs = class_idxs[c]
            np.random.shuffle(idxs)
            dist = proportions[c]
            dist_cumsum = np.cumsum(dist)
            dist_cumsum = dist_cumsum * len(idxs)
            dist_cumsum = dist_cumsum.astype(int)

            start = 0
            for client_id in range(num_clients):
                end = dist_cumsum[client_id]
                client_idxs[client_id].extend(idxs[start:end])
                start = end
    else:
        # 独立同分布划分
        idxs = np.random.permutation(len(dataset))
        client_size = len(dataset) // num_clients
        client_idxs = []
        for client_id in range(num_clients):
            start = client_id * client_size
            end = (client_id + 1) * client_size
            if client_id == num_clients - 1:
                end = len(dataset)
            client_idxs.append(idxs[start:end])

    # 创建客户端数据集
    for client_id in range(num_clients):
        np.random.shuffle(client_idxs[client_id])
        client_data[client_id] = Subset(dataset, client_idxs[client_id])

    return client_data

def main():
    # 1. 加载配置
    config = load_config()
    logger.info("=" * 50)
    logger.info("联邦学习配置:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

    # 2. 准备数据集
    logger.info("准备数据集...")
    train_dataset, test_dataset = get_dataset(
        dir=config['data_dir'],
        name=config['dataset']
    )
    logger.info(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 3. 划分数据集给多个客户端
    logger.info("划分数据集给客户端...")
    client_data = split_dataset_to_clients(
        train_dataset,
        num_clients=config['num_clients'],
        non_iid=config['non_iid'],
        alpha=config['non_iid_alpha']
    )

    # 打印每个客户端的数据量
    logger.info("\n客户端数据分布:")
    for client_id, data_subset in client_data.items():
        logger.info(f"客户端 {client_id}: {len(data_subset)} 个样本")

    # 4. 初始化服务器
    logger.info("\n初始化服务器...")
    server = Server(config, test_dataset)

    # 5. 初始化客户端
    logger.info("初始化客户端...")
    clients = {}
    for client_id, data_subset in client_data.items():
        clients[client_id] = Client(
            id=client_id,
            config=config,
            local_dataset=data_subset
        )
    logger.info(f"已初始化 {len(clients)} 个客户端")

    # 6. 训练循环
    logger.info("\n开始联邦学习训练...")
    history = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'clients_selected': []
    }

    start_time = time.time()

    for round in range(config['num_rounds']):
        round_start = time.time()
        logger.info(f"\n=== 训练轮次 {round + 1}/{config['num_rounds']} ===")

        # a. 服务器选择客户端
        selected_clients = server.select_clients(
            list(clients.values()),
            fraction=config['client_fraction']
        )
        client_ids = [client.id for client in selected_clients]
        logger.info(f"本轮选中客户端: {client_ids}")
        history['clients_selected'].append(len(selected_clients))

        # b. 服务器发送全局模型状态字典给选中的客户端
        global_state = server.global_model.state_dict()
        for client in selected_clients:
            client.get_model(copy.deepcopy(global_state))

        # c. 客户端本地训练
        client_updates = {}
        for client in selected_clients:
            local_state = client.local_train()
            client_updates[client.id] = {
                'model_state': local_state,
                'num_samples': len(client.local_dataset)
            }

        # d. 客户端上传模型更新给服务器
        for client_id, update in client_updates.items():
            server.get_local_model(
                client_id,
                update['model_state'],
                update['num_samples']
            )

        # e. 服务器聚合模型更新
        logger.info("服务器聚合模型更新...")
        server.fed_avg()

        # f. 服务器测试全局模型
        logger.info("测试全局模型性能...")
        test_results = server.test_model()

        # g. 记录结果
        history['round'].append(round + 1)
        history['accuracy'].append(test_results['accuracy'])
        history['loss'].append(test_results['loss'])

        # h. 进入下一轮
        server.next_round()

        round_time = time.time() - round_start
        logger.info(f"本轮完成，耗时: {round_time:.2f}秒")

    # 7. 训练结束
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 50)
    logger.info(f"联邦学习训练完成!")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"最终准确率: {history['accuracy'][-1]:.2f}%")

    # 8. 保存结果
    save_results(config, history, server)

    return history


def save_results(config, history, server):
    """保存训练结果和模型"""
    # 创建结果目录
    result_dir = config.get('result_dir', 'results')
    os.makedirs(result_dir, exist_ok=True)

    # 生成时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # 保存训练历史
    history_path = os.path.join(result_dir, f"history_{timestamp}.npy")
    np.save(history_path, history)
    logger.info(f"训练历史保存至: {history_path}")

    # 保存模型
    model_path = os.path.join(result_dir, f"global_model_{timestamp}.pth")
    torch.save(server.global_model.state_dict(), model_path)
    logger.info(f"全局模型保存至: {model_path}")

    # 保存配置文件
    config_path = os.path.join(result_dir, f"config_{timestamp}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"配置文件保存至: {config_path}")

    # 打印最终性能
    logger.info("\n训练轮次性能:")
    for i in range(len(history['round'])):
        logger.info(f"轮次 {history['round'][i]}: 准确率={history['accuracy'][i]:.2f}%, 损失={history['loss'][i]:.4f}")


if __name__ == "__main__":
    history = main()
