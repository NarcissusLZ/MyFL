import numpy as np
import torch
from torch.utils.data import Subset


def split_dataset_to_clients(dataset, num_clients, non_iid=False, alpha=0.5, seed=42):
    """
    将数据集划分给多个客户端
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_samples = len(dataset)
    client_data = {}
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    # 所有样本的索引
    all_idxs = np.arange(num_samples)

    if non_iid:
        # 非独立同分布划分，但保证每个客户端数据量大致相等
        samples_per_client = num_samples // num_clients
        client_idxs = [[] for _ in range(num_clients)]

        # 按类别对索引进行分组
        class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]

        # 为每个客户端生成类别分布
        # proportions 的维度是 (num_clients, num_classes)
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes), num_clients)

        # 确保每个客户端的样本总数不会超过其应得的份额
        # 这也防止了某个客户端因为类别偏好而取走过多样本
        proportions = proportions / proportions.sum(axis=1, keepdims=True)

        # 记录每个类别还剩多少可用索引
        remaining_class_counts = [len(idxs) for idxs in class_idxs]

        # 为每个客户端分配数据
        for client_id in range(num_clients):
            client_prop = proportions[client_id]
            target_num_samples = samples_per_client

            # 根据客户端的类别偏好，计算每个类别应分配的样本数
            target_samples_per_class = np.round(client_prop * target_num_samples).astype(int)

            # 修正总数，确保与 target_num_samples 一致
            correction = target_num_samples - target_samples_per_class.sum()
            if correction > 0:
                add_indices = np.random.choice(num_classes, correction, p=client_prop)
                np.add.at(target_samples_per_class, add_indices, 1)
            elif correction < 0:
                # 优先从数量多的类别中减去
                sorted_indices = np.argsort(-target_samples_per_class)
                for i in range(abs(correction)):
                    target_samples_per_class[sorted_indices[i % num_classes]] -= 1

            client_samples = []
            # 从各类别的池中抽取样本
            for c in range(num_classes):
                num_to_take = min(target_samples_per_class[c], remaining_class_counts[c])

                if num_to_take > 0:
                    # 从类别c的可用索引中随机抽取
                    taken_idxs = np.random.choice(class_idxs[c], num_to_take, replace=False)
                    client_samples.extend(taken_idxs)

                    # 更新类别c的可用索引和数量
                    class_idxs[c] = np.setdiff1d(class_idxs[c], taken_idxs, assume_unique=True)
                    remaining_class_counts[c] -= num_to_take

            client_idxs[client_id] = client_samples

        # 将剩余未分配的样本（由于取整或样本耗尽导致）轮流分给客户端
        all_assigned_idxs = np.concatenate(client_idxs)
        unassigned_idxs = np.setdiff1d(all_idxs, all_assigned_idxs)

        for i, idx in enumerate(unassigned_idxs):
            client_idxs[i % num_clients].append(idx)

    else:
        # 改进的独立同分布划分 - 保证每个客户端类别分布均匀
        class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]
        client_idxs = [[] for _ in range(num_clients)]

        # 为每个类别平均分配给所有客户端
        for c in range(num_classes):
            idxs = class_idxs[c]
            np.random.shuffle(idxs)

            # 计算每个客户端应该分配多少该类别的样本
            samples_per_client_class = len(idxs) // num_clients
            remainder = len(idxs) % num_clients

            start = 0
            for client_id in range(num_clients):
                # 如果有余数，前几个客户端多分配一个样本
                end = start + samples_per_client_class + (1 if client_id < remainder else 0)
                client_idxs[client_id].extend(idxs[start:end])
                start = end

    # 创建客户端数据集
    for client_id in range(num_clients):
        np.random.shuffle(client_idxs[client_id])
        client_data[client_id] = Subset(dataset, client_idxs[client_id])

    return client_data
