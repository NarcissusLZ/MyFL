import numpy as np
import torch
from torch.utils.data import Subset

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