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

    # 计算每个客户端应得的精确样本数
    samples_per_client = num_samples // num_clients

    # 所有样本的索引
    all_idxs = np.arange(num_samples)

    if non_iid:
        # 非独立同分布划分，严格保证每个客户端数据量相等
        client_idxs = [[] for _ in range(num_clients)]

        # 按类别对索引进行分组
        class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]

        # 为每个类别的索引打乱顺序
        for c in range(num_classes):
            np.random.shuffle(class_idxs[c])

        # 为每个客户端生成类别分布
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes), num_clients)

        # 为每个客户端分配数据
        for client_id in range(num_clients):
            client_prop = proportions[client_id]

            # 根据客户端的类别偏好，计算每个类别应分配的样本数
            target_samples_per_class = np.round(client_prop * samples_per_client).astype(int)

            # 严格修正总数，确保与 samples_per_client 完全一致
            diff = samples_per_client - target_samples_per_class.sum()

            while diff != 0:
                if diff > 0:
                    # 需要增加样本，优先给比例高的类别
                    sorted_indices = np.argsort(-client_prop)
                    for idx in sorted_indices:
                        if diff == 0:
                            break
                        target_samples_per_class[idx] += 1
                        diff -= 1
                else:
                    # 需要减少样本，优先从数量多的类别减
                    sorted_indices = np.argsort(-target_samples_per_class)
                    for idx in sorted_indices:
                        if diff == 0 or target_samples_per_class[idx] == 0:
                            break
                        target_samples_per_class[idx] -= 1
                        diff += 1

            # 从各类别的池中抽取样本
            for c in range(num_classes):
                num_to_take = target_samples_per_class[c]

                if num_to_take > 0:
                    # 如果该类别样本不足，从其他类别补充
                    available = len(class_idxs[c])

                    if available >= num_to_take:
                        taken_idxs = class_idxs[c][:num_to_take]
                        class_idxs[c] = class_idxs[c][num_to_take:]
                    else:
                        # 取完当前类别的所有样本
                        taken_idxs = class_idxs[c].tolist()
                        class_idxs[c] = np.array([], dtype=int)

                        # 从其他有剩余样本的类别中补充
                        shortage = num_to_take - available
                        for other_c in range(num_classes):
                            if shortage == 0:
                                break
                            if len(class_idxs[other_c]) > 0:
                                supplement = min(shortage, len(class_idxs[other_c]))
                                taken_idxs.extend(class_idxs[other_c][:supplement])
                                class_idxs[other_c] = class_idxs[other_c][supplement:]
                                shortage -= supplement

                    client_idxs[client_id].extend(taken_idxs)

            # 最终检查：如果还不够，从剩余样本中随机补充
            current_size = len(client_idxs[client_id])
            if current_size < samples_per_client:
                remaining = []
                for c in range(num_classes):
                    remaining.extend(class_idxs[c])

                shortage = samples_per_client - current_size
                if len(remaining) >= shortage:
                    supplement = np.random.choice(remaining, shortage, replace=False)
                    client_idxs[client_id].extend(supplement)

                    # 从 class_idxs 中移除已使用的样本
                    for c in range(num_classes):
                        class_idxs[c] = np.setdiff1d(class_idxs[c], supplement)

    else:
        # 独立同分布划分
        class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]
        client_idxs = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idxs = class_idxs[c]
            np.random.shuffle(idxs)

            samples_per_client_class = len(idxs) // num_clients
            remainder = len(idxs) % num_clients

            start = 0
            for client_id in range(num_clients):
                end = start + samples_per_client_class + (1 if client_id < remainder else 0)
                client_idxs[client_id].extend(idxs[start:end])
                start = end

    # 创建客户端数据集，并确保每个客户端数据量完全一致
    for client_id in range(num_clients):
        # 严格截取或补充到 samples_per_client
        if len(client_idxs[client_id]) > samples_per_client:
            client_idxs[client_id] = client_idxs[client_id][:samples_per_client]

        np.random.shuffle(client_idxs[client_id])
        client_data[client_id] = Subset(dataset, client_idxs[client_id])

    # --- 新增：打印每个客户端的数据分布比例 ---
    print(f"\n[Client Split Distribution] (Mode: {'Non-IID' if non_iid else 'IID'})")
    for client_id, subset in client_data.items():
        # 获取该客户端所有样本的标签
        client_indices = subset.indices
        client_labels = labels[client_indices] # labels 是函数开头定义的 dataset.targets

        unique_cls, counts = np.unique(client_labels, return_counts=True)
        total = len(client_labels)

        dist_str = ", ".join([f"C{u}:{c}({c/total:.0%})" for u, c in zip(unique_cls, counts)])
        print(f"  Client {client_id:<2}: Total {total:<5} | {dist_str}")
    # ------------------------------------------

    return client_data
