import numpy as np
import torch
from torch.utils.data import Subset


def split_dataset_to_clients(dataset, num_clients, non_iid=False, alpha=0.5, seed=42, min_require_size=10):
    """
    将数据集划分给多个客户端 (方案一：基于类别的全局Dirichlet切分)
    并且在打印时输出所有类别的统计详情（包括数量为0的类别）
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 获取标签 (兼容 List 和 Tensor)
    if isinstance(dataset.targets, torch.Tensor):
        labels = dataset.targets.numpy()
    else:
        labels = np.array(dataset.targets)

    num_samples = len(dataset)
    # 假设类别从 0 开始连续，如果不是，需要调整这里
    num_classes = len(np.unique(labels))
    client_data = {}

    # ---------------------------------------------------------
    # Non-IID 划分逻辑 (方案一：允许数量不平衡，保证分布准确)
    # ---------------------------------------------------------
    if non_iid:
        client_idcs = [[] for _ in range(num_clients)]

        while True:
            current_client_idcs = [[] for _ in range(num_clients)]

            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)

                # Dirichlet 划分
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

                # 平衡修正：如果某个客户端已经有太多数据，按比例减少它获取的概率（可选，这里保持纯概率）
                proportions = np.array([p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                                        zip(proportions, current_client_idcs)])

                # 防止全零
                if proportions.sum() == 0:
                    proportions = np.ones(num_clients)  # 兜底逻辑

                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_split = np.split(idx_k, proportions)

                for client_id in range(num_clients):
                    current_client_idcs[client_id].extend(idx_split[client_id])

            min_size = min([len(c) for c in current_client_idcs])
            if min_size >= min_require_size:
                client_idcs = current_client_idcs
                break
            else:
                print(f"  [Warning] Random split resulted in a client with {min_size} samples. Retrying...")

    # ---------------------------------------------------------
    # IID 划分逻辑
    # ---------------------------------------------------------
    else:
        idxs = np.arange(num_samples)
        np.random.shuffle(idxs)
        samples_per_client = num_samples // num_clients
        client_idcs = [idxs[i * samples_per_client: (i + 1) * samples_per_client] for i in range(num_clients)]

    # ---------------------------------------------------------
    # 创建 Subset 并输出详细统计（含0样本类别）
    # ---------------------------------------------------------
    for client_id in range(num_clients):
        np.random.shuffle(client_idcs[client_id])
        client_data[client_id] = Subset(dataset, client_idcs[client_id])

    # --- 修改后的打印逻辑 ---
    print(f"\n[Client Split Distribution] (Mode: {'Non-IID' if non_iid else 'IID'}, Alpha: {alpha})")
    for client_id, subset in client_data.items():
        client_indices = subset.indices
        client_labels = labels[client_indices]
        total = len(client_labels)

        # 1. 初始化全零数组
        counts_full = np.zeros(num_classes, dtype=int)

        # 2. 统计当前存在的类别和数量
        unique_cls, counts_actual = np.unique(client_labels, return_counts=True)

        # 3. 填入对应的位置
        counts_full[unique_cls] = counts_actual

        # 4. 格式化输出所有类别 (C0, C1, ... Cn)
        # 使用 :<3 占位符让数字对齐，方便查看
        dist_str = " ".join([f"C{k}:{c:<3}" for k, c in enumerate(counts_full)])

        print(f"  Client {client_id:<2}: Total {total:<5} | {dist_str}")

    return client_data