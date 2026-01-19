import os
import torch
import torch.nn as nn
import torchaudio
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import pyarrow.csv as pv
import pyarrow as pa
import gc


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# --- 新增：IoT-23 数据集封装类 ---
class IoT23Dataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features: 预处理后的 numpy array 或 tensor
            labels: 预处理后的标签列表
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(labels)  # self.targets 属性供 split.py 使用

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)


def load_iot23_data(root_dir):
    file_path = os.path.join(root_dir, 'iot23.csv')
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return None, None

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # ================= 修改开始 =================
    # 1. 定义要剔除的“泄露特征”
    # resp_port 是最大的作弊嫌疑人。
    # 如果未来你有 src_ip, dst_ip, timestamp, flow_id 等列，也要在这里剔除。
    drop_columns = ['resp_port', 'label']

    # 2. 检查这些列是否存在，防止报错
    existing_drop_cols = [c for c in drop_columns if c in df.columns]

    # 3. 提取特征 (X) 和 标签 (y)
    # 显式指定：除了 drop_columns 以外的所有列都是特征
    feature_cols = [c for c in df.columns if c not in existing_drop_cols]

    print(f"Selected {len(feature_cols)} features: {feature_cols}")
    print(f"Dropped suspicious columns: {existing_drop_cols}")

    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    # ================= 修改结束 =================

    return X, y

class GoogleSpeechWrapper(Dataset):
    def __init__(self, root, subset, transform=None):
        """
        封装 torchaudio 的 SPEECHCOMMANDS 数据集，使其表现得像 MNIST/CIFAR：
        1. 提供 .targets 属性（用于 split.py）
        2. __getitem__ 返回 (img_tensor, label_int)
        3. 自动处理音频填充和频谱图转换
        """
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset=subset)
        self.transform = transform

        # 定义核心标签集合 (V2 数据集标准 35 个词)
        self.classes = sorted(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
                               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
                               'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
                               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 预计算 targets
        # === 优化开始 ===
        # 原代码遍历 self.dataset[i] 会触发音频加载，导致初始化极其缓慢。
        # 这里改为直接解析文件路径获取标签。
        print(f"Processing {subset} dataset metadata for split generation (Optimized)...")
        self.targets = []
        self._indices = []

        # _walker 存储了所有音频文件的路径
        for i, file_path in enumerate(self.dataset._walker):
            # 获取路径字符串
            path_str = str(file_path)

            # 解析标签：通常结构为 .../dataset_root/label/filename.wav
            # 获取文件所在的父文件夹名即为标签
            label = os.path.basename(os.path.dirname(path_str))

            # 兼容性处理：防止路径解析异常
            if not label:
                parts = path_str.split(os.sep)
                if len(parts) > 1:
                    label = parts[-2]

            # 仅保留我们在 self.classes 中定义的感兴趣的类别
            if label in self.class_to_idx:
                self.targets.append(self.class_to_idx[label])
                self._indices.append(i)
        # === 优化结束 ===

        print(f"Loaded {len(self.targets)} samples efficiently.")

    def __getitem__(self, index):
        # 获取原始数据 (映射回原始数据集的索引)
        original_index = self._indices[index]
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[original_index]

        # 1. 音频长度标准化 (Pad or Truncate to 1 second / 16000 samples)
        # 绝大多数样本是 16000，少部分短于 16000
        if waveform.size(1) < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.size(1)))
        else:
            waveform = waveform[:, :16000]

        # 2. 转换为梅尔频谱图 (Mel Spectrogram)
        # 输出尺寸约: [1, 40, 81] (Channel, n_mels, time_frames)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=40,  # 频域特征数，类似图像的高度
            n_fft=400,
            hop_length=200
        )(waveform)

        # 转换幅度到对数刻度 (Log-Mel)
        mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

        # 3. 应用额外的 Transform (如归一化)
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        target = self.class_to_idx[label]
        return mel_spectrogram, target

    def __len__(self):
        return len(self.targets)


def get_dataset(dir, name):
    if name == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.MNIST(dir, train=False, transform=transform_test)

    elif name == 'fashion-mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.FashionMNIST(dir, train=False, transform=transform_test)

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)

    # --- Google Speech 处理逻辑 ---
    elif name == 'googlespeech':
        # 因为在 Dataset Wrapper 里已经做了 log 转换，这里简单归一化即可
        # 这里的 mean 和 std 是基于 Log-Mel 频谱图的大致估算值，用于加速收敛
        transform_common = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = GoogleSpeechWrapper(dir, subset='training', transform=transform_common)
        # Google Speech 有 validation 和 testing 两个集，这里使用 testing 作为评估集
        eval_dataset = GoogleSpeechWrapper(dir, subset='testing', transform=transform_common)

    elif name == 'imagenet':
        # ImageNet 标准均值和方差
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # 假设目录结构为:
        # dir/train/n01440764/xxx.JPEG
        # dir/val/n01440764/xxx.JPEG
        train_dir = os.path.join(dir, 'train')
        val_dir = os.path.join(dir, 'val')

        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
        eval_dataset = datasets.ImageFolder(val_dir, transform=transform_test)

    elif name == 'iot23':

        # 1. 加载清洗后的数据 (去除了端口)
        X, y = load_iot23_data(dir)
        if X is None:
            raise FileNotFoundError("Could not load iot23.csv")
        # 2. 划分训练集和测试集
        # 注意：如果你的数据是按时间顺序排列的，最好不要 shuffle=True，而是直接截断。
        # 这里为了简单通用，还是保留 stratify，但要注意这可能会切碎攻击流。
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # 3. 归一化 (非常重要，尤其是剔除了端口这种大数值特征后)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        train_dataset = IoT23Dataset(X_train, y_train)
        eval_dataset = IoT23Dataset(X_test, y_test)

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # --- 新增：打印训练集与测试集的详细分布信息 ---
    def print_class_distribution(dataset, title):
        print(f"\n[{title}] Distribution Info:")
        if hasattr(dataset, 'targets'):
            # 处理 Tensor 或 list 类型的 targets
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            else:
                targets = np.array(targets)

            total_count = len(targets)
            unique, counts = np.unique(targets, return_counts=True)

            for u, c in zip(unique, counts):
                print(f"  Class {u}: {c:<5} ({c/total_count:.2%})")
            print(f"  Total: {total_count}")
        else:
            print("  (Cannot retrieve .targets for distribution)")

    print_class_distribution(train_dataset, "Train Dataset")
    print_class_distribution(eval_dataset, "Test Dataset")
    # -----------------------------------------------

    return train_dataset, eval_dataset

