import os
import torch
import torch.nn as nn
import torchaudio
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

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


# --- 在 getdata.py 顶部补充 import ---
import pyarrow.csv as pv
import pyarrow as pa
import gc


# --- 替换原有的 load_iot23_data 函数 ---
def load_iot23_data(root_dir):
    file_path = os.path.join(root_dir, 'iot23.csv')

    if os.path.exists(file_path):
        print(f"Loading IoT-23 data from {file_path}...")
        print("Using PyArrow for multi-threaded high-performance reading...")

        try:
            # 1. 使用 PyArrow 多线程读取
            # read_options: 配置多线程
            # convert_options: 可以在读取时直接指定列类型（可选，这里我们在后续处理）
            table = pv.read_csv(
                file_path,
                read_options=pv.ReadOptions(use_threads=True),  # 启用多线程
                parse_options=pv.ParseOptions(delimiter=',')
            )

            print(f"  - CSV Parsed. Shape: {table.shape}")

            # 2. 转换为 Pandas (开启 self_destruct 以节省内存)
            # split_blocks=True 允许 PyArrow 并行转换
            df = table.to_pandas(self_destruct=True, split_blocks=True)

            # 手动释放 Arrow table 内存
            del table
            gc.collect()

            print("  - Converted to Pandas DataFrame.")

            # 3. 提取特征和标签
            # 假设最后一列是 Label
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # 释放 DataFrame 内存
            del df
            gc.collect()

            # 4. 内存优化与类型转换
            print("  - Optimizing memory types...")
            X = X.astype(np.float32)  # 3.25亿 * 10 * 4 bytes ≈ 13 GB
            y = y.astype(np.int64)  # PyTorch CrossEntropyLoss 需要 LongTensor (int64)

            # 5. 归一化 (StandardScaler)
            # 注意：如果内存不足，这里可能会炸。
            # 3.25亿数据做 fit_transform 需要额外的内存计算均值方差。
            print("  - Normalizing features (StandardScaler)...")
            try:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            except np.core._exceptions.MemoryError:
                print("⚠️ 内存不足，无法进行全局归一化！尝试分块归一化...")
                # 分块归一化逻辑 (Partial Fit)
                scaler = StandardScaler()
                chunk_size = 10000000  # 1000万一批
                for i in range(0, len(X), chunk_size):
                    scaler.partial_fit(X[i:i + chunk_size])

                # Transform 也可以分块做，或者原地修改
                for i in range(0, len(X), chunk_size):
                    X[i:i + chunk_size] = scaler.transform(X[i:i + chunk_size])

            print(f"✅ Data loaded successfully. Shape: {X.shape}")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # 如果 PyArrow 失败，回退到 Pandas 读取一部分用于测试
            print("Fallback: Reading first 1M rows with standard Pandas...")
            df = pd.read_csv(file_path, nrows=1000000)
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.int64)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    else:
        print(f"Warning: {file_path} not found. Generating synthetic data.")
        X = np.random.randn(10000, 10).astype(np.float32)
        y = np.random.randint(0, 5, size=(10000,)).astype(np.int64)

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
        # 加载数据
        X, y = load_iot23_data(dir)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = IoT23Dataset(X_train, y_train)
        eval_dataset = IoT23Dataset(X_test, y_test)

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return train_dataset, eval_dataset