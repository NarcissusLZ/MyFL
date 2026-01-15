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


def load_iot23_data(root_dir):
    file_path = os.path.join(root_dir, 'iot23.csv')

    # === 配置采样策略 ===
    # 你的数据分布已更新：
    # 2 (PortScan): 213M (多数)
    # 4 (Malware):  61M  (多数) -> 必须限制！
    # 0 (Benign):   30M  (多数)
    # 1 (DDoS):     19M  (多数)
    # 3 (C&C):      56k  (少数) -> 全部保留

    # 建议：每类最多 20万，这样总数据量约 85万 (4*20w + 5.6w)
    # 既能保证训练速度，又能保证有足够的特征，且相对平衡。
    # 如果服务器性能强，可以设为 500000 (总数约 205万)
    LIMIT_PER_CLASS = 500000

    if os.path.exists(file_path):
        print(f"Loading IoT-23 data from {file_path} with Balanced Sampling...")
        print(f"  Strategy: Max {LIMIT_PER_CLASS} samples per class.")

        # 1. 初始化容器
        dfs = []
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        total_kept = 0

        try:
            # 2. 打开 CSV 流 (Stream)
            reader = pv.open_csv(
                file_path,
                read_options=pv.ReadOptions(use_threads=True, block_size=10 * 1024 * 1024)
            )

            # 3. 分块读取并过滤
            for chunk in reader:
                df_chunk = chunk.to_pandas()

                if 'label' not in df_chunk.columns:
                    df_chunk.rename(columns={df_chunk.columns[-1]: 'label'}, inplace=True)

                filtered_rows = []

                # 按类别分组筛选
                for cls_id in [0, 1, 2, 3, 4]:
                    # 如果该类已经攒够了，就跳过
                    if class_counts[cls_id] >= LIMIT_PER_CLASS:
                        continue

                    # 选出当前 chunk 里该类的样本
                    df_cls = df_chunk[df_chunk['label'] == cls_id]

                    if not df_cls.empty:
                        needed = LIMIT_PER_CLASS - class_counts[cls_id]
                        if len(df_cls) > needed:
                            df_cls = df_cls.iloc[:needed]

                        filtered_rows.append(df_cls)
                        class_counts[cls_id] += len(df_cls)

                if filtered_rows:
                    dfs.append(pd.concat(filtered_rows))

                # === 优化逻辑修改 ===
                # 只有当所有的“多数类” (0, 1, 2, 4) 都满了，我们才只关注 3
                # 但因为 3 (C&C) 极其稀有且分布在整个文件中，
                # 我们实际上还是得读完整个文件来寻找 Class 3。
                # 这里的判断主要用于控制台打印，告知用户哪些类已经满了。
                if class_counts[0] >= LIMIT_PER_CLASS and \
                   class_counts[1] >= LIMIT_PER_CLASS and \
                   class_counts[2] >= LIMIT_PER_CLASS and \
                   class_counts[4] >= LIMIT_PER_CLASS:
                    # 此时 0,1,2,4 都满了，脚本正在全力搜索 Class 3
                    pass

                # 打印进度
                current_total = sum(class_counts.values())
                if current_total - total_kept > 50000: # 每收集5万条打印一次
                    print(f"  Collecting... {class_counts}")
                    total_kept = current_total

            # 4. 合并所有数据
            if not dfs:
                print("❌ No data found!")
                return None, None

            full_df = pd.concat(dfs)
            print(f"✅ Sampling Done! Final Stats: {class_counts}")
            print(f"  Total samples: {len(full_df)}")

            # 5. 提取与归一化
            X = full_df.iloc[:, :-1].values.astype(np.float32)
            y = full_df['label'].values.astype(np.int64)

            del full_df, dfs

            print("  Normalizing features...")
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            return X, y

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    else:
        # Fallback 模拟数据
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