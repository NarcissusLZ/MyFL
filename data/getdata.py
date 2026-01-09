import torch
import torch.nn as nn
import torchaudio
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

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
        # 如果你想只做 12 分类 (10 command + silence + unknown)，需要在这里做额外的过滤逻辑
        # 这里为了通用性，我们使用数据集中出现的所有标签
        self.classes = sorted(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
                               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
                               'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
                               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 预计算 targets (耗时操作，但在初始化时必须做一次，以便 split.py 使用)
        # 注意：这里我们遍历一次数据集来构建 targets 列表
        print(f"Processing {subset} dataset metadata for split generation...")
        self.targets = []
        self._indices = []

        # 这种遍历方式确保我们过滤掉不在 classes 列表中的异常数据（如果有的话）
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            label = item[2]  # label is at index 2
            if label in self.class_to_idx:
                self.targets.append(self.class_to_idx[label])
                self._indices.append(i)

        print(f"Loaded {len(self.targets)} samples.")

    def __getitem__(self, index):
        # 获取原始数据
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

    # --- 新增：Google Speech 处理逻辑 ---
    elif name == 'googlespeech':
        # 这里可以是针对频谱图的归一化，或者留空
        # 因为我们在 Dataset Wrapper 里已经做了 log 转换，这里简单归一化即可
        # 这里的 mean 和 std 是基于 Log-Mel 频谱图的大致估算值
        transform_common = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)) # 简单的标准化
        ])

        train_dataset = GoogleSpeechWrapper(dir, subset='training', transform=transform_common)
        # Google Speech 有 validation 和 testing 两个集，通常为了简单这里只取 testing 作 eval
        eval_dataset = GoogleSpeechWrapper(dir, subset='testing', transform=transform_common)

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return train_dataset, eval_dataset