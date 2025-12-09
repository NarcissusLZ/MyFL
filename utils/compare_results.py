import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

# =================================================================
# 1. 配置文件路径
# =================================================================
# 请确保以下路径指向您保存实验结果的目录
# 脚本会搜索 RESULTS_ROOT_DIR 下所有以 'training_results_' 开头的文件夹。
RESULTS_ROOT_DIR = 'results'

# =================================================================
# 2. 数据加载函数
# =================================================================

def generate_label(path_suffix, index):
    """
    根据路径后缀生成一个可读的图例标签。
    新的逻辑：直接使用文件夹名称的后缀作为标签。
    例如：'training_results_tcp' 会生成标签 'TCP'。
    """
    # 直接返回路径后缀，并转换为大写以美化图例
    return path_suffix.upper()


def load_all_experiment_data():
    """自动扫描并加载所有符合模式的实验数据"""
    all_datasets = []
    print("--- Automatically Scanning and Loading Experiment Data ---")

    # 使用 glob 查找所有符合模式的文件夹
    # 搜索 'results/training_results_*'
    search_pattern = os.path.join(RESULTS_ROOT_DIR, "training_results_*")
    experiment_dirs = sorted(glob.glob(search_pattern))

    if not experiment_dirs:
        print(f"No 'training_results_*' directories found in '{RESULTS_ROOT_DIR}'.")
        return all_datasets

    for index, dir_path in enumerate(experiment_dirs):
        # 提取路径后缀作为标识符
        path_suffix = os.path.basename(dir_path).replace("training_results_", "")
        full_json_path = os.path.join(dir_path, "training_info.json")
        label = generate_label(path_suffix, index)

        if not os.path.exists(full_json_path):
            print(f"Warning: JSON file not found at {full_json_path}. Skipping '{label}'.")
            continue

        try:
            with open(full_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取绘图所需的核心历史数据
            dataset = {
                'label': label,
                'rounds': data.get('rounds', []),
                'accuracy': data.get('accuracy_history', []),
                'loss': data.get('loss_history', []),
                'transmission_time': data.get('transmission_time_history', []),
                'transport': data.get('transport', 'N/A'),
                'model': data.get('model', 'N/A'),
                'dataset_name': data.get('dataset', 'N/A'),
                # 提取用于通信量柱状图的总量
                'total_up_communication_mb': data.get('total_up_communication_mb', 0.0),
                'total_robust_communication_mb': data.get('total_robust_communication_mb', 0.0),
                'total_critical_communication_mb': data.get('total_critical_communication_mb', 0.0),
                'total_robust_transmissions': data.get('total_robust_transmissions', 0),
                'total_critical_transmissions': data.get('total_critical_transmissions', 0)
            }
            all_datasets.append(dataset)
            print(f"Successfully loaded data for: {label} (from {path_suffix})")

        except Exception as e:
            print(f"Error loading or parsing {full_json_path}: {e}")

    return all_datasets


# =================================================================
# 3. 绘图函数 (与原脚本保持一致)
# =================================================================
# 保持 plot_comparison 和 plot_communication_summary 函数不变

def plot_comparison(datasets, metric, ylabel, title_suffix, filename, loc='best'):
    """绘制对比图 (适用于随轮次变化的指标)"""
    if not datasets:
        return

    plt.figure(figsize=(10, 6))

    # 获取模型和数据集名称（假设所有实验相同或取第一个）
    model = datasets[0].get('model', '')
    dataset_name = datasets[0].get('dataset_name', '')

    for data in datasets:
        # 确保数据长度匹配
        x = data['rounds']
        y = data[metric]
        if len(x) != len(y):
            print(f"Warning: Rounds and {metric} lengths mismatch for {data['label']}.")
            continue

        plt.plot(x, y, label=f"{data['label']}", linewidth=2, marker='.', markersize=6)

    plt.title(f'{title_suffix} Comparison ({model} on {dataset_name})', fontsize=16)
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc=loc, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_ROOT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated comparison chart: {filename}")


def plot_communication_summary(datasets, filename):
    """绘制通信量柱状图"""
    if not datasets:
        return

    plt.figure(figsize=(12, 7))
    num_experiments = len(datasets)

    # 柱状图分组设置
    bar_width = 0.25
    x = np.arange(3)  # 鲁棒层, 关键层, 总上行

    # 获取模型和数据集名称
    model = datasets[0].get('model', '')
    dataset_name = datasets[0].get('dataset_name', '')

    for i, data in enumerate(datasets):
        offset = i * bar_width - (num_experiments - 1) * bar_width / 2

        robust = data['total_robust_communication_mb']
        critical = data['total_critical_communication_mb']
        total = data['total_up_communication_mb']

        values = [robust, critical, total]
        counts = [data['total_robust_transmissions'], data['total_critical_transmissions'], 0]  # 传输次数

        bars = plt.bar(x + offset, values, bar_width, label=data['label'], alpha=0.8)

        # 添加值和传输次数标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.01,
                     f'{height:.2f}MB',
                     ha='center', va='bottom', fontsize=9, rotation=45)

            if j < 2 and counts[j] > 0:  # 仅对鲁棒层和关键层显示传输次数
                plt.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.12,
                         f'({counts[j]} trans)',
                         ha='center', va='bottom', fontsize=8, color='darkred', rotation=45)

    plt.xticks(x, ['Robust Layers', 'Critical Layers', 'Total Uplink'], fontsize=12)
    plt.title(f'Total Communication Comparison ({model} on {dataset_name})', fontsize=16)
    plt.ylabel('Communication (MB)', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_ROOT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated comparison chart: {filename}")


# =================================================================
# 4. 主执行逻辑
# =================================================================

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs(RESULTS_ROOT_DIR, exist_ok=True)

    # 1. 自动加载数据
    data_to_plot = load_all_experiment_data()

    if not data_to_plot:
        print(
            "\n*** ERROR: No valid data files loaded. Please ensure 'training_results_*' folders and 'training_info.json' exist in the 'results' directory. ***")
    else:
        # 2. 绘图
        print("\n--- Generating Comparison Charts ---")

        # 准确率对比
        plot_comparison(
            data_to_plot,
            metric='accuracy',
            ylabel='Accuracy (%)',
            title_suffix='Accuracy',
            filename='comparison_accuracy.png',
            loc='lower right'
        )

        # 损失对比
        plot_comparison(
            data_to_plot,
            metric='loss',
            ylabel='Loss',
            title_suffix='Loss',
            filename='comparison_loss.png',
            loc='upper right'
        )

        # 传输时间对比
        plot_comparison(
            data_to_plot,
            metric='transmission_time',
            ylabel='Transmission Time (s)',
            title_suffix='Transmission Time per Round',
            filename='comparison_transmission_time.png',
            loc='upper right'
        )

        # 通信量总结柱状图对比
        plot_communication_summary(
            data_to_plot,
            filename='comparison_communication_summary.png'
        )

    print("\nComparison script finished.")