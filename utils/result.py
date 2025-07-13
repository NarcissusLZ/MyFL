import os
import torch
import logging
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 或者直接设置字体
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

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

    result_plc(history, result_dir, timestamp,config)

def result_plc(history, result_dir, timestamp, config):
    """Generate training process chart with English labels"""
    logger.info("\nGenerating training process chart...")

    # Extract key parameters from config
    model_name = config.get('model_name', 'Unspecified')
    dataset_name = config.get('dataset_name', 'Unspecified')
    packet_loss_rate = config.get('packet_loss_rate', 0.0)
    num_clients = config.get('num_clients', 0)
    client_fraction = config.get('client_fraction', 0.0)
    local_epochs = config.get('local_epochs', 0)

    # Create figure
    fig = plt.figure(figsize=(14, 8))

    # Set title text
    param_text = (
        f"Model: {model_name} | Dataset: {dataset_name} | Packet Loss Rate: {packet_loss_rate:.2f} | "
        f"Total Clients: {num_clients} | Fraction: {client_fraction:.2f} | Local Epochs: {local_epochs}"
    )
    fig.suptitle(param_text, fontsize=14, y=0.98)
    plt.subplots_adjust(top=0.85)

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history['round'], history['accuracy'], 'b-', marker='o', linewidth=2, markersize=4)
    plt.title('Training Accuracy', fontsize=14)
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show final accuracy
    final_accuracy = history['accuracy'][-1]
    plt.annotate(f'Final: {final_accuracy:.2f}%',
                 xy=(history['round'][-1], final_accuracy),
                 xytext=(-100, -30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'),
                 fontsize=12)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['round'], history['loss'], 'r-', marker='s', linewidth=2, markersize=4)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show final loss
    final_loss = history['loss'][-1]
    plt.annotate(f'Final: {final_loss:.4f}',
                 xy=(history['round'][-1], final_loss),
                 xytext=(-80, 30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'),
                 fontsize=12)

    # Add timestamp watermark
    fig.text(0.95, 0.01, f'Generated: {timestamp}', fontsize=8,
             ha='right', va='bottom', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save chart
    chart_path = os.path.join(result_dir, f"training_chart_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved to: {chart_path}")

    # Show final performance summary
    logger.info(f"\nTraining completed - Final accuracy: {final_accuracy:.2f}%, Final loss: {final_loss:.4f}")

    plt.show()
