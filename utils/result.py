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

    result_plc(history, result_dir, timestamp)

def result_plc(history, result_dir, timestamp):
    logger.info("\n生成训练过程图表...")

    plt.figure(figsize=(12, 5))

    # 准确率图表
    plt.subplot(1, 2, 1)
    plt.plot(history['round'], history['accuracy'], 'b-', marker='o', linewidth=2, markersize=4)
    plt.title('Training Accuracy', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)

    # 损失图表
    plt.subplot(1, 2, 2)
    plt.plot(history['round'], history['loss'], 'r-', marker='s', linewidth=2, markersize=4)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.tight_layout()

    # 保存图表
    chart_path = os.path.join(result_dir, f"training_chart_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"训练图表保存至: {chart_path}")

    # 显示最终性能摘要
    final_accuracy = history['accuracy'][-1]
    final_loss = history['loss'][-1]
    logger.info(f"\n训练完成 - 最终准确率: {final_accuracy:.2f}%, 最终损失: {final_loss:.4f}")

    plt.show()