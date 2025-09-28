import os
import torch
import logging
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt

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
    # history_path = os.path.join(result_dir, f"history_{timestamp}.npy")
    # np.save(history_path, history)
    # logger.info(f"训练历史保存至: {history_path}")

    # # 保存模型
    # model_path = os.path.join(result_dir, f"global_model_{timestamp}.pth")
    # torch.save(server.global_model.state_dict(), model_path)
    # logger.info(f"全局模型保存至: {model_path}")
    #
    # # 保存配置文件
    # config_path = os.path.join(result_dir, f"config_{timestamp}.yaml")
    # with open(config_path, 'w') as f:
    #     yaml.dump(config, f)
    # logger.info(f"配置文件保存至: {config_path}")

    # 保存通信统计
    comm_stats = server.get_communication_stats()
    logger.info("\n通信统计:")
    # logger.info(f"总下行通信量: {comm_stats['总下行通信量(MB)']:.2f} MB")
    logger.info(f"总上行通信量: {comm_stats['总上行通信量(MB)']:.2f} MB")
    #logger.info(f"总通信量: {comm_stats['总通信量(MB)']:.2f} MB")

    # 将通信量和传输时间添加到历史记录中
    for i, round_data in enumerate(comm_stats['每轮通信量记录']):
        if i < len(history['round']):
            history['up_communication'] = history.get('up_communication', []) + [round_data['up_communication']]
            history['robust_layer_communication'] = history.get('robust_layer_communication', []) + [round_data['robust_layer_communication']]
            history['critical_layer_communication'] = history.get('critical_layer_communication', []) + [round_data['critical_layer_communication']]

    # 添加传输时间到历史记录
    history['transmission_time'] = comm_stats['每轮最大传输时间']

    # 打印最终性能
    logger.info("\n训练轮次性能:")
    for i in range(len(history['round'])):
        logger.info(f"轮次 {history['round'][i]}: 准确率={history['accuracy'][i]:.2f}%, 损失={history['loss'][i]:.4f}")

    result_plc(history, result_dir, timestamp, config, comm_stats)

def result_plc(history, result_dir, timestamp, config, comm_stats=None):
    """Generate training process chart with English labels"""
    logger.info("\nGenerating training process chart...")

    # Extract key parameters from config
    model_name = config.get('model', 'Unspecified')
    dataset_name = config.get('dataset', 'Unspecified')
    Transport = config.get('Transport', 0.0)
    num_clients = config.get('num_clients', 0)
    client_fraction = config.get('client_fraction', 0.0)
    local_epochs = config.get('local_epochs', 0)

    # 获取丢失层信息
    layers_to_drop = config.get('layers_to_drop', [])
    layers_info = ", ".join(layers_to_drop) if layers_to_drop else "None"

    # Create figure
    fig = plt.figure(figsize=(15, 10))

    # 添加丢失层信息
    dropped_layers_text = f"Dropped Layers: {layers_info}"

    # Set title
    param_text = (
        f"Model: {model_name} | Dataset: {dataset_name} | Transport: {Transport} | "
        f"Clients: {num_clients} | Fraction: {client_fraction:.2f} | Local Epochs: {local_epochs}"
    )
    fig.suptitle(param_text, fontsize=14, y=0.98)

    # 在主标题下方添加丢失层信息
    plt.figtext(0.5, 0.92, dropped_layers_text, ha='center', fontsize=12,
                bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})

    plt.subplots_adjust(top=0.85, hspace=0.3)

    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(history['round'], history['accuracy'], 'b-', marker='o', linewidth=2, markersize=4)
    plt.title('Training Accuracy', fontsize=14)
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(history['round'], history['loss'], 'r-', marker='s', linewidth=2, markersize=4)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)


    # Communication layers bar chart
    if comm_stats and '每轮通信量记录' in comm_stats:
        plt.subplot(2, 2, 3)
        
        # 获取总流量数据
        total_robust = comm_stats.get('总鲁棒层通信量(MB)', 0)
        total_critical = comm_stats.get('总关键层通信量(MB)', 0)
        total_overall = comm_stats.get('总上行通信量(MB)', 0)
        
        # 创建柱状图数据
        categories = ['Total\nCommunication', 'Robust\nLayers', 'Critical\nLayers']
        values = [total_overall, total_robust, total_critical]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
        
        bars = plt.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 在柱子上添加数值标注
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.2f}MB',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.title('Total Communication by Layer Type', fontsize=14)
        plt.ylabel('Communication (MB)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 设置Y轴范围，留出空间给标注
        plt.ylim(0, max(values) * 1.15)
        
        # 添加百分比信息
        if total_overall > 0:
            robust_percent = (total_robust / total_overall) * 100
            critical_percent = (total_critical / total_overall) * 100
            plt.text(0.02, 0.98, 
                    f'Robust: {robust_percent:.1f}%\nCritical: {critical_percent:.1f}%',
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                    fontsize=10, verticalalignment='top')

        # 传输时间图
        if 'transmission_time' in history and history['transmission_time']:
            plt.subplot(2, 2, 4)
            plt.plot(history['round'], history['transmission_time'], 'orange', marker='D', linewidth=2, markersize=4)
            plt.title('Transmission Time per Round (s)', fontsize=14)
            plt.xlabel('Rounds', fontsize=12)
            plt.ylabel('Transmission Time (s)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)

            # 添加总传输时间标注
            total_time = np.sum(history['transmission_time'])
            plt.text(0.05, 0.95, f'Total: {total_time:.4f}s',
                     transform=plt.gca().transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                     fontsize=12, verticalalignment='top')

            # # 显示平均传输时间
            # avg_time = np.mean(history['transmission_time'])
            # plt.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_time:.4f}s')
            # plt.legend()



    # Add timestamp watermark
    fig.text(0.95, 0.01, f'Generated: {timestamp}', fontsize=8,
             ha='right', va='bottom', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save chart
    chart_path = os.path.join(result_dir, f"training_chart_{timestamp}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved to: {chart_path}")

    # Show final performance and transmission time summary
    if 'transmission_time' in history:
        total_transmission_time = np.sum(history['transmission_time'])
        avg_transmission_time = np.mean(history['transmission_time'])
        logger.info(f"\nTraining completed - Total transmission time: {total_transmission_time:.4f}s")
        logger.info(f"Average transmission time per round: {avg_transmission_time:.4f}s")


    plt.show()
