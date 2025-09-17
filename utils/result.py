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

    # 添加传输时间到历史记录
    history['transmission_time'] = comm_stats['每轮最大传输时间']

    # 打印最终性能
    logger.info("\n训练轮次性能:")
    for i in range(len(history['round'])):
        logger.info(f"轮次 {history['round'][i]}: 准确率={history['accuracy'][i]:.2f}%, 损失={history['loss'][i]:.4f}")

    result_plc(history, result_dir, timestamp, config)

def result_plc(history, result_dir, timestamp, config):
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

    # Show final accuracy
    final_accuracy = history['accuracy'][-1]
    plt.annotate(f'Final: {final_accuracy:.2f}%',
                 xy=(history['round'][-1], final_accuracy),
                 xytext=(-100, -30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'),
                 fontsize=12)

    # Loss plot
    plt.subplot(2, 2, 2)
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

    # Communication per round plot
    if 'up_communication' in history:
        plt.subplot(2, 2, 3)
        # plt.plot(history['round'], history['down_communication'], 'g-', marker='d', label='Downstream')
        plt.plot(history['round'], history['up_communication'], 'm-', marker='^', label='Upstream')
        plt.title('Communication per Round (MB)', fontsize=14)
        plt.xlabel('Rounds', fontsize=12)
        plt.ylabel('Communication (MB)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加总通信量标注
        total_comm = np.sum(history['up_communication'])
        plt.text(0.05, 0.95, f'Total: {total_comm:.2f} MB',
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                 fontsize=12, verticalalignment='top')

        # 传输时间图 (替换累积通信量图)
        if 'transmission_time' in history and history['transmission_time']:
            plt.subplot(2, 2, 4)
            plt.plot(history['round'], history['transmission_time'], 'orange', marker='D', linewidth=2, markersize=4)
            plt.title('Transmission Time per Round (s)', fontsize=14)
            plt.xlabel('Rounds', fontsize=12)
            plt.ylabel('Transmission Time (s)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)

            # 显示平均传输时间
            avg_time = np.mean(history['transmission_time'])
            plt.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_time:.4f}s')
            plt.legend()



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
