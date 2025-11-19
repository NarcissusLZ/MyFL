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
    """Generate separate charts and save training information as JSON"""
    logger.info("\nGenerating separate training charts...")

    # Extract key parameters from config
    model_name = config.get('model', 'Unspecified')
    dataset_name = config.get('dataset', 'Unspecified')
    Transport = config.get('Transport', 'TCP')
    num_clients = config.get('num_clients', 0)
    client_fraction = config.get('client_fraction', 0.0)
    local_epochs = config.get('local_epochs', 0)
    non_iid = config.get('non_iid', False)
    non_iid_alpha = config.get('non_iid_alpha', 1.0)
    layers_to_drop = config.get('layers_to_drop', [])

    # 创建单独的结果文件夹
    charts_dir = os.path.join(result_dir, f"training_results_{timestamp}")
    os.makedirs(charts_dir, exist_ok=True)

    # 准备训练信息字典
    training_info = {
        "timestamp": timestamp,
        "model": model_name,
        "dataset": dataset_name,
        "transport": Transport,
        "num_clients": num_clients,
        "client_fraction": client_fraction,
        "local_epochs": local_epochs,
        "non_iid": non_iid,
        "non_iid_alpha": non_iid_alpha,
        "layers_to_drop": layers_to_drop,
        "batch_size": config.get('batch_size', 0),
        "learning_rate": config.get('lr', 0.0),
        "momentum": config.get('momentum', 0.0),
        "num_rounds": config.get('num_rounds', 0),
        "device": config.get('device', 'cpu'),
        "random_seed": config.get('random_seed', 42),
        "final_accuracy": history['accuracy'][-1] if history['accuracy'] else 0.0,
        "final_loss": history['loss'][-1] if history['loss'] else 0.0
    }

    # 添加通信统计信息
    if comm_stats:
        training_info.update({
            "total_up_communication_mb": comm_stats.get('总上行通信量(MB)', 0.0),
            "total_robust_communication_mb": comm_stats.get('总鲁棒层通信量(MB)', 0.0),
            "total_critical_communication_mb": comm_stats.get('总关键层通信量(MB)', 0.0),
            "total_robust_transmissions": comm_stats.get('总鲁棒层传输次数', 0),
            "total_critical_transmissions": comm_stats.get('总关键层传输次数', 0)
        })

        if 'transmission_time' in history and history['transmission_time']:
            training_info.update({
                "total_transmission_time_s": float(np.sum(history['transmission_time'])),
                "avg_transmission_time_s": float(np.mean(history['transmission_time']))
            })

    # 保存JSON文件
    json_path = os.path.join(charts_dir, "training_info.json")
    import json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    logger.info(f"Training info saved to: {json_path}")

    # 图1: 准确率
    plt.figure(figsize=(10, 6))
    plt.plot(history['round'], history['accuracy'], 'b-', marker='o', linewidth=2, markersize=4)
    plt.title(f'Training Accuracy - {model_name} on {dataset_name}', fontsize=16)
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0.02, 0.98, f'Final: {training_info["final_accuracy"]:.2f}%',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2: 损失
    plt.figure(figsize=(10, 6))
    plt.plot(history['round'], history['loss'], 'r-', marker='s', linewidth=2, markersize=4)
    plt.title(f'Training Loss - {model_name} on {dataset_name}', fontsize=16)
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0.02, 0.98, f'Final: {training_info["final_loss"]:.4f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
             fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3: 通信量统计
    if comm_stats and '每轮通信量记录' in comm_stats:
        plt.figure(figsize=(10, 6))

        total_robust = comm_stats.get('总鲁棒层通信量(MB)', 0)
        total_critical = comm_stats.get('总关键层通信量(MB)', 0)
        total_overall = comm_stats.get('总上行通信量(MB)', 0)
        robust_count = comm_stats.get('总鲁棒层传输次数', 0)
        critical_count = comm_stats.get('总关键层传输次数', 0)

        categories = ['Robust\nLayers', 'Critical\nLayers', 'Total\nCommunication']
        values = [total_robust, total_critical, total_overall]
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

        bars = plt.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.01,
                     f'{value:.2f}MB',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

            if i < 2:
                count = robust_count if i == 0 else critical_count
                plt.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.06,
                         f'({count} trans)',
                         ha='center', va='bottom', fontsize=10, style='italic', color='darkred')

        plt.title(f'Total Communication by Layer Type - {Transport} Transport', fontsize=16)
        plt.ylabel('Communication (MB)', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, max(values) * 1.2)

        if total_overall > 0:
            robust_percent = (total_robust / total_overall) * 100
            critical_percent = (total_critical / total_overall) * 100
            plt.text(0.02, 0.98,
                     f'Robust: {robust_percent:.1f}% ({robust_count} transmissions)\n'
                     f'Critical: {critical_percent:.1f}% ({critical_count} transmissions)',
                     transform=plt.gca().transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                     fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "communication.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 图4: 传输时间
    if 'transmission_time' in history and history['transmission_time']:
        plt.figure(figsize=(10, 6))
        plt.plot(history['round'], history['transmission_time'], 'orange', marker='D', linewidth=2, markersize=4)
        plt.title(f'Transmission Time per Round - {Transport} Transport', fontsize=16)
        plt.xlabel('Rounds', fontsize=14)
        plt.ylabel('Transmission Time (s)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        total_time = np.sum(history['transmission_time'])
        avg_time = np.mean(history['transmission_time'])
        plt.text(0.02, 0.98, f'Total: {total_time:.4f}s\nAverage: {avg_time:.4f}s',
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                 fontsize=12, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "transmission_time.png"), dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"All charts and training info saved to: {charts_dir}")

    # 显示训练完成信息
    logger.info(f"\nTraining completed - Final accuracy: {training_info['final_accuracy']:.2f}%")
    if 'total_transmission_time_s' in training_info:
        logger.info(f"Total transmission time: {training_info['total_transmission_time_s']:.4f}s")

