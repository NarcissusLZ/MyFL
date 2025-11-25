import os
import torch
import logging
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt  # 尽管不绘图，但保留此导入以防np依赖

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def save_results(config, history, server, timestamp):
    """保存训练结果和模型"""
    # 创建结果目录
    result_dir = config.get('result_dir', 'results')
    os.makedirs(result_dir, exist_ok=True)

    # 保存通信统计
    comm_stats = server.get_communication_stats()
    logger.info("\n通信统计:")
    # logger.info(f"总下行通信量: {comm_stats['总下行通信量(MB)']:.2f} MB")
    logger.info(f"总上行通信量: {comm_stats['总上行通信量(MB)']:.2f} MB")
    # logger.info(f"总通信量: {comm_stats['总通信量(MB)']:.2f} MB")

    # 将通信量和传输时间添加到历史记录中
    for i, round_data in enumerate(comm_stats['每轮通信量记录']):
        if i < len(history['round']):
            history['up_communication'] = history.get('up_communication', []) + [round_data['up_communication']]
            history['robust_layer_communication'] = history.get('robust_layer_communication', []) + [
                round_data['robust_layer_communication']]
            history['critical_layer_communication'] = history.get('critical_layer_communication', []) + [
                round_data['critical_layer_communication']]

    # 添加传输时间到历史记录
    history['transmission_time'] = comm_stats['每轮最大传输时间']

    # 打印最终性能
    logger.info("\n训练轮次性能:")
    for i in range(len(history['round'])):
        logger.info(f"轮次 {history['round'][i]}: 准确率={history['accuracy'][i]:.2f}%, 损失={history['loss'][i]:.4f}")

    # 调用修改后的函数，仅保存数据，不绘图
    result_plc(history, result_dir, timestamp, config, comm_stats)


def result_plc(history, result_dir, timestamp, config, comm_stats=None):
    """保存训练信息和历史数据到 JSON 文件"""
    logger.info("\nSaving training data to JSON file...")

    # 从配置中提取关键参数
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
        "final_loss": history['loss'][-1] if history['loss'] else 0.0,

        # 扩展：将所有历史数据列表添加到 JSON 中
        "rounds": history.get('round', []),
        "accuracy_history": history.get('accuracy', []),
        "loss_history": history.get('loss', []),
        "up_communication_history": history.get('up_communication', []),
        "robust_layer_communication_history": history.get('robust_layer_communication', []),
        "critical_layer_communication_history": history.get('critical_layer_communication', []),
        # 确保 numpy 类型可以被 JSON 序列化
        "transmission_time_history": [float(t) for t in history.get('transmission_time', [])]
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
                # 确保 numpy 类型可以被 JSON 序列化
                "total_transmission_time_s": float(np.sum(history['transmission_time'])),
                "avg_transmission_time_s": float(np.mean(history['transmission_time']))
            })

    # 保存JSON文件
    json_path = os.path.join(charts_dir, "training_info.json")
    import json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    logger.info(f"Training data saved to: {json_path}")

    logger.info(f"All training data saved to: {charts_dir}")

    # 显示训练完成信息
    logger.info(f"\nTraining completed - Final accuracy: {training_info['final_accuracy']:.2f}%")
    if 'total_transmission_time_s' in training_info:
        logger.info(f"Total transmission time: {training_info['total_transmission_time_s']:.4f}s")