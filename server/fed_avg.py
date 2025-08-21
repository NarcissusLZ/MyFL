import torch
import copy
import logging

logger = logging.getLogger(__name__)


def fed_avg(global_model, client_weights, device=None):
    """
    执行支持部分层丢包的FedAvg聚合算法

    参数:
        global_model: 全局模型
        client_weights: 客户端权重字典
        device: 运行设备

    返回:
        更新后的全局模型状态字典
    """
    logger.info("开始模型聚合...")

    if not client_weights:
        logger.warning("警告: 没有客户端更新数据，跳过聚合")
        return global_model.state_dict()

    # 初始化全局模型参数
    global_state = copy.deepcopy(global_model.state_dict())

    # 遍历模型的每一层参数
    for key in global_state:
        # 收集所有客户端该层的更新
        layer_updates = []
        layer_weights = []

        # 遍历所有客户端
        for client_id, client_data in client_weights.items():
            # 检查该客户端是否有这一层的更新
            if key in client_data['state_dict']:
                layer_updates.append(client_data['state_dict'][key])
                layer_weights.append(client_data['num_samples'])

        # 如果没有客户端提供该层的更新，则保持原样
        if not layer_updates:
            logger.info(f"层 {key} 没有收到任何客户端的更新，保持原值")
            continue

        # 计算该层的加权平均
        total_weight = sum(layer_weights)

        # 跳过不需要聚合的参数
        if 'num_batches_tracked' not in key:
            # 重置该层参数
            global_state[key] = torch.zeros_like(global_state[key])

            # 加权平均
            for update, weight in zip(layer_updates, layer_weights):
                # 确保设备一致
                if device and update.device != device:
                    update = update.to(device)

                # 加权累加
                global_state[key] += (weight / total_weight) * update
        else:
            # 对于num_batches_tracked，使用第一个客户端的值
            global_state[key] = layer_updates[0].clone()

    logger.info("模型聚合完成")
    return global_state