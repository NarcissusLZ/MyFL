import torch
import copy
import logging

logger = logging.getLogger(__name__)


def fed_avg(global_model, client_weights):
    """
    使用加权平均聚合客户端模型更新
    所有客户端都应该有完整的参数（丢失的层已用全局模型替代）
    """
    if not client_weights:
        print("警告：没有客户端权重可用于聚合")
        return global_model.state_dict()

    # 确定服务器设备
    server_device = next(global_model.parameters()).device
    print(f"模型聚合在设备 {server_device} 上进行")

    # 获取全局模型参数的副本
    global_state = copy.deepcopy(global_model.state_dict())
    
    # 计算总权重
    total_weight = sum(weights['num_samples'] for weights in client_weights.values())
    
    print(f"开始聚合 {len(client_weights)} 个客户端，总样本数: {total_weight}")

    # 对每个参数层进行聚合
    for key in global_state.keys():
        param_type = global_state[key].dtype

        if param_type in [torch.long, torch.int64, torch.int32]:
            continue

        # 重置该层为零，然后累加所有客户端的贡献
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32, device=server_device)

        for client_id, weights in client_weights.items():
            client_state = weights['state_dict']
            weight = weights['num_samples']
            update = client_state[key].to(device=server_device, dtype=torch.float32)
            # 使用总权重作分母进行标准联邦平均
            global_state[key] += (weight / total_weight) * update

    print("模型聚合完成")
    return global_state