import torch
import copy
import logging

logger = logging.getLogger(__name__)


def fed_avg(global_model, client_weights):
    """
    使用加权平均聚合客户端模型更新
    """
    if not client_weights:
        print("警告：没有客户端权重可用于聚合")
        return global_model.state_dict()

    # 确定服务器设备
    server_device = next(global_model.parameters()).device
    print(f"模型聚合在设备 {server_device} 上进行")

    # 获取全局模型参数的副本
    global_state = copy.deepcopy(global_model.state_dict())

    # 对每个参数层单独处理
    for key in global_state.keys():
        param_type = global_state[key].dtype

        if param_type in [torch.long, torch.int64, torch.int32]:
            continue

        # 计算该层的有效客户端权重总和
        layer_total_weight = 0
        for client_id, weights in client_weights.items():
            if key in weights['state_dict']:
                layer_total_weight += weights['num_samples']

        if layer_total_weight == 0:
            # 该层完全丢失，保持全局模型原值
            continue

        # 重置该层为零，然后累加有效客户端的贡献
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32, device=server_device)

        for client_id, weights in client_weights.items():
            if key in weights['state_dict']:
                client_state = weights['state_dict']
                weight = weights['num_samples']
                update = client_state[key].to(device=server_device, dtype=torch.float32)
                # 关键：用该层的实际权重总和作分母
                global_state[key] += (weight / layer_total_weight) * update

    print("模型聚合完成")
    return global_state