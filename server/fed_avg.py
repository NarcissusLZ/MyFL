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

    # 计算所有客户端的样本总数
    total_weight = sum(weights['num_samples'] for client_id, weights in client_weights.items())

    print(f"参与聚合的客户端数量：{len(client_weights)}")
    print(f"样本总数：{total_weight}")

    # 对每个参数，累加所有客户端的加权贡献
    for key in global_state.keys():
        # 检查参数类型
        param_type = global_state[key].dtype

        # 针对不同数据类型的处理
        if param_type == torch.long or param_type == torch.int64 or param_type == torch.int32:
            # 对于整数类型的参数，直接使用原始值，跳过聚合
            continue
        else:
            # 对于浮点类型的参数，重置并累加
            # 创建与原始张量形状相同但类型为 float32 的零张量
            global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32, device=server_device)

            # 累加每个客户端的加权贡献
            for client_id, weights in client_weights.items():
                client_state = weights['state_dict']
                weight = weights['num_samples']

                if key in client_state:
                    # 确保张量在服务器设备上，并转换为浮点类型
                    update = client_state[key].to(device=server_device, dtype=torch.float32)
                    global_state[key] += (weight / total_weight) * update

    print("模型聚合完成")
    return global_state