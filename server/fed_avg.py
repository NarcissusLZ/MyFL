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

    # 使用加权平均聚合更新
    for client_id, weights in client_weights.items():
        client_state = weights['state_dict']
        weight = weights['num_samples']

        # 显示每个客户端的贡献权重
        contribution = (weight / total_weight) * 100
        print(f"客户端 {client_id} 贡献：{contribution:.2f}%，样本数：{weight}")

        # 遍历所有参数，进行加权平均
        for key in global_state.keys():
            if key in client_state:
                # 确保张量在同一设备上
                update = client_state[key].to(server_device)
                global_state[key] = global_state[key] + (weight / total_weight) * update

    print("模型聚合完成")
    return global_state