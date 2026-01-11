import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# 导入自定义模块
from data.getdata import get_dataset
from client.client import Client
from server.server import Server
from data.split import split_dataset_to_clients
from utils.result import *
from server.fed_avg import fed_avg


def load_config(config_path='config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 1. 加载配置
    config = load_config()
    logger.info("=" * 50)
    logger.info("联邦学习配置:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

    # 检测可用的GPU数量
    available_gpus = torch.cuda.device_count()
    logger.info(f"检测到 {available_gpus} 个可用GPU")

    if available_gpus <= 1:
        logger.warning("GPU数量不足，无法为客户端分配独立GPU。将使用默认设备。")

    # 2. 准备数据集
    logger.info("准备数据集...")
    train_dataset, test_dataset = get_dataset(
        dir=config['data_dir'],
        name=config['dataset']
    )
    logger.info(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 3. 划分数据集给多个客户端
    logger.info("划分数据集给客户端...")
    client_data = split_dataset_to_clients(
        train_dataset,
        num_clients=config['num_clients'],
        non_iid=config['non_iid'],
        alpha=config['non_iid_alpha']
    )

    # 4. 初始化服务器
    logger.info("\n初始化服务器...")
    server = Server(config, test_dataset)

    # 5. 初始化客户端
    logger.info("初始化客户端...")
    clients = {}
    for i, (client_id, data_subset) in enumerate(client_data.items()):
        gpu_id = None
        if available_gpus > 1:
            gpu_id = (i % (available_gpus - 1)) + 1

        clients[client_id] = Client(
            id=client_id,
            config=config,
            local_dataset=data_subset,
            gpu_id=gpu_id
        )

        if gpu_id is not None:
            logger.info(f"客户端 {client_id} 分配到GPU {gpu_id}")

    logger.info("\n客户端数据分布:")
    result_dir = config.get('result_dir', 'results')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    charts_dir = os.path.join(result_dir, f"training_results_{timestamp}")
    os.makedirs(charts_dir, exist_ok=True)
    client_info_path = os.path.join(charts_dir, f"client_distribution_{timestamp}.txt")

    with open(client_info_path, 'w', encoding='utf-8') as f:
        f.write("客户端数据分布\n")
        f.write("=" * 50 + "\n\n")

        for client_id, data_subset in client_data.items():
            labels = [data_subset[i][1] for i in range(len(data_subset))]
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_distribution = dict(zip(unique_labels, counts))
            total_samples = len(data_subset)
            label_ratios = {label: count / total_samples for label, count in label_distribution.items()}
            client = clients[client_id]
            info_lines = [
                f"客户端 {client_id}: {total_samples} 个样本",
                f"  距离: {client.distance:.2f}m, 丢包率: {client.packet_loss:.3f}",
                f"  类别分布: {label_distribution}",
                f"  类别比例: {dict((k, f'{v:.2%}') for k, v in label_ratios.items())}"
            ]
            for line in info_lines:
                logger.info(line)
                f.write(line + "\n")
            f.write("\n")

    logger.info(f"\n客户端分布信息已保存至: {client_info_path}")
    logger.info("-" * 30)

    logger.info("服务器向所有客户端下发初始模型...")
    server.broadcast_model(list(clients.values()))
    logger.info(f"已初始化 {len(clients)} 个客户端")

    # 6. 训练循环
    logger.info("\n开始联邦学习训练...")
    history = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'clients_selected': []
    }

    start_time = time.time()

    # === 新增配置项：过量选择比例 ===
    over_selection_factor = 1.2

    for round in range(config['num_rounds']):
        round_start = time.time()
        logger.info("\n" + '=' * 20 + f"训练轮次 {round + 1}/{config['num_rounds']}" + '=' * 20)

        # a. 计算本轮目标客户端数量与过量选择数量
        total_clients = len(clients)
        target_k = int(total_clients * config['client_fraction'])
        select_num = int(target_k * over_selection_factor)

        # 确保选择数量不超过总数
        select_num = min(select_num, total_clients)

        # 使用自定义的比例调用 select_clients
        actual_fraction = select_num / total_clients if total_clients > 0 else 0

        selected_clients = server.select_clients(
            list(clients.values()),
            fraction=actual_fraction
        )
        client_ids = [client.id for client in selected_clients]
        logger.info(f"本轮调度 {len(selected_clients)} 个客户端进行训练 (目标聚合: Top {target_k})")
        logger.info(f"调度详情: {client_ids}")
        history['clients_selected'].append(len(selected_clients))

        # b. 服务器向选中客户端广播全局模型参数
        server.broadcast_model(selected_clients)

        # c. 客户端本地训练 (并行模拟)
        client_updates = {}
        for client in selected_clients:
            model_state, num_samples = client.local_train()
            client_updates[client.id] = {
                'model_state': model_state,
                'num_samples': num_samples,
                'client': client
            }

        # d. 客户端上传模型更新 (模拟传输并收集结果)
        simulation_results = []

        for client_id, update in client_updates.items():
            # 使用新方法 simulat_transmission 获取结果但不立即更新 Global Stats
            result = server.simulate_transmission(
                update['client'],
                update['model_state'],
                update['num_samples']
            )

            if result:
                simulation_results.append(result)
            else:
                logger.warning(f"客户端 {client_id} 传输完全失败 (模型损坏)")

        # === 核心逻辑修改：基于传输时间排序并截取 Top-K ===
        # 1. 按传输时间升序排序
        simulation_results.sort(key=lambda x: x['transmission_time'])

        # 2. 截取前 K 个最快完成的节点
        winners = simulation_results[:target_k]

        if len(winners) < target_k:
            logger.warning(f"注意: 成功回传的节点数 ({len(winners)}) 少于目标数 ({target_k})")

        winner_ids = [res['client_id'] for res in winners]
        times = [res['transmission_time'] for res in winners]
        logger.info(f"聚合 Top-{len(winners)} 节点: {winner_ids}")
        if times:
            logger.info(f"  传输时间范围: {min(times):.4f}s - {max(times):.4f}s")

        # 3. 将胜者的数据注册到服务器（更新流量统计和 client_weights）
        server.update_server_stats(winners)

        # 4. 记录本轮最大传输时间 (由 Top-K 中的最慢者决定)
        server.finalize_round_transmission_time()

        # e. 服务器聚合模型更新
        logger.info("服务器聚合模型更新...")
        updated_state_dict = fed_avg(server.global_model, server.client_weights)
        server.global_model.load_state_dict(updated_state_dict)

        # f. 服务器测试全局模型
        logger.info("测试全局模型性能...")
        test_results = server.test_model()

        # g. 记录结果
        history['round'].append(round + 1)
        history['accuracy'].append(test_results['accuracy'])
        history['loss'].append(test_results['loss'])

        # h. 进入下一轮
        server.next_round(current_loss=test_results['loss'])

        round_time = time.time() - round_start
        logger.info(f"本轮完成，耗时: {round_time:.2f}秒")

    # 7. 训练结束
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 50)
    logger.info(f"联邦学习训练完成!")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"最终准确率: {history['accuracy'][-1]:.2f}%")

    # 8. 保存结果
    save_results(config, history, server, timestamp)

    return history


if __name__ == "__main__":
    history = main()
