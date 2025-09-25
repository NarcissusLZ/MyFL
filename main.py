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
        # 为客户端分配GPU ID，跳过GPU 0（保留给服务器）
        gpu_id = None
        if available_gpus > 1:
            gpu_id = (i % (available_gpus - 1)) + 1  # 从GPU 1开始循环分配

        clients[client_id] = Client(
            id=client_id,
            config=config,
            local_dataset=data_subset,
            gpu_id=gpu_id
        )

        if gpu_id is not None:
            logger.info(f"客户端 {client_id} 分配到GPU {gpu_id}")

    # 打印每个客户端的数据量和类别分布
    logger.info("\n客户端数据分布:")
    for client_id, data_subset in client_data.items():
        # 获取该客户端的所有标签
        labels = [data_subset[i][1] for i in range(len(data_subset))]

        # 统计类别分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))

        # 计算类别比例
        total_samples = len(data_subset)
        label_ratios = {label: count / total_samples for label, count in label_distribution.items()}

        # 获取对应客户端对象以访问距离和丢包率
        client = clients[client_id]

        logger.info(f"客户端 {client_id}: {total_samples} 个样本")
        logger.info(f"  距离: {client.distance}m, 丢包率: {client.packet_loss:.3f}")
        logger.info(f"  类别分布: {label_distribution}")
        logger.info(f"  类别比例: {dict((k, f'{v:.2%}') for k, v in label_ratios.items())}")

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

    for round in range(config['num_rounds']):
        round_start = time.time()
        logger.info("\n"+'='*20+f"训练轮次 {round + 1}/{config['num_rounds']}"+'='*20)

        # a. 服务器选择客户端
        selected_clients = server.select_clients(
            list(clients.values()),
            fraction=config['client_fraction']
        )
        client_ids = [client.id for client in selected_clients]
        logger.info(f"本轮选中客户端: {client_ids}")
        history['clients_selected'].append(len(selected_clients))

        # b. 服务器向选中客户端广播全局模型参数
        server.broadcast_model(selected_clients)

        # c. 客户端本地训练
        client_updates = {}
        for client in selected_clients:
            model_state, num_samples = client.local_train()
            client_updates[client.id] = {
                'model_state': model_state,
                'num_samples': num_samples,
                'client': client  # 保存客户端对象引用
            }

        # d. 客户端上传模型更新给服务器
        current_accuracy = None
        if round > 0:  # 第一轮没有历史精度
            current_accuracy = history['accuracy'][-1]  # 使用上一轮的精度
        
        for client_id, update in client_updates.items():
            success = server.receive_local_model(
                update['client'],  # 传递客户端对象而非ID
                update['model_state'],
                update['num_samples'],
                current_round=round + 1,  # ✅ 添加当前轮次
                current_accuracy=current_accuracy  # ✅ 添加当前精度
            )
            
            if not success:
                logger.warning(f"客户端 {client_id} 的模型更新接收失败")

        # 记录本轮最大传输时间（在聚合前）
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
        server.next_round()

        round_time = time.time() - round_start
        logger.info(f"本轮完成，耗时: {round_time:.2f}秒")

    # 7. 训练结束
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 50)
    logger.info(f"联邦学习训练完成!")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"最终准确率: {history['accuracy'][-1]:.2f}%")

    # 8. 保存结果
    save_results(config, history, server)

    return history



if __name__ == "__main__":
    history = main()
