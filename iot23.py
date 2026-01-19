import os
import glob
import pandas as pd
import numpy as np
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# === 路径配置 ===
DATASET_ROOT = './datasets/opt'
OUTPUT_CSV = './datasets/iot23.csv'

# === 采样配额 ===
# 建议保持 20万，足够训练且速度快
MAX_SAMPLES_PER_CLASS = 50000

# === 映射表配置 ===

# 1. 协议映射
PROTO_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2, 'ipv6-icmp': 3, 'igmp': 4, 'arp': 5}

# 2. 连接状态映射
STATE_MAP = {
    'S0': 0, 'S1': 1, 'SF': 2, 'REJ': 3, 'S2': 4, 'S3': 5,
    'RSTO': 6, 'RSTR': 7, 'RSTOS0': 8, 'RSTRH': 9,
    'SH': 10, 'SHR': 11, 'OTH': 12
}

# 3. [新增] 常见应用层服务映射
# 即使是加密流量，Zeek通常也能识别出是SSL或SSH
SERVICE_MAP = {
    '-': 0, 'http': 1, 'dns': 2, 'ssh': 3, 'ssl': 4,
    'dhcp': 5, 'irc': 6, 'ftp': 7, 'pop3': 8
}

# 4. [新增] History 字符集 (TCP 标志位统计)
# S=Syn, h=Syn+Ack, A=Ack, D=Data, F=Fin, R=Rst
HISTORY_CHARS = ['S', 'h', 'A', 'D', 'F', 'R']

# === 特征列定义 ===
# 基础数值 + 类别 + 新增的端口 + 历史统计 + 衍生特征
FEATURE_COLS = [
                   'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                   'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
                   'proto', 'conn_state', 'service', 'resp_port',  # 基础特征
                   'avg_orig_ip_bytes', 'avg_resp_ip_bytes'  # 衍生特征
               ] + [f'hist_{c}' for c in HISTORY_CHARS]  # History 统计特征

# Zeek 原始列名 (用于读取对齐)
COL_NAMES_23 = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
    'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
    'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
    'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
    'tunnel_parents', 'label', 'detailed-label'
]

# 消除 Pandas 警告
pd.set_option('future.no_silent_downcasting', True)


def get_label_id(label_str):
    """
    鲁棒的标签匹配逻辑
    """
    s = str(label_str).lower()
    if 'okiru' in s: return 4
    if 'ddos' in s: return 1
    if 'portscan' in s: return 2
    if 'c&c' in s or 'botnet' in s or 'heartbeat' in s: return 3
    if 'attack' in s or 'malware' in s or 'virus' in s: return 4
    if 'benign' in s: return 0
    if 'malicious' in s: return 4
    return 0


def clean_and_convert(df):
    if df.empty: return pd.DataFrame()

    # === 1. 标签提取 (优化版) ===
    # 取最后3列，使用向量化拼接 (比 apply 快)
    last_cols = df.iloc[:, -3:].astype(str)
    combined = last_cols.iloc[:, 0].str.cat([last_cols.iloc[:, 1], last_cols.iloc[:, 2]], sep=' ')
    df['label'] = combined.apply(get_label_id)

    # === 2. 基础清洗 ===
    # 将 '-' 和 '(empty)' 替换为 0 (对于 duration, 填 0 意味着极短或未记录，这是常用做法)
    df = df.replace({'-': 0, '(empty)': 0}).infer_objects(copy=False)

    # 强制转换基础数值列
    num_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    for c in num_cols:
        # 使用 pd.to_numeric 处理可能混入的非数字字符
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)

    # === 3. 特征工程 (补全遗漏) ===

    # [新增] Service 映射 (不在字典里的归为 9-Other)
    df['service'] = df['service'].astype(str).str.lower().map(lambda x: SERVICE_MAP.get(x, 9))

    # [原有] Proto & State
    df['proto'] = df['proto'].astype(str).str.lower().map(lambda x: PROTO_MAP.get(x, 6))
    df['conn_state'] = df['conn_state'].astype(str).map(lambda x: STATE_MAP.get(x, 13))

    # [新增] 目的端口 (关键特征)
    # id.resp_p 可能是 '-' 或数字，强制转 float 再转 int (为了安全)
    df['resp_port'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0)

    # [新增] History 字符串统计 (向量化计算)
    df['history'] = df['history'].astype(str)
    for char in HISTORY_CHARS:
        # 计算 S, h, A... 在 history 字符串中出现的次数
        # 比如 "ShAdDaF" -> S:1, h:1, A:1, D:2 ...
        df[f'hist_{char}'] = df['history'].str.count(char)

    # [新增] 平均包大小 (防止除以0，加一个极小值)
    df['avg_orig_ip_bytes'] = df['orig_ip_bytes'] / (df['orig_pkts'] + 1e-5)
    df['avg_resp_ip_bytes'] = df['resp_ip_bytes'] / (df['resp_pkts'] + 1e-5)

    # === 4. 最终输出整理 ===
    target_cols = FEATURE_COLS + ['label']

    # 确保所有列都存在 (防止某些罕见情况列丢失)
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[target_cols]

    # 统一类型: 特征用 float32 (省内存且兼容性好), 标签用 int8
    df = df.astype(np.float32)
    df['label'] = df['label'].astype(np.int8)

    return df


def process_file_with_quota(args):
    file_path, shared_counts, lock = args
    CHUNK_SIZE = 1_000_000
    final_dfs = []

    try:
        # 使用 names=COL_NAMES_23 强制对齐列名
        reader = pd.read_csv(
            file_path, sep='\t', comment='#', names=COL_NAMES_23,
            chunksize=CHUNK_SIZE,
            low_memory=False, quoting=csv.QUOTE_NONE, on_bad_lines='skip'
        )

        for chunk_df in reader:
            cleaned_chunk = clean_and_convert(chunk_df)
            if cleaned_chunk.empty: continue

            # 分组筛选
            groups = cleaned_chunk.groupby('label')

            with lock:
                for label, group_df in groups:
                    label = int(label)
                    current_count = shared_counts.get(label, 0)

                    if current_count >= MAX_SAMPLES_PER_CLASS:
                        continue

                    needed = MAX_SAMPLES_PER_CLASS - current_count
                    to_take = group_df.iloc[:needed]

                    shared_counts[label] = current_count + len(to_take)
                    final_dfs.append(to_take)

        if not final_dfs: return None
        return pd.concat(final_dfs)

    except Exception as e:
        # 可以在这里 print(e) 调试，但在多进程中可能会乱序
        return None


def main():
    if os.path.exists(OUTPUT_CSV): os.remove(OUTPUT_CSV)

    # 递归查找 conn.log.labeled 文件
    files = glob.glob(os.path.join(DATASET_ROOT, '**', 'conn.log.labeled'), recursive=True)
    if not files:
        print(f"❌ 未在 {DATASET_ROOT} 找到 conn.log.labeled 文件")
        return

    print(f"🚀 启动 IoT-23 数据处理 (Full Features Version)")
    print(f"📋 特征数量: {len(FEATURE_COLS)} (含 History 统计 & Service)")
    print(f"🎯 每类配额: {MAX_SAMPLES_PER_CLASS}")

    # 写入 CSV 头部
    pd.DataFrame(columns=FEATURE_COLS + ['label']).to_csv(OUTPUT_CSV, index=False)

    manager = multiprocessing.Manager()
    shared_counts = manager.dict({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    lock = manager.Lock()

    tasks = [(f, shared_counts, lock) for f in files]

    # 根据你的内存大小调整 max_workers (推荐 4-8)
    SAFE_WORKERS = 4

    with ProcessPoolExecutor(max_workers=SAFE_WORKERS) as executor:
        futures = {executor.submit(process_file_with_quota, task): task for task in tasks}

        pbar = tqdm(as_completed(futures), total=len(files), desc="Processing Files")

        for future in pbar:
            try:
                result_df = future.result()
                current_stats = dict(shared_counts)

                # 动态更新进度条信息
                pbar.set_postfix(
                    Benign=f"{current_stats[0] // 1000}k",
                    DDoS=f"{current_stats[1] // 1000}k",
                    PortScan=f"{current_stats[2] // 1000}k",
                    C_C=f"{current_stats[3]}",
                    Malware=f"{current_stats[4] // 1000}k"
                )

                if result_df is not None and not result_df.empty:
                    # 追加写入 CSV (不写头部)
                    result_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                    del result_df

            except Exception as e:
                pass

    print("\n" + "=" * 50)
    print(f"✅ 处理完成！文件已保存至: {OUTPUT_CSV}")
    print(f"📊 最终统计: {dict(shared_counts)}")
    print("=" * 50)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()