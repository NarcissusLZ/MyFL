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

# === 映射表 ===
LABEL_MAP = {
    'benign': 0, 'ddos': 1, 'portscan': 2, 'c&c': 3,
    'attack': 4, 'malware': 4, 'okiru': 4, 'malicious': 4, 'virus': 4
}
PROTO_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2, 'ipv6-icmp': 3, 'igmp': 4, 'arp': 5}
STATE_MAP = {'S0': 0, 'S1': 1, 'SF': 2, 'REJ': 3, 'S2': 4, 'S3': 5, 'RSTO': 6, 'RSTR': 7, 'RSTOS0': 8, 'RSTRH': 9,
             'SH': 10, 'SHR': 11, 'OTH': 12}

FEATURE_COLS = [
    'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
    'proto', 'conn_state'
]
# 我们依然定义这些列名，主要是为了让前面的特征列对齐
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
    # 优先匹配具体攻击
    if 'okiru' in s: return 4
    if 'ddos' in s: return 1
    if 'portscan' in s: return 2
    if 'c&c' in s or 'botnet' in s or 'heartbeat' in s: return 3
    if 'attack' in s or 'malware' in s or 'virus' in s: return 4

    # 最后匹配良性
    if 'benign' in s: return 0

    # 如果包含 malicious 但没有具体细分，归为 4
    if 'malicious' in s: return 4

    return 0


def clean_and_convert(df):
    if df.empty: return pd.DataFrame()

    # === [关键修复] 标签提取逻辑 ===
    # 不依赖列名，强制取最后 3 列并将它们拼成一个字符串
    # 这样无论数据是 "-   Malicious   Okiru" 还是分开的 tab，都能被捕获

    # 取最后3列（防止越界，如果列数不够就取全部）
    last_cols = df.iloc[:, -3:].astype(str)

    # 将这几列的内容用空格拼起来
    df['combined_search_text'] = last_cols.apply(lambda x: ' '.join(x), axis=1)

    # 应用标签匹配
    df['label'] = df['combined_search_text'].apply(get_label_id)

    # === 下面是常规清洗逻辑 ===
    df = df.replace({'-': 0, '(empty)': 0}).infer_objects(copy=False)

    num_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    for c in num_cols:
        # 如果列名不对齐，可能某些特征会在最后几列，这里做个兜底
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['proto'] = df['proto'].astype(str).map(lambda x: PROTO_MAP.get(x.lower(), 6))
    df['conn_state'] = df['conn_state'].astype(str).map(lambda x: STATE_MAP.get(x, 13))

    # 选取最终特征
    df = df[FEATURE_COLS + ['label']]
    df = df.astype(np.float32)
    df['label'] = df['label'].astype(np.int8)

    return df


def process_file_with_quota(args):
    file_path, shared_counts, lock = args
    CHUNK_SIZE = 1_000_000
    final_dfs = []

    try:
        # 即使列对不齐，我们也按 23 列读，反正我们只关心前几列特征和最后几列标签
        # 使用 names=COL_NAMES_23 会强制 Pandas 扩展列，不够的补 NaN，这对我们很有利
        reader = pd.read_csv(
            file_path, sep='\t', comment='#', names=COL_NAMES_23,
            chunksize=CHUNK_SIZE,
            low_memory=False, quoting=csv.QUOTE_NONE, on_bad_lines='skip'  # 忽略坏行
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

    except Exception:
        return None


def main():
    if os.path.exists(OUTPUT_CSV): os.remove(OUTPUT_CSV)

    files = glob.glob(os.path.join(DATASET_ROOT, '**', 'conn.log.labeled'), recursive=True)
    if not files:
        print("未找到文件")
        return

    print(f"🚀 启动终极修复版 (Smart Label Detection)")
    print(f"🎯 每类配额: {MAX_SAMPLES_PER_CLASS}")

    pd.DataFrame(columns=FEATURE_COLS + ['label']).to_csv(OUTPUT_CSV, index=False)

    manager = multiprocessing.Manager()
    shared_counts = manager.dict({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    lock = manager.Lock()

    tasks = [(f, shared_counts, lock) for f in files]

    # 限制并发，防止内存爆炸
    SAFE_WORKERS = 4

    with ProcessPoolExecutor(max_workers=SAFE_WORKERS) as executor:
        futures = {executor.submit(process_file_with_quota, task): task for task in tasks}

        pbar = tqdm(as_completed(futures), total=len(files), desc="Processing")

        for future in pbar:
            try:
                result_df = future.result()
                current_stats = dict(shared_counts)
                # 动态更新进度条
                pbar.set_postfix(
                    Benign=f"{current_stats[0] // 1000}k",
                    DDoS=f"{current_stats[1] // 1000}k",
                    PortScan=f"{current_stats[2] // 1000}k",
                    C_C=f"{current_stats[3]}",
                    Malware=f"{current_stats[4] // 1000}k"
                )

                if result_df is not None and not result_df.empty:
                    result_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                    del result_df
            except Exception as e:
                pass

    print("\n" + "=" * 50)
    print(f"✅ 处理完成！最终统计: {dict(shared_counts)}")
    print("=" * 50)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()