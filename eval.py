import json
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import FastText
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import argparse
from model import CustomDataset, VAE, loss_function
import pickle as pkl
from datetime import datetime


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def get_MSE(model, features):
    # 检查输入类型并优化转换
    if isinstance(features, list) and all(isinstance(arr, np.ndarray) for arr in features):
        features = np.stack(features)  # 合并多个NumPy数组
    
    x = torch.from_numpy(np.asarray(features)).float().to(device)  # 统一转换 n
    
    with torch.no_grad():
        x_recon, mu, logvar = model(x)
        mse_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=1).cpu().numpy()
    
    return mse_loss.tolist()


def load_test_data_and_timestamps(dataset_path, test_file):
    """加载测试数据并建立节点ID到时间戳的映射"""
    node_timestamps = {}
    
    with open(f"{dataset_path}{test_file}", 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                src_ip_port = data.get('src_ip_port', '')
                dest_ip_port = data.get('dest_ip_port', '')
                timestamp = data.get('timestamp', '')
                
                if src_ip_port and timestamp:
                    node_timestamps[src_ip_port] = timestamp
                if dest_ip_port and timestamp:
                    node_timestamps[dest_ip_port] = timestamp
            except json.JSONDecodeError:
                continue
    
    return node_timestamps


def load_attack_nodes(attack_file):
    """加载真实攻击节点列表"""
    attack_nodes = set()
    with open(attack_file, 'r', encoding='utf-8') as f:
        for line in f:
            node_id = line.strip()
            if node_id:
                attack_nodes.add(node_id)
    return attack_nodes

def load_attack_connections(test_file):
    """加载攻击连接信息，包括源节点和目标节点"""
    attack_connections = set()
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('type') == 'udp attack':
                    src_node = data.get('src_ip_port', '')
                    dest_node = data.get('dest_ip_port', '')
                    if src_node and dest_node:
                        attack_connections.add((src_node, dest_node))
            except json.JSONDecodeError:
                continue
    return attack_connections


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    args = parser.parse_args()
    dataset = args.dataset
    dataset_path = f'./dataset/{dataset}/'

    dataset_file_map = {
        'optc_day23-flow' : {
            'train': 'train_conn_23_0-1.json',
            'test': 'test_conn_23_15-16.json',
            'ecarbro': 'ecarbro_23red_0201.json'
        },
        'mydata-flow' : {
            'train': 'train_conn_23_0-1.json',
            'test': 'test_conn_23_15-16.json',
            'ecarbro': 'ecarbro_23red_0201.json'
        }
    }

    EACRBRO_FILE = f"{dataset_path}{dataset_file_map[dataset]['ecarbro']}"
    OUTPUT_FILE = f'{dataset_path}net_alarms.txt'
    ATTACK_FILE = f'{dataset_path}net_attack.txt'
    FASTTEXT_PATH = './models/FastText.model'
    VAE_PATH = './models/VAE.model'
    TEST_FILE = dataset_file_map[dataset]['test']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    w2vmodel = FastText.load(FASTTEXT_PATH)
    model = VAE().to(device)
    model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    model.eval()

    threshold =  20.08073849487302

    # 加载测试数据和时间戳映射
    node_timestamps = load_test_data_and_timestamps(dataset_path, TEST_FILE)

    # 加载真实攻击节点
    attack_nodes = load_attack_nodes(ATTACK_FILE)
    
    # 加载攻击连接信息
    attack_connections = load_attack_connections(f"{dataset_path}{TEST_FILE}")
    attack_target_nodes = set()
    for src, dest in attack_connections:
        attack_target_nodes.add(dest)

    # load test data
    with open(f"dataset/{dataset}/test_features.pkl", "rb") as f:
        features = pkl.load(f)
    with open(f"dataset/{dataset}/test_node_map_idx.pkl", "rb") as f:
        node_map_idx = pkl.load(f)
    node_ids = list(node_map_idx)

    # detection
    test_mse = get_MSE(model, features)
    anomalies = []
    
    for id, mse in zip(node_ids, test_mse):
        if mse > threshold:
            # 获取节点对应的时间戳，如果没有则使用默认值
            timestamp = node_timestamps.get(id, "unknown")
            # 判断是否为真实攻击节点
            is_attack = "是" if id in attack_nodes else "否"
            # 判断节点类型
            if id in attack_nodes:
                node_type = "攻击源节点"
            elif id in attack_target_nodes:
                node_type = "攻击目标节点"
            else:
                node_type = "误报节点"
            anomalies.append({
                'node_id': id,
                'timestamp': timestamp,
                'mse': mse,
                'threshold': threshold,
                'is_attack': is_attack,
                'node_type': node_type
            })
    
    print(f"total entities: {len(node_ids)} \nnet anomalies: {len(anomalies)}")

    # 输出详细格式的警报文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
        for anomaly in anomalies:
            file.write(f"{anomaly['node_id']} | {anomaly['timestamp']} | {anomaly['mse']:.4f} | {anomaly['threshold']:.4f} | {anomaly['is_attack']} | {anomaly['node_type']}\n")

    end_time = time.time()
    print(f'Finish eval: {end_time - start_time}')
