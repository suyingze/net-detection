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
from model import CustomDataset, NeighborhoodAwareVAE, neighborhood_aware_loss_function
from net_detect import Featurize_With_Neighbors, train_NeighborhoodAware_VAE, get_NeighborhoodAware_MSE
import pickle as pkl


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def train_NeighborhoodAware_VAE_Complete(train_X, neighbor_map, num_epochs=8):
    """完整的邻域感知VAE训练流程"""
    print("开始训练邻域感知VAE...")
    
    data_loader = DataLoader(train_X, batch_size=128, shuffle=True, pin_memory=True)
    model = NeighborhoodAwareVAE()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    loss_list = []
    min_epoch_loss = 999
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (x) in enumerate(data_loader):
            x = x.to(device)
            
            # 获取对应的邻域信息
            batch_neighbors = get_batch_neighbors(neighbor_map, batch_idx, len(x))
            
            x_recon, mu, logvar = model(x, batch_neighbors)
            # 使用改进的损失函数
            loss = neighborhood_aware_loss_function(x_recon, x, mu, logvar, batch_neighbors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_loss = train_loss / len(data_loader.dataset)
        loss_list.append(epoch_loss)
        print(f"Neighborhood-Aware VAE Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")

    # 保存模型
    model_path = './dataset/OpTC/FLASH/NeighborhoodAware_VAE.pkl'
    torch.save(model.state_dict(), model_path)
    print(f'Neighborhood-Aware VAE Training finish. Model saved to {model_path}')
    
    # 绘制训练loss曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_list)+1), loss_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neighborhood-Aware VAE Training Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('neighborhood_vae_loss_curve.png')
    
    return model


def get_batch_neighbors(neighbor_map, batch_idx, batch_size):
    """获取批次的邻域信息"""
    batch_neighbors = {}
    start_idx = batch_idx * batch_size
    
    for i in range(batch_size):
        global_idx = start_idx + i
        if global_idx in neighbor_map:
            # 将全局索引转换为批次内索引
            batch_neighbors[i] = {idx - start_idx for idx in neighbor_map[global_idx] if start_idx <= idx < start_idx + batch_size}
    
    return batch_neighbors


def get_NeighborhoodAware_threshold(model, features, neighbor_map):
    """借鉴ARGUS的邻域感知阈值确定"""
    validate_mse = get_NeighborhoodAware_MSE(model, features, neighbor_map)
    threshold = np.percentile(validate_mse, 99.55)
    print('90: ', np.percentile(validate_mse, 90))
    print('80: ', np.percentile(validate_mse, 80))
    print('95: ', np.percentile(validate_mse, 95))
    print('99: ', np.percentile(validate_mse, 99))
    print('99.5: ', np.percentile(validate_mse, 99.5))
    print('99.55: ', threshold)
    return threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mydata-flow', help='Dataset name')
    args = parser.parse_args()
    
    # 设置路径
    global VAE_PATH, FASTTEXT_PATH, device
    VAE_PATH = f'./dataset/{args.dataset}/FLASH/NeighborhoodAware_VAE.pkl'
    FASTTEXT_PATH = f'./dataset/{args.dataset}/FLASH/fasttext.model'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    
    # 加载数据
    print("Loading data...")
    df = pd.read_csv(f'./dataset/{args.dataset}/processed_data.csv')
    
    # 使用改进的特征提取（包含邻域信息）
    print("Extracting features with neighbors...")
    features, node_map_idx, neighbor_map = Featurize_With_Neighbors(df)
    
    # 训练邻域感知VAE
    print("Training Neighborhood-Aware VAE...")
    model = train_NeighborhoodAware_VAE_Complete(features, neighbor_map)
    
    # 确定阈值
    print("Determining threshold...")
    threshold = get_NeighborhoodAware_threshold(model, features, neighbor_map)
    
    print(f"Final threshold: {threshold}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 