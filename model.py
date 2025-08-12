from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.FloatTensor(sample)
        return sample
    

class VAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=16):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# 借鉴ARGUS的邻域感知方法，改进VAE
class NeighborhoodAwareVAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=16):
        super(NeighborhoodAwareVAE, self).__init__()
        
        # 原有VAE结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        # 借鉴ARGUS的邻域聚合层
        self.neighbor_aggregator = nn.Linear(input_dim, hidden_dim)
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 借鉴ARGUS的权重参数
        self.lambda1 = 0.5  # 边分数权重
        self.lambda2 = 0.5  # 邻域分数权重
    
    def aggregate_neighbors(self, node_features, neighbor_map):
        """借鉴ARGUS的邻域聚合方法"""
        neighbor_features = []
        for node_id, neighbors in neighbor_map.items():
            if neighbors and len(neighbors) > 0:
                # 计算邻域平均特征（借鉴ARGUS的邻域聚合思想）
                neighbor_feats = [node_features[neighbor] for neighbor in neighbors]
                neighbor_avg = torch.mean(torch.stack(neighbor_feats), dim=0)
            else:
                neighbor_avg = torch.zeros_like(node_features[0])
            neighbor_features.append(neighbor_avg)
        return torch.stack(neighbor_features)
    
    def get_neighborhood_score(self, node_features, neighbor_map):
        """借鉴ARGUS的get_src_score方法"""
        neighborhood_scores = []
        
        for node_id, neighbors in neighbor_map.items():
            if neighbors and len(neighbors) > 0:
                # 计算邻域平均分数
                neighbor_scores = [node_features[neighbor] for neighbor in neighbors]
                neighbor_avg_score = torch.mean(torch.stack(neighbor_scores), dim=0)
                neighborhood_scores.append(neighbor_avg_score)
            else:
                neighborhood_scores.append(torch.zeros_like(node_features[0]))
        
        return torch.stack(neighborhood_scores)
    
    def forward(self, x, neighbor_map=None):
        if neighbor_map is not None:
            # 借鉴ARGUS的邻域聚合
            neighbor_features = self.aggregate_neighbors(x, neighbor_map)
            neighbor_encoded = F.relu(self.neighbor_aggregator(neighbor_features))
            
            # 融合节点特征和邻域特征（借鉴ARGUS的融合思想）
            combined = torch.cat([x, neighbor_encoded], dim=1)
            x = F.relu(self.fusion_layer(combined))
        
        # VAE编码
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # VAE解码
        return self.decode(z), mu, logvar


# 定义损失函数
def loss_function(x_recon, x, mu, logvar):
    # 重构损失
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + KLD


# 借鉴ARGUS的邻域感知损失函数
def neighborhood_aware_loss_function(x_recon, x, mu, logvar, neighbor_map=None):
    """改进的损失函数，加入邻域信息"""
    # 基础VAE损失
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 借鉴ARGUS的邻域一致性损失
    neighbor_loss = 0
    if neighbor_map is not None:
        neighbor_loss = calculate_neighbor_consistency_loss(x_recon, x, neighbor_map)
    
    # 总损失（借鉴ARGUS的权重设计）
    total_loss = recon_loss + KLD + 0.1 * neighbor_loss
    
    return total_loss


def calculate_neighbor_consistency_loss(x_recon, x, neighbor_map):
    """借鉴ARGUS的邻域一致性计算"""
    consistency_loss = 0
    for node_id, neighbors in neighbor_map.items():
        if neighbors and len(neighbors) > 0:
            # 计算重构特征与原始特征的邻域一致性
            recon_neighbor_avg = torch.mean(x_recon[list(neighbors)], dim=0)
            orig_neighbor_avg = torch.mean(x[list(neighbors)], dim=0)
            consistency_loss += F.mse_loss(recon_neighbor_avg, orig_neighbor_avg)
    
    return consistency_loss