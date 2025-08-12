#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyOD模型阈值比较测试
选择PyOD模型，用FastText数据训练，然后用阈值比较系统评估
"""

import numpy as np
import pickle as pkl
import json
import argparse
import torch
from integrated_anomaly_detection import IntegratedAnomalyDetector

# 导入现有的阈值比较系统
from threshold_comparison import (
    improved_threshold_v2, improved_threshold_v3, improved_threshold_v4,
    improved_threshold_v5, improved_threshold_v6,
    plot_threshold_comparison_with_original_detailed
)

def test_pyod_with_threshold_comparison(method='iforest', dataset='mydata-flow'):
    """测试PyOD模型并使用阈值比较系统"""
    print(f"=== PyOD {method.upper()} 阈值比较测试 ===")
    
    # 数据集配置
    dataset_path = f'./dataset/{dataset}/'
    
    # 加载FastText数据
    print("Loading FastText features...")
    with open(f"{dataset_path}train_features.pkl", "rb") as f:
        train_features = pkl.load(f)
    
    with open(f"{dataset_path}test_features.pkl", "rb") as f:
        test_features = pkl.load(f)
    
    with open(f"{dataset_path}test_node_map_idx.pkl", "rb") as f:
        node_map_idx = pkl.load(f)
    
    node_ids = list(node_map_idx)
    
    # 加载攻击节点信息
    attack_nodes = set()
    try:
        with open(f"{dataset_path}net_attack.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    attack_nodes.add(line)
        print(f"Loaded {len(attack_nodes)} attack nodes")
    except FileNotFoundError:
        print(f"Warning: Attack file not found")
    
    print(f"Train features: {len(train_features)}")
    print(f"Test features: {len(test_features)}")
    print(f"Node IDs: {len(node_ids)}")
    
    try:
        # 1. 训练PyOD模型
        print(f"\n1. Training PyOD {method} model...")
        detector = IntegratedAnomalyDetector(method=method, contamination=0.1)
        detector.fit(train_features)
        
        # 2. 获取异常分数（替代VAE的get_MSE）
        print(f"2. Getting anomaly scores...")
        anomaly_scores = detector.predict(test_features)
        
        print(f"   Scores range: {min(anomaly_scores):.4f} - {max(anomaly_scores):.4f}")
        print(f"   Mean score: {np.mean(anomaly_scores):.4f}")
        print(f"   Std score: {np.std(anomaly_scores):.4f}")
        
        # 3. 使用现有的阈值比较系统
        print(f"3. Running threshold comparison...")
        
        # 计算训练数据的分数用于阈值计算
        train_scores = detector.predict(train_features)
        
        # 使用现有的阈值函数
        improved_methods = {
            'GMM': lambda x: improved_threshold_v2(x),
            '局部异常因子': lambda x: improved_threshold_v3(x),
            'IQR': lambda x: improved_threshold_v4(x),
            'Z-score': lambda x: improved_threshold_v5(x),
            '自适应百分位数': lambda x: improved_threshold_v6(x)
        }
        
        # 计算原始阈值（基于训练数据）
        original_threshold = np.percentile(train_scores, 98.55)
        
        # 计算改进的阈值
        improved_results = {}
        for method_name, method_func in improved_methods.items():
            try:
                threshold = method_func(train_scores)
                
                # 计算性能指标
                y_true = [1 if node_id in attack_nodes else 0 for node_id in node_ids]
                y_pred = [1 if score > threshold else 0 for score in anomaly_scores]
                
                tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
                fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
                fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                improved_results[method_name] = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn
                }
                
                print(f"   {method_name}: threshold={threshold:.4f}, F1={f1_score:.4f}")
                
            except Exception as e:
                print(f"   Error with {method_name}: {e}")
        
        # 4. 生成对比图表
        print(f"4. Generating comparison chart...")
        
        # 计算原始方法性能
        y_true = [1 if node_id in attack_nodes else 0 for node_id in node_ids]
        y_pred_original = [1 if score > original_threshold else 0 for score in anomaly_scores]
        
        tp_original = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_original[i] == 1)
        fp_original = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred_original[i] == 1)
        fn_original = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred_original[i] == 0)
        
        precision_original = tp_original / (tp_original + fp_original) if (tp_original + fp_original) > 0 else 0
        recall_original = tp_original / (tp_original + fn_original) if (tp_original + fn_original) > 0 else 0
        
        # 生成图表
        plot_threshold_comparison_with_original_detailed(
            anomaly_scores, original_threshold, improved_results,
            precision_original, recall_original, tp_original, fp_original, fn_original,
            attack_nodes, node_ids
        )
        
        # 5. 保存模型
        model_path = f'./models/{method}_pyod_model.pkl'
        detector.save_model(model_path)
        print(f"5. Model saved to {model_path}")
        
        print(f"\n✓ PyOD {method.upper()} successfully tested with threshold comparison!")
        
    except Exception as e:
        print(f"✗ Error testing {method}: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyOD模型阈值比较测试')
    parser.add_argument("--method", type=str, default="iforest", 
                       choices=['iforest', 'copod', 'lof', 'cblof', 'knn', 'ocsvm', 'ecod', 'autoencoder', 'vae'],
                       help="PyOD模型选择")
    parser.add_argument("--dataset", type=str, default="mydata-flow",
                       help="数据集名称")
    args = parser.parse_args()
    
    print("可用的PyOD模型:")
    print("  - iforest: Isolation Forest")
    print("  - copod: COPOD (Copula-based Outlier Detection)")
    print("  - lof: Local Outlier Factor")
    print("  - cblof: Cluster-based Local Outlier Factor")
    print("  - knn: k-Nearest Neighbors")
    print("  - ocsvm: One-Class SVM")
    print("  - ecod: Empirical Cumulative Distribution")
    print("  - autoencoder: AutoEncoder")
    print("  - vae: Variational AutoEncoder")
    print()
    
    test_pyod_with_threshold_comparison(args.method, args.dataset)

if __name__ == "__main__":
    main() 