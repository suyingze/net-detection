#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的阈值函数，使用多种策略来减少误报
特别针对攻击期间正常节点的误报问题
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
font_path = None
for font_name in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'PingFang SC', 'Heiti SC']:
    try:
        font_path = fm.findfont(fm.FontProperties(family=font_name))
        if font_path:
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    except:
        continue

if not font_path:
    print("Warning: No suitable Chinese font found. Please install one or specify a valid font path.")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

# ========== 改进的阈值函数 ==========

def improved_threshold_v2(mse_scores, n_components=2):
    """
    改进的阈值函数版本2：高斯混合模型
    
    Args:
        mse_scores: MSE分数列表
        n_components: 高斯分量数
    
    Returns:
        threshold: 基于GMM的阈值
    """
    # 转换为2D数组
    X = np.array(mse_scores).reshape(-1, 1)
    
    # 拟合高斯混合模型
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # 计算每个点的概率
    probabilities = gmm.predict_proba(X)
    
    # 找到概率最低的组件（异常组件）
    component_weights = gmm.weights_
    anomaly_component = np.argmin(component_weights)
    
    # 计算该组件的阈值（使用均值+2倍标准差）
    mean_anomaly = gmm.means_[anomaly_component][0]
    std_anomaly = np.sqrt(gmm.covariances_[anomaly_component][0][0])
    threshold = mean_anomaly + 2 * std_anomaly
    
    return threshold

def improved_threshold_v3(mse_scores, window_size=0.1):
    """
    改进的阈值函数版本3：局部异常因子(LOF)思想
    
    Args:
        mse_scores: MSE分数列表
        window_size: 窗口大小比例
    
    Returns:
        threshold: 基于局部密度的阈值
    """
    sorted_scores = np.sort(mse_scores)
    n = len(sorted_scores)
    
    # 计算局部密度
    window_idx = int(n * window_size)
    local_scores = sorted_scores[-window_idx:]
    
    # 计算局部统计量
    local_mean = np.mean(local_scores)
    local_std = np.std(local_scores)
    
    # 使用局部统计量计算阈值
    threshold = local_mean + 1.5 * local_std
    
    return threshold

def improved_threshold_v4(mse_scores, alpha=0.01):
    """
    改进的阈值函数版本4：基于IQR的异常检测
    
    Args:
        mse_scores: MSE分数列表
        alpha: 显著性水平
    
    Returns:
        threshold: 基于IQR的阈值
    """
    # 计算四分位数
    Q1 = np.percentile(mse_scores, 25)
    Q3 = np.percentile(mse_scores, 75)
    IQR = Q3 - Q1
    
    # 使用IQR方法计算异常阈值
    threshold = Q3 + 1.5 * IQR
    
    # 进一步调整基于数据分布
    upper_bound = np.percentile(mse_scores, 99.5)
    threshold = max(threshold, upper_bound)
    
    return threshold

def improved_threshold_v5(mse_scores, z_threshold=2.5):
    """
    改进的阈值函数版本5：基于Z-score的异常检测
    
    Args:
        mse_scores: MSE分数列表
        z_threshold: Z-score阈值
    
    Returns:
        threshold: 基于Z-score的阈值
    """
    # 计算均值和标准差
    mean_score = np.mean(mse_scores)
    std_score = np.std(mse_scores)
    
    # 使用Z-score方法计算阈值
    threshold = mean_score + z_threshold * std_score
    
    # 确保阈值不会过低
    min_threshold = np.percentile(mse_scores, 95)
    threshold = max(threshold, min_threshold)
    
    return threshold

def improved_threshold_v6(mse_scores, percentile=99.2):
    """
    改进的阈值函数版本6：基于百分位数的自适应方法
    
    Args:
        mse_scores: MSE分数列表
        percentile: 百分位数
    
    Returns:
        threshold: 基于自适应百分位数的阈值
    """
    # 计算基础百分位数阈值
    base_threshold = np.percentile(mse_scores, percentile)
    
    # 计算数据分布的偏度和峰度
    mean_score = np.mean(mse_scores)
    std_score = np.std(mse_scores)
    
    # 计算偏度
    skewness = np.mean(((mse_scores - mean_score) / std_score) ** 3)
    
    # 根据偏度调整阈值
    if abs(skewness) > 1.5:  # 高度偏斜
        # 使用更保守的百分位数
        adjusted_percentile = percentile - 0.3
        threshold = np.percentile(mse_scores, adjusted_percentile)
    elif abs(skewness) > 0.5:  # 中度偏斜
        # 使用稍微保守的百分位数
        adjusted_percentile = percentile - 0.1
        threshold = np.percentile(mse_scores, adjusted_percentile)
    else:  # 接近正态分布
        threshold = base_threshold
    
    return threshold

def plot_threshold_comparison_histogram(mse_scores, original_threshold, improved_results):
    """
    绘制阈值对比直方图
    
    Args:
        mse_scores: MSE分数列表
        original_threshold: 原始阈值
        improved_results: 改进方法的结果字典
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    plt.hist(mse_scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    
    # 绘制原始阈值
    plt.axvline(original_threshold, color='red', linestyle='-', linewidth=3,
               label=f'原始阈值: {original_threshold:.2f}')
    
    # 绘制改进阈值
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, (method_name, result) in enumerate(improved_results.items()):
        if i < len(colors):
            plt.axvline(result['threshold'], color=colors[i], linestyle='--', linewidth=2,
                       label=f'{method_name}: {result["threshold"]:.2f}')
    
    plt.title('原始 vs 改进阈值对比', fontsize=16, fontweight='bold')
    plt.xlabel('MSE分数', fontsize=14)
    plt.ylabel('频次', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_comparison_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("阈值对比直方图已保存为 'threshold_comparison_histogram.png'")

def plot_threshold_comparison_table(mse_scores, original_threshold, improved_results, 
                                   original_precision, original_recall, original_tp, original_fp, original_fn,
                                   attack_nodes, node_ids):
    """
    绘制阈值对比性能表格
    
    Args:
        mse_scores: MSE分数列表
        original_threshold: 原始阈值
        improved_results: 改进方法的结果字典
        original_precision: 原始精度
        original_recall: 原始召回率
        original_tp: 原始方法真正例数
        original_fp: 原始方法假正例数
        original_fn: 原始方法假负例数
        attack_nodes: 真实攻击节点集合
        node_ids: 节点ID列表
    """
    plt.figure(figsize=(16, 10))
    plt.axis('off')
    
    # 准备表格数据
    table_data = []
    
    # 计算原始方法的F1分数
    if (original_precision + original_recall) > 0:
        original_f1 = 2 * original_precision * original_recall / (original_precision + original_recall)
    else:
        original_f1 = 0.0
    
    # 计算原始方法的其他指标
    original_tn = len([i for i, node_id in enumerate(node_ids) if node_id not in attack_nodes and mse_scores[i] <= original_threshold])
    original_tnr = original_tn / (original_tn + original_fp) if (original_tn + original_fp) > 0 else 0.0
    
    # 计算原始方法的AUC
    y_true_original = [1 if node_id in attack_nodes else 0 for node_id in node_ids]
    try:
        original_auc = roc_auc_score(y_true_original, mse_scores)
    except:
        original_auc = 0.0
    
    # 计算原始方法的PR-AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_original, mse_scores)
        original_pr_auc = auc(recall_curve, precision_curve)
    except:
        original_pr_auc = 0.0
    
    # 计算原始方法的MCC和平衡精度
    y_pred_original = [1 if score > original_threshold else 0 for score in mse_scores]
    try:
        original_mcc = matthews_corrcoef(y_true_original, y_pred_original)
    except:
        original_mcc = 0.0
    
    try:
        original_balanced_acc = balanced_accuracy_score(y_true_original, y_pred_original)
    except:
        original_balanced_acc = 0.0
    
    # 添加原始方法
    original_row = ['原始方法', f'{original_threshold:.4f}', 
                    f'{original_precision:.4f}', 
                    f'{original_recall:.4f}',
                    f'{original_f1:.4f}',
                    f'{original_tp}',
                    f'{original_fp}',
                    f'{original_fn}',
                    f'{original_tnr:.4f}',
                    f'{original_auc:.4f}',
                    f'{original_pr_auc:.4f}',
                    f'{original_mcc:.4f}',
                    f'{original_balanced_acc:.4f}']
    table_data.append(original_row)
    
    # 添加改进方法
    for method_name, result in improved_results.items():
        # 计算其他指标
        tn = len([i for i, node_id in enumerate(node_ids) if node_id not in attack_nodes and mse_scores[i] <= result['threshold']])
        tnr = tn / (tn + result['false_positives']) if (tn + result['false_positives']) > 0 else 0.0
        
        # 计算AUC
        try:
            method_auc = roc_auc_score(y_true_original, mse_scores)
        except:
            method_auc = 0.0
        
        # 计算PR-AUC
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_original, mse_scores)
            method_pr_auc = auc(recall_curve, precision_curve)
        except:
            method_pr_auc = 0.0
        
        # 计算MCC和平衡精度
        y_pred_method = [1 if score > result['threshold'] else 0 for score in mse_scores]
        try:
            method_mcc = matthews_corrcoef(y_true_original, y_pred_method)
        except:
            method_mcc = 0.0
        
        try:
            method_balanced_acc = balanced_accuracy_score(y_true_original, y_pred_method)
        except:
            method_balanced_acc = 0.0
        
        method_row = [method_name[:15], f'{result["threshold"]:.4f}',
                      f'{result["precision"]:.4f}',
                      f'{result["recall"]:.4f}',
                      f'{result["f1_score"]:.4f}',
                      f'{result["true_positives"]}',
                      f'{result["false_positives"]}',
                      f'{result["false_negatives"]}',
                      f'{tnr:.4f}',
                      f'{method_auc:.4f}',
                      f'{method_pr_auc:.4f}',
                      f'{method_mcc:.4f}',
                      f'{method_balanced_acc:.4f}']
        table_data.append(method_row)
    
    # 找出性能最好的行（基于F1分数、MCC和平衡精度的综合评估）
    best_row_index = 0
    best_score = 0
    
    for i, row in enumerate(table_data):
        # 综合评分：F1分数 * 0.4 + MCC * 0.3 + 平衡精度 * 0.3
        f1_score = float(row[4])
        mcc_score = float(row[11])
        balanced_acc = float(row[12])
        
        # 处理可能的NaN值
        if np.isnan(f1_score): f1_score = 0
        if np.isnan(mcc_score): mcc_score = 0
        if np.isnan(balanced_acc): balanced_acc = 0
        
        composite_score = f1_score * 0.4 + mcc_score * 0.3 + balanced_acc * 0.3
        
        if composite_score > best_score:
            best_score = composite_score
            best_row_index = i
    
    # 创建表格
    table = plt.table(cellText=table_data, 
                     colLabels=['方法', '阈值', '准确率', '召回率', 'F1分数', 'TP', 'FP', 'FN', 'TNN', 'AUC', 'PR-AUC', 'MCC', '平衡精度'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.12, 0.08, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.08])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2)
    
    # 高亮显示最佳性能行
    for col in range(len(table_data[0])):
        # 设置最佳行的背景色为浅绿色，字体加粗
        cell = table[(best_row_index + 1, col)]  # +1 因为第0行是表头
        cell.set_facecolor('#90EE90')  # 浅绿色背景
        cell.set_text_props(weight='bold', size=11)
        
        # 设置表头样式
        header_cell = table[(0, col)]
        header_cell.set_facecolor('#FFE4B5')  # 浅橙色背景
        header_cell.set_text_props(weight='bold', size=11)
    
    # 设置表格标题
    plt.title('异常检测方法性能对比表', fontsize=18, fontweight='bold', pad=20)
    
    # 添加最佳方法说明
    best_method_name = table_data[best_row_index][0]
    plt.figtext(0.5, 0.02, f'最佳性能方法: {best_method_name} (综合评分: {best_score:.4f})', 
                ha='center', fontsize=14, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('threshold_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"性能对比表格已保存为 'threshold_comparison_table.png'")
    print(f"最佳性能方法: {best_method_name} (综合评分: {best_score:.4f})")

def plot_threshold_comparison_with_original_detailed(mse_scores, original_threshold, improved_results, 
                                                   original_precision, original_recall, original_tp, original_fp, original_fn,
                                                   attack_nodes, node_ids):
    """
    绘制包含原始阈值的对比图（详细版本，包含TP、FP、FN、AUC等指标）
    现在分别生成两个图：直方图和表格
    
    Args:
        mse_scores: MSE分数列表
        original_threshold: 原始阈值
        improved_results: 改进方法的结果字典
        original_precision: 原始精度
        original_recall: 原始召回率
        original_tp: 原始方法真正例数
        original_fp: 原始方法假正例数
        original_fn: 原始方法假负例数
        attack_nodes: 真实攻击节点集合
        node_ids: 节点ID列表
    """
    # 生成直方图
    plot_threshold_comparison_histogram(mse_scores, original_threshold, improved_results)
    
    # 生成性能表格
    plot_threshold_comparison_table(mse_scores, original_threshold, improved_results,
                                   original_precision, original_recall, original_tp, original_fp, original_fn,
                                   attack_nodes, node_ids)
    
    print("对比图表已分别保存为直方图和表格")

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

# 使用示例
if __name__ == "__main__":
    import argparse
    import pickle as pkl
    import json
    import torch
    from net_detect import VAE, get_MSE
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='阈值函数对比测试')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    args = parser.parse_args()
    dataset = args.dataset
    dataset_path = f'./dataset/{dataset}/'
    
    # 数据集文件映射
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
    
    TEST_FILE = dataset_file_map[dataset]['test']
    ATTACK_FILE = f'{dataset_path}net_attack.txt'
    VAE_PATH = './models/VAE.model'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=== 使用真实测试数据的阈值函数对比测试 ===")
    
    # 加载VAE模型
    print("正在加载VAE模型...")
    model = VAE().to(device)
    model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    model.eval()
    
    # 使用训练数据计算原本阈值（与您的原本方法一致）
    print("正在加载训练数据计算原本阈值...")
    with open(f"dataset/{dataset}/train_features.pkl", "rb") as f:
        train_features = pkl.load(f)
    
    # 计算训练数据的MSE分数
    train_mse = get_MSE(model, train_features, device)
    print(f"训练数据MSE分数范围: {min(train_mse):.4f} - {max(train_mse):.4f}")
    print(f"训练数据MSE统计:")
    print(f"  均值: {np.mean(train_mse):.4f}")
    print(f"  标准差: {np.std(train_mse):.4f}")
    print(f"  中位数: {np.median(train_mse):.4f}")
    print(f"  99.55%分位数: {np.percentile(train_mse, 99.55):.4f}")
    print()
    
    # 计算原本阈值（基于训练数据）
    original_threshold_value = np.percentile(train_mse, 99.55)
    print(f"基于训练数据的原本阈值 (99.55%): {original_threshold_value:.4f}")
    
 
    
    # 加载测试特征数据
    print("正在加载测试特征数据...")
    with open(f"dataset/{dataset}/test_features.pkl", "rb") as f:
        features = pkl.load(f)
    with open(f"dataset/{dataset}/test_node_map_idx.pkl", "rb") as f:
        node_map_idx = pkl.load(f)
    node_ids = list(node_map_idx)
    
    # 计算测试数据的MSE分数
    print("正在计算测试数据MSE分数...")
    mse_scores = get_MSE(model, features, device)
    
    print(f"数据点总数: {len(mse_scores)}")
    print(f"节点ID数量: {len(node_ids)}")
    print(f"MSE分数范围: {min(mse_scores):.4f} - {max(mse_scores):.4f}")
    
    # 加载真实攻击节点信息
    print("正在加载攻击节点信息...")
    attack_nodes = set()
    try:
        with open(ATTACK_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    attack_nodes.add(line)
        print(f"真实攻击节点数量: {len(attack_nodes)}")
    except FileNotFoundError:
        print(f"警告: 未找到攻击文件 {ATTACK_FILE}")
        attack_nodes = set()
    
    # 加载攻击连接信息
    attack_connections = set()
    attack_target_nodes = set()
    try:
        with open(f"{dataset_path}{TEST_FILE}", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    src_node = data.get('src_ip_port', '')
                    dest_node = data.get('dest_ip_port', '')
                    if src_node and dest_node:
                        attack_connections.add((src_node, dest_node))
                except json.JSONDecodeError:
                    continue
        
        for src, dest in attack_connections:
            attack_target_nodes.add(dest)
        print(f"攻击目标节点数量: {len(attack_target_nodes)}")
    except Exception as e:
        print(f"警告: 加载攻击连接信息失败: {e}")
    
    # 分类数据
    normal_scores = []
    attack_source_scores = []
    attack_target_scores = []
    other_scores = []
    
    for i, (node_id, mse_score) in enumerate(zip(node_ids, mse_scores)):
        if node_id in attack_nodes:
            attack_source_scores.append(mse_score)
        elif node_id in attack_target_nodes:
            attack_target_scores.append(mse_score)
        else:
            normal_scores.append(mse_score)
    
    print(f"正常节点数量: {len(normal_scores)}")
    print(f"攻击源节点数量: {len(attack_source_scores)}")
    print(f"攻击目标节点数量: {len(attack_target_scores)}")
    print()
    
    # 加载测试数据和时间戳映射
    print("正在加载时间戳信息...")
    node_timestamps = load_test_data_and_timestamps(dataset_path, TEST_FILE)
    print(f"加载到的时间戳数量: {len(node_timestamps)}")
    print()
    
    # 添加数据分布分析
    print("\n=== 数据分布分析 ===")
    print(f"MSE分数统计:")
    print(f"  均值: {np.mean(mse_scores):.4f}")
    print(f"  标准差: {np.std(mse_scores):.4f}")
    print(f"  中位数: {np.median(mse_scores):.4f}")
    print(f"  25%分位数: {np.percentile(mse_scores, 25):.4f}")
    print(f"  75%分位数: {np.percentile(mse_scores, 75):.4f}")
    print(f"  90%分位数: {np.percentile(mse_scores, 90):.4f}")
    print(f"  95%分位数: {np.percentile(mse_scores, 95):.4f}")
    print(f"  99%分位数: {np.percentile(mse_scores, 99):.4f}")
    print(f"  99.5%分位数: {np.percentile(mse_scores, 99.5):.4f}")
    print()
    
    # 分析攻击节点和正常节点的MSE分布
    if len(attack_source_scores) > 0:
        print(f"攻击源节点MSE统计:")
        print(f"  数量: {len(attack_source_scores)}")
        print(f"  均值: {np.mean(attack_source_scores):.4f}")
        print(f"  标准差: {np.std(attack_source_scores):.4f}")
        print(f"  最小值: {min(attack_source_scores):.4f}")
        print(f"  最大值: {max(attack_source_scores):.4f}")
        print(f"  25%分位数: {np.percentile(attack_source_scores, 25):.4f}")
        print(f"  50%分位数: {np.percentile(attack_source_scores, 50):.4f}")
        print(f"  75%分位数: {np.percentile(attack_source_scores, 75):.4f}")
        print(f"  90%分位数: {np.percentile(attack_source_scores, 90):.4f}")
        print()
    
    if len(normal_scores) > 0:
        print(f"正常节点MSE统计:")
        print(f"  数量: {len(normal_scores)}")
        print(f"  均值: {np.mean(normal_scores):.4f}")
        print(f"  标准差: {np.std(normal_scores):.4f}")
        print(f"  最小值: {min(normal_scores):.4f}")
        print(f"  最大值: {max(normal_scores):.4f}")
        print(f"  25%分位数: {np.percentile(normal_scores, 25):.4f}")
        print(f"  50%分位数: {np.median(normal_scores):.4f}")
        print(f"  75%分位数: {np.percentile(normal_scores, 75):.4f}")
        print(f"  90%分位数: {np.percentile(normal_scores, 90):.4f}")
        print()
    
    print()
    
    # ========== 原本的阈值函数测试 ==========
    print("--- 原本的阈值函数 ---")
    # 使用训练数据计算的阈值，而不是测试数据
    original_threshold_value = np.percentile(train_mse, 99.55)
    print(f"原始阈值 (99.55%): {original_threshold_value:.4f}")
    
    # 计算原本方法的完整性能（与警报文件一致）
    original_anomalies = []
    for i, (node_id, mse_score) in enumerate(zip(node_ids, mse_scores)):
        if mse_score > original_threshold_value:
            is_attack = node_id in attack_nodes
            original_anomalies.append({
                'node_id': node_id,
                'mse': mse_score,
                'is_attack': is_attack
            })
    
    original_true_positives = sum(1 for a in original_anomalies if a['is_attack'])
    original_false_positives = sum(1 for a in original_anomalies if not a['is_attack'])
    original_false_negatives = len(attack_source_scores) - original_true_positives
    
    if len(original_anomalies) > 0:
        original_precision = original_true_positives / len(original_anomalies)
    else:
        original_precision = 0.0
        
    if len(attack_source_scores) > 0:
        original_recall = original_true_positives / len(attack_source_scores)
    else:
        original_recall = 0.0
    
    if len(attack_source_scores) > 0:
        original_detection_rate = original_true_positives / len(attack_source_scores)
    else:
        original_detection_rate = 0.0
    
    if len(normal_scores) > 0:
        original_false_positive_rate = original_false_positives / len(normal_scores)
    else:
        original_false_positive_rate = 0.0
    
    print(f"原本方法检测率: {original_detection_rate:.2%}")
    print(f"原本方法误报率: {original_false_positive_rate:.2%}")
    print(f"原本方法精度: {original_precision:.4f}")
    print(f"原本方法召回率: {original_recall:.4f}")
    print(f"原本方法TP: {original_true_positives}, FP: {original_false_positives}, FN: {original_false_negatives}")
    print()
    
    # 计算原本方法的F1分数
    if (original_precision + original_recall) > 0:
        original_f1 = 2 * original_precision * original_recall / (original_precision + original_recall)
    else:
        original_f1 = 0.0
    
    print("原本方法:")
    print(f"  阈值: {original_threshold_value:.4f}")
    print(f"  精度: {original_precision:.4f}")
    print(f"  召回率: {original_recall:.4f}")
    print(f"  F1分数: {original_f1:.4f}")
    print(f"  TP: {original_true_positives}, FP: {original_false_positives}, FN: {original_false_negatives}")
    print()
    
    # ========== 改进的阈值函数测试 ==========
    print("--- 改进的阈值函数 ---")
    
    # 测试各种改进方法（都基于训练数据计算阈值）
    improved_methods = {
        'GMM': lambda x: improved_threshold_v2(x),
        '局部异常因子': lambda x: improved_threshold_v3(x),
        'IQR': lambda x: improved_threshold_v4(x),
        'Z-score': lambda x: improved_threshold_v5(x),
        '自适应百分位数': lambda x: improved_threshold_v6(x)
    }
    
    results = {}
    
    for method_name, method_func in improved_methods.items():
        try:
            # 使用训练数据计算改进阈值
            improved_threshold_value = method_func(train_mse)
            
            # 添加调试信息
            if method_name in ['攻击感知', 'Z-score', '自适应百分位数']:
                print(f"  {method_name}调试信息:")
                print(f"    训练数据99.0%分位数: {np.percentile(train_mse, 99.0):.4f}")
                print(f"    训练数据均值: {np.mean(train_mse):.4f}")
                print(f"    训练数据标准差: {np.std(train_mse):.4f}")
                print(f"    计算出的阈值: {improved_threshold_value:.4f}")
                
                # 为自适应百分位数添加特殊调试信息
                if method_name == '自适应百分位数':
                    mean_score = np.mean(train_mse)
                    std_score = np.std(train_mse)
                    skewness = np.mean(((train_mse - mean_score) / std_score) ** 3)
                    print(f"    数据偏度: {skewness:.4f}")
                    print(f"    基础百分位数: 99.2%")
                    print(f"    调整后百分位数: {99.2 - 0.3 if abs(skewness) > 1.5 else 99.2 - 0.1 if abs(skewness) > 0.5 else 99.2}")
                
                # 为Z-score添加特殊调试信息
                if method_name == 'Z-score':
                    mean_score = np.mean(train_mse)
                    std_score = np.std(train_mse)
                    z_threshold = 2.5
                    print(f"    均值: {mean_score:.4f}")
                    print(f"    标准差: {std_score:.4f}")
                    print(f"    Z-score阈值: {z_threshold}")
                    print(f"    理论阈值: {mean_score + z_threshold * std_score:.4f}")
                    print(f"    最小阈值(95%分位数): {np.percentile(train_mse, 95):.4f}")
            
            # 使用测试数据计算性能（与警报文件计算方式一致）
            anomalies_for_method = []
            for i, (node_id, mse_score) in enumerate(zip(node_ids, mse_scores)):
                if mse_score > improved_threshold_value:
                    is_attack = node_id in attack_nodes
                    anomalies_for_method.append({
                        'node_id': node_id,
                        'mse': mse_score,
                        'is_attack': is_attack
                    })
            
            # 计算性能指标（与警报文件一致）
            true_positives = sum(1 for a in anomalies_for_method if a['is_attack'])
            false_positives = sum(1 for a in anomalies_for_method if not a['is_attack'])
            false_negatives = len(attack_source_scores) - true_positives
            
            if len(anomalies_for_method) > 0:
                precision = true_positives / len(anomalies_for_method)
            else:
                precision = 0.0
                
            if len(attack_source_scores) > 0:
                recall = true_positives / len(attack_source_scores)
            else:
                recall = 0.0
                
            if (precision + recall) > 0:
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                f1_score = 0.0
            
            results[method_name] = {
                'threshold': improved_threshold_value,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
            
            print(f"{method_name}:")
            print(f"  阈值: {improved_threshold_value:.4f}")
            print(f"  精度: {precision:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  F1分数: {f1_score:.4f}")
            print(f"  TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
            print()
            
        except Exception as e:
            print(f"{method_name}: 失败 - {e}")
            print()
    
    # ========== 对比分析 ==========
    print("=== 对比分析 ===")
    print(f"原始阈值: {original_threshold_value:.4f}")
    print(f"原始检测率: {original_detection_rate:.2%}")
    print(f"原始误报率: {original_false_positive_rate:.2%}")
    print()
    
    # 找出最佳改进方法
    best_method = None
    best_improvement = 0
    
    for method_name, result in results.items():
        # 计算综合改进（提高F1分数）
        f1_improvement = result['f1_score'] - original_f1  # 使用实际计算出的原始F1分数
        
        if f1_improvement > best_improvement:
            best_improvement = f1_improvement
            best_method = method_name
    
    if best_method:
        print(f"最佳改进方法: {best_method}")
        print(f"F1分数改进: {best_improvement:.4f}")
        best_result = results[best_method]
        print(f"  阈值: {best_result['threshold']:.4f}")
        print(f"  精度: {best_result['precision']:.4f}")
        print(f"  召回率: {best_result['recall']:.4f}")
        print(f"  F1分数: {best_result['f1_score']:.4f}")
        print(f"  TP: {best_result['true_positives']}, FP: {best_result['false_positives']}, FN: {best_result['false_negatives']}")
    else:
        print("没有找到改进的方法")
    
    # ========== 可视化对比 ==========
    print("\n=== 生成对比图表 ===")
    plot_threshold_comparison_with_original_detailed(mse_scores, original_threshold_value, results, 
                                                     original_precision, original_recall, original_true_positives, original_false_positives, original_false_negatives,
                                                     attack_nodes, node_ids)
    
    # ========== 输出最佳方法的警报文件 ==========
    if best_method:
        print(f"\n=== 输出最佳方法的警报文件 ===")
        best_result = results[best_method]
        best_threshold = best_result['threshold']
        
        # 生成警报列表
        anomalies = []
        for i, (node_id, mse_score) in enumerate(zip(node_ids, mse_scores)):
            if mse_score > best_threshold:
                # 获取节点对应的时间戳
                timestamp = node_timestamps.get(node_id, "unknown")
                # 判断是否为真实攻击节点
                is_attack = "是" if node_id in attack_nodes else "否"
                # 判断节点类型
                if node_id in attack_nodes:
                    node_type = "攻击源节点"
                elif node_id in attack_target_nodes:
                    node_type = "攻击目标节点"
                else:
                    node_type = "误报节点"
                anomalies.append({
                    'node_id': node_id,
                    'timestamp': timestamp,
                    'mse': mse_score,
                    'threshold': best_threshold,
                    'is_attack': is_attack,
                    'node_type': node_type
                })
        
        # 输出警报文件
        output_file = f'{dataset_path}net_alarms_{best_method.replace(" ", "_")}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for anomaly in anomalies:
                f.write(f"{anomaly['node_id']} | {anomaly['timestamp']} | {anomaly['mse']:.4f} | {anomaly['threshold']:.4f} | {anomaly['is_attack']} | {anomaly['node_type']}\n")
        
        print(f"警报文件已保存: {output_file}")
        print(f"检测到的异常节点数量: {len(anomalies)}")
        
        # 计算详细的性能指标
        true_positives = sum(1 for a in anomalies if a['is_attack'] == "是")
        false_positives = sum(1 for a in anomalies if a['is_attack'] == "否")
        false_negatives = len(attack_source_scores) - true_positives
        
        precision = true_positives / len(anomalies) if len(anomalies) > 0 else 0
        recall = true_positives / len(attack_source_scores) if len(attack_source_scores) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"真正(TP): {true_positives}")
        print(f"假正(FP): {false_positives}")
        print(f"假负(FN): {false_negatives}")
        print(f"精度(Precision): {precision:.4f}")
        print(f"召回率(Recall): {recall:.4f}")
        print(f"F1分数(F1-score): {f1_score:.4f}") 