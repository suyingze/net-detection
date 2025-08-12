#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IForest参数调优测试脚本
测试不同参数组合对异常检测性能的影响
使用IQR阈值函数进行评估
"""

import numpy as np
import pickle as pkl
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 导入异常检测方法
from integrated_anomaly_detection import IntegratedAnomalyDetector

# 导入IQR阈值函数
from threshold_comparison import improved_threshold_v4

class IForestParameterTuning:
    """IForest参数调优测试类"""
    
    def __init__(self, dataset="mydata-flow"):
        self.dataset = dataset
        self.dataset_path = f'./dataset/{dataset}/'
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        print("Loading data...")
        
        # 加载训练特征
        with open(f"{self.dataset_path}train_features.pkl", "rb") as f:
            self.train_features = pkl.load(f)
        
        # 加载测试特征
        with open(f"{self.dataset_path}test_features.pkl", "rb") as f:
            self.test_features = pkl.load(f)
        
        # 加载节点映射
        with open(f"{self.dataset_path}test_node_map_idx.pkl", "rb") as f:
            self.node_map_idx = pkl.load(f)
        
        self.node_ids = list(self.node_map_idx)
        
        # 加载攻击节点信息
        self.attack_nodes = set()
        try:
            with open(f"{self.dataset_path}net_attack.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.attack_nodes.add(line)
            print(f"Loaded {len(self.attack_nodes)} attack nodes")
        except FileNotFoundError:
            print(f"Warning: Attack file not found: {self.dataset_path}net_attack.txt")
        
        print(f"Train features: {len(self.train_features)}")
        print(f"Test features: {len(self.test_features)}")
        print(f"Node IDs: {len(self.node_ids)}")
    
    def test_n_estimators(self):
        """测试不同的树的数量"""
        print("\n=== Testing Number of Trees (n_estimators) ===")
        
        n_estimators_list = [50, 100, 200, 300, 500]
        
        results = {}
        
        for n_estimators in n_estimators_list:
            print(f"\nTesting n_estimators: {n_estimators}")
            
            try:
                # 训练模型 - 使用pyod_threshold_test.py的默认值
                detector = IntegratedAnomalyDetector(
                    method='iforest',
                    contamination=0.1,           # 默认值
                    n_estimators=n_estimators,   # 调优参数
                    max_samples='auto',          # 默认值
                    max_features=1.0,            # 默认值
                    bootstrap=False,             # 默认值
                    random_state=42              # 默认值
                )
                
                detector.fit(self.train_features)
                
                # 获取异常分数
                train_scores = detector.predict(self.train_features)
                test_scores = detector.predict(self.test_features)
                
                # 详细调试信息
                print(f"   训练分数范围: {min(train_scores):.6f} - {max(train_scores):.6f}")
                
                # 使用与pyod_threshold_test.py相同的阈值计算方式
                # 使用98.55%分位数，确保结果可比
                threshold_percentile = 98.55
                threshold = np.percentile(train_scores, threshold_percentile)
                
                print(f"   使用百分位数: {threshold_percentile:.2f}%")
                print(f"   计算阈值: {threshold:.6f}")
                
                # 计算性能指标
                performance = self.calculate_performance_metrics(test_scores, threshold)
                performance['n_estimators'] = n_estimators
                performance['threshold'] = threshold
                
                results[f"树数量{n_estimators}"] = performance
                
                print(f"   Threshold: {threshold:.6f} (百分位数: {threshold_percentile:.2f}%)")
                print(f"   F1 Score: {performance['f1_score']:.4f}")
                print(f"   Precision: {performance['precision']:.4f}")
                print(f"   Recall: {performance['recall']:.4f}")
                
            except Exception as e:
                print(f"   Error: {e}")
                results[f"树数量{n_estimators}"] = {'error': str(e)}
        
        return results
    
    def test_max_samples(self):
        """测试不同的每棵树样本数"""
        print("\n=== Testing Max Samples per Tree ===")
        
        max_samples_list = ['auto', 100, 200, 500, 1000]
        
        results = {}
        
        for max_samples in max_samples_list:
            print(f"\nTesting max_samples: {max_samples}")
            
            try:
                # 训练模型 - 使用pyod_threshold_test.py的默认值
                detector = IntegratedAnomalyDetector(
                    method='iforest',
                    contamination=0.1,           # 默认值
                    n_estimators=100,            # 默认值
                    max_samples=max_samples,      # 调优参数
                    max_features=1.0,            # 默认值
                    bootstrap=False,             # 默认值
                    random_state=42              # 默认值
                )
                
                detector.fit(self.train_features)
                
                # 获取异常分数
                train_scores = detector.predict(self.train_features)
                test_scores = detector.predict(self.test_features)
                
                # 详细调试信息
                print(f"   训练分数范围: {min(train_scores):.6f} - {max(train_scores):.6f}")
                print(f"   测试分数范围: {min(test_scores):.6f} - {max(test_scores):.6f}")
                
                # 使用与pyod_threshold_test.py相同的阈值计算方式
                # 使用98.55%分位数，确保结果可比
                threshold_percentile = 98.55
                threshold = np.percentile(train_scores, threshold_percentile)
                
                print(f"   使用百分位数: {threshold_percentile:.2f}%")
                print(f"   计算阈值: {threshold:.6f}")
                
                # 计算性能指标
                performance = self.calculate_performance_metrics(test_scores, threshold)
                performance['max_samples'] = str(max_samples)
                performance['threshold'] = threshold
                
                results[f"样本数{max_samples}"] = performance
                
                print(f"   Threshold: {threshold:.6f} (百分位数: {threshold_percentile:.2f}%)")
                print(f"   F1 Score: {performance['f1_score']:.4f}")
                print(f"   Precision: {performance['precision']:.4f}")
                print(f"   Recall: {performance['recall']:.4f}")
                
            except Exception as e:
                print(f"   Error: {e}")
                results[f"样本数{max_samples}"] = {'error': str(e)}
        
        return results
    
    def test_contamination(self):
        """测试不同的异常比例"""
        print("\n=== Testing Contamination Rates ===")
        
        contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
        
        results = {}
        
        for contamination in contamination_rates:
            print(f"\nTesting contamination: {contamination}")
            
            try:
                # 训练模型 - 使用pyod_threshold_test.py的默认值
                detector = IntegratedAnomalyDetector(
                    method='iforest',
                    contamination=contamination,  # 调优参数
                    n_estimators=100,            # 默认值
                    max_samples='auto',          # 默认值
                    max_features=1.0,            # 默认值
                    bootstrap=False,             # 默认值
                    random_state=42              # 默认值
                )
                
                detector.fit(self.train_features)
                
                # 获取异常分数
                train_scores = detector.predict(self.train_features)
                test_scores = detector.predict(self.test_features)
                
                # 调试信息：查看异常分数范围
                print(f"   训练分数范围: {min(train_scores):.6f} - {max(train_scores):.6f}")
                print(f"   测试分数范围: {min(test_scores):.6f} - {max(test_scores):.6f}")
                
                # 使用与pyod_threshold_test.py相同的阈值计算方式
                # 使用98.55%分位数，确保结果可比
                threshold_percentile = 98.55
                threshold = np.percentile(train_scores, threshold_percentile)
                
                print(f"   使用百分位数: {threshold_percentile:.2f}%")
                print(f"   计算阈值: {threshold:.6f}")
                
                # 计算性能指标
                performance = self.calculate_performance_metrics(test_scores, threshold)
                performance['contamination'] = contamination
                performance['threshold'] = threshold
                
                results[f"异常比例{contamination}"] = performance
                
                print(f"   Threshold: {threshold:.6f} (百分位数: {threshold_percentile:.2f}%)")
                print(f"   F1 Score: {performance['f1_score']:.4f}")
                print(f"   Precision: {performance['precision']:.4f}")
                print(f"   Recall: {performance['recall']:.4f}")
                
            except Exception as e:
                print(f"   Error: {e}")
                results[f"异常比例{contamination}"] = {'error': str(e)}
        
        return results
    
    def test_max_features(self):
        """测试不同的特征数量"""
        print("\n=== Testing Max Features ===")
        
        # 扩展测试列表，增加更多选项
        max_features_list = ['auto', 'sqrt', 'log2', 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        
        results = {}
        
        for max_features in max_features_list:
            print(f"\nTesting max_features: {max_features}")
            
            try:
                # 训练模型 - 使用pyod_threshold_test.py的默认值
                detector = IntegratedAnomalyDetector(
                    method='iforest',
                    contamination=0.1,           # 默认值
                    n_estimators=100,            # 默认值
                    max_samples='auto',          # 默认值
                    max_features=max_features,    # 调优参数
                    bootstrap=False,             # 默认值
                    random_state=42              # 默认值
                )
                
                detector.fit(self.train_features)
                
                # 获取异常分数
                train_scores = detector.predict(self.train_features)
                test_scores = detector.predict(self.test_features)
                
                # 使用与pyod_threshold_test.py相同的阈值计算方式
                # 使用98.55%分位数，确保结果可比
                threshold_percentile = 98.55
                threshold = np.percentile(train_scores, threshold_percentile)
                
                # 计算性能指标
                performance = self.calculate_performance_metrics(test_scores, threshold)
                performance['max_features'] = str(max_features)
                performance['threshold'] = threshold
                
                results[f"特征数{max_features}"] = performance
                
                print(f"   Threshold: {threshold:.6f} (百分位数: {threshold_percentile:.2f}%)")
                print(f"   F1 Score: {performance['f1_score']:.4f}")
                print(f"   Precision: {performance['precision']:.4f}")
                print(f"   Recall: {performance['recall']:.4f}")
                
            except Exception as e:
                print(f"   Error: {e}")
                results[f"特征数{max_features}"] = {'error': str(e)}
        
        # 为特征数量测试单独生成清晰的性能图表
        self.create_max_features_chart(results)
        
        return results
    
    def create_max_features_chart(self, results):
        """为特征数量测试创建专门的性能图表"""
        print("\n=== 生成特征数量性能图表 ===")
        
        # 过滤掉有错误的结果
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("没有有效的结果")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # 准备数据
        method_names = list(valid_results.keys())
        f1_scores = [valid_results[name]['f1_score'] for name in method_names]
        precision_scores = [valid_results[name]['precision'] for name in method_names]
        recall_scores = [valid_results[name]['recall'] for name in method_names]
        
        # 找出最佳性能
        best_idx = np.argmax(f1_scores)
        
        # 第一个子图：F1分数对比
        bars1 = ax1.bar(range(len(method_names)), f1_scores, color='lightgreen', alpha=0.7, width=0.6)
        bars1[best_idx].set_color('red')
        bars1[best_idx].set_alpha(0.8)
        
        ax1.set_title('IForest 特征数量 F1分数对比', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('F1分数', fontsize=12)
        ax1.set_xticks(range(len(method_names)))
        ax1.set_xticklabels(method_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars1, f1_scores)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 第二个子图：Precision和Recall对比
        x_pos = np.arange(len(method_names))
        width = 0.35
        
        bars2 = ax2.bar(x_pos - width/2, precision_scores, width, label='Precision', color='skyblue', alpha=0.7)
        bars3 = ax2.bar(x_pos + width/2, recall_scores, width, label='Recall', color='lightcoral', alpha=0.7)
        
        # 高亮最佳性能行
        bars2[best_idx].set_color('red')
        bars3[best_idx].set_color('red')
        bars2[best_idx].set_alpha(0.8)
        bars3[best_idx].set_alpha(0.8)
        
        ax2.set_title('IForest 特征数量 Precision vs Recall 对比', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('分数', fontsize=12)
        ax2.set_xlabel('特征数量', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(method_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        # 添加数值标签
        for i, (p_score, r_score) in enumerate(zip(precision_scores, recall_scores)):
            ax2.text(x_pos[i] - width/2, p_score + 0.01, f'{p_score:.3f}', 
                    ha='center', va='bottom', fontsize=9)
            ax2.text(x_pos[i] + width/2, r_score + 0.01, f'{r_score:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # 添加最佳参数说明
        best_method_name = method_names[best_idx]
        best_f1 = f1_scores[best_idx]
        plt.figtext(0.5, 0.02, f'最佳特征数量: {best_method_name} (F1分数: {best_f1:.4f})', 
                    ha='center', fontsize=14, fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig('iforest_max_features_performance_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"特征数量性能图表已保存为 'iforest_max_features_performance_chart.png'")
        print(f"最佳特征数量: {best_method_name} (F1分数: {best_f1:.4f})")
    
    def test_bootstrap(self):
        """测试是否使用bootstrap采样"""
        print("\n=== Testing Bootstrap Sampling ===")
        
        bootstrap_options = [True, False]
        
        results = {}
        
        for bootstrap in bootstrap_options:
            print(f"\nTesting bootstrap: {bootstrap}")
            
            try:
                # 训练模型 - 使用pyod_threshold_test.py的默认值
                detector = IntegratedAnomalyDetector(
                    method='iforest',
                    contamination=0.1,           # 默认值
                    n_estimators=100,            # 默认值
                    max_samples='auto',          # 默认值
                    max_features=1.0,            # 默认值
                    bootstrap=bootstrap,         # 调优参数
                    random_state=42              # 默认值
                )
                
                detector.fit(self.train_features)
                
                # 获取异常分数
                train_scores = detector.predict(self.train_features)
                test_scores = detector.predict(self.test_features)
                
                # 使用与pyod_threshold_test.py相同的阈值计算方式
                # 使用98.55%分位数，确保结果可比
                threshold_percentile = 98.55
                threshold = np.percentile(train_scores, threshold_percentile)
                
                # 计算性能指标
                performance = self.calculate_performance_metrics(test_scores, threshold)
                performance['bootstrap'] = bootstrap
                performance['threshold'] = threshold
                
                results[f"Bootstrap{bootstrap}"] = performance
                
                print(f"   Threshold: {threshold:.6f} (百分位数: {threshold_percentile:.2f}%)")
                print(f"   F1 Score: {performance['f1_score']:.4f}")
                print(f"   Precision: {performance['precision']:.4f}")
                print(f"   Recall: {performance['recall']:.4f}")
                
            except Exception as e:
                print(f"   Error: {e}")
                results[f"Bootstrap{bootstrap}"] = {'error': str(e)}
        
        return results
    
    def calculate_performance_metrics(self, scores, threshold):
        """计算性能指标"""
        y_true = [1 if node_id in self.attack_nodes else 0 for node_id in self.node_ids]
        y_pred = [1 if score > threshold else 0 for score in scores]
        
        # 计算基本指标
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算其他指标
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        try:
            auc_score = roc_auc_score(y_true, scores)
        except:
            auc_score = 0.0
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
            pr_auc = auc(recall_curve, precision_curve)
        except:
            pr_auc = 0.0
        
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except:
            mcc = 0.0
        
        try:
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
        except:
            balanced_acc = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negative_rate': tnr,
            'auc': auc_score,
            'pr_auc': pr_auc,
            'mcc': mcc,
            'balanced_accuracy': balanced_acc
        }
    
    def create_performance_table(self, results, parameter_name):
        """创建性能对比表格"""
        print(f"\n=== {parameter_name} 性能对比表 ===")
        
        # 过滤掉有错误的结果
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("没有有效的结果")
            return
        
        # 创建表格数据
        table_data = []
        for method_name, result in valid_results.items():
            row = [
                method_name,
                f"{result['threshold']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1_score']:.4f}",
                f"{result['true_positives']}",
                f"{result['false_positives']}",
                f"{result['false_negatives']}",
                f"{result['true_negative_rate']:.4f}",
                f"{result['auc']:.4f}",
                f"{result['pr_auc']:.4f}",
                f"{result['mcc']:.4f}",
                f"{result['balanced_accuracy']:.4f}"
            ]
            table_data.append(row)
        
        # 找出最佳性能行
        best_row_index = 0
        best_score = 0
        
        for i, row in enumerate(table_data):
            f1_score = float(row[4])
            mcc_score = float(row[11])
            balanced_acc = float(row[12])
            
            composite_score = f1_score * 0.4 + mcc_score * 0.3 + balanced_acc * 0.3
            
            if composite_score > best_score:
                best_score = composite_score
                best_row_index = i
        
        # 创建表格
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('off')
        
        table = ax.table(cellText=table_data, 
                        colLabels=['参数设置', '阈值', '准确率', '召回率', 'F1分数', 'TP', 'FP', 'FN', 'TNN', 'AUC', 'PR-AUC', 'MCC', '平衡精度'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.08, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.08])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 2)
        
        # 高亮显示最佳性能行
        for col in range(len(table_data[0])):
            # 设置最佳行的背景色为浅绿色，字体加粗
            cell = table[(best_row_index + 1, col)]
            cell.set_facecolor('#90EE90')
            cell.set_text_props(weight='bold', size=10)
            
            # 设置表头样式
            header_cell = table[(0, col)]
            header_cell.set_facecolor('#FFE4B5')
            header_cell.set_text_props(weight='bold', size=10)
        
        # 设置表格标题
        plt.title(f'IForest {parameter_name} 参数调优性能对比表', fontsize=16, fontweight='bold', pad=20)
        
        # 添加最佳参数说明
        best_method_name = table_data[best_row_index][0]
        plt.figtext(0.5, 0.02, f'最佳参数: {best_method_name} (综合评分: {best_score:.4f})', 
                    ha='center', fontsize=12, fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig(f'iforest_{parameter_name}_performance_table.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"性能对比表格已保存为 'iforest_{parameter_name}_performance_table.png'")
        print(f"最佳参数: {best_method_name} (综合评分: {best_score:.4f})")
        
        return valid_results
    
    def create_performance_charts(self, all_results):
        """创建性能对比图表"""
        print("\n=== 创建性能对比图表 ===")
        
        # 准备数据
        parameters = ['树的数量', '每棵树样本数', '异常比例', '特征数量', 'Bootstrap采样']
        
        # 创建F1分数对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('IForest参数调优性能对比', fontsize=16, fontweight='bold')
        
        # 为每个参数创建F1分数对比图
        for i, (param_name, results) in enumerate(all_results.items()):
            if i >= 5:  # 只显示5个参数
                break
                
            ax = axes[i // 3, i % 3]
            
            # 过滤有效结果
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if valid_results:
                method_names = list(valid_results.keys())
                f1_scores = [valid_results[name]['f1_score'] for name in method_names]
                
                bars = ax.bar(method_names, f1_scores, color='lightgreen', alpha=0.7)
                
                # 高亮最佳性能
                best_idx = np.argmax(f1_scores)
                bars[best_idx].set_color('red')
                bars[best_idx].set_alpha(0.8)
                
                ax.set_title(f'{param_name} - F1分数对比')
                ax.set_ylabel('F1分数')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for j, (bar, score) in enumerate(zip(bars, f1_scores)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 隐藏多余的子图
        if len(all_results) < 6:
            for i in range(len(all_results), 6):
                ax = axes[i // 3, i % 3]
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('iforest_parameter_tuning_performance_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("性能对比图表已保存为 'iforest_parameter_tuning_performance_charts.png'")
    
    def run_complete_tuning(self):
        """运行完整的参数调优测试"""
        print("=== IForest参数调优测试 ===")
        
        all_results = {}
        
        # 1. 测试树的数量
        n_estimators_results = self.test_n_estimators()
        all_results['树的数量'] = n_estimators_results
        self.create_performance_table(n_estimators_results, '树的数量')
        
        # 2. 测试每棵树样本数
        max_samples_results = self.test_max_samples()
        all_results['每棵树样本数'] = max_samples_results
        self.create_performance_table(max_samples_results, '每棵树样本数')
        
        # 3. 测试异常比例
        contamination_results = self.test_contamination()
        all_results['异常比例'] = contamination_results
        self.create_performance_table(contamination_results, '异常比例')
        
        # 4. 测试特征数量
        max_features_results = self.test_max_features()
        all_results['特征数量'] = max_features_results
        self.create_performance_table(max_features_results, '特征数量')
        
        # 5. 测试Bootstrap采样
        bootstrap_results = self.test_bootstrap()
        all_results['Bootstrap采样'] = bootstrap_results
        self.create_performance_table(bootstrap_results, 'Bootstrap采样')
        
        # 6. 创建综合性能对比图表
        self.create_performance_charts(all_results)
        
        # 7. 保存所有结果
        with open('iforest_parameter_tuning_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n所有结果已保存到 'iforest_parameter_tuning_results.json'")
        print("=== 参数调优测试完成 ===")
        
        return all_results

# ========== 主函数 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IForest Parameter Tuning')
    parser.add_argument("--dataset", type=str, default="mydata-flow")
    args = parser.parse_args()
    
    # 创建调优实例
    tuner = IForestParameterTuning(dataset=args.dataset)
    
    # 运行完整调优测试
    results = tuner.run_complete_tuning()
    
    # 输出最佳参数组合
    print(f"\n=== 最佳参数组合推荐 ===")
    for param_name, param_results in results.items():
        best_method = max(param_results.items(), key=lambda x: x[1].get('f1_score', 0) if 'error' not in x[1] else 0)
        if 'error' not in best_method[1]:
            print(f"{param_name}: {best_method[0]} (F1: {best_method[1]['f1_score']:.4f})") 