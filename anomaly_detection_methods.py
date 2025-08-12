#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多种异常检测方法，可以直接调用接口
适配FASTtext数据格式
"""

import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ========== PyOD 异常检测方法 ==========
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.cblof import CBLOF
    from pyod.models.knn import KNN
    from pyod.models.ocsvm import OCSVM
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE as PyOD_VAE
    from pyod.models.rod import ROD
    from pyod.models.sos import SOS
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.deep_svdd import DeepSVDD
    PYOD_AVAILABLE = True
except ImportError:
    print("PyOD not available. Install with: pip install pyod")
    PYOD_AVAILABLE = False

# ========== 其他异常检测库 ==========
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Scikit-learn not available")
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available")
    TORCH_AVAILABLE = False

class AnomalyDetectionMethods:
    """多种异常检测方法的统一接口"""
    
    def __init__(self, feature_dim=64):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.models = {}
        self.scores = {}
        
    def load_fasttext_features(self, features_path):
        """加载FASTtext特征数据"""
        with open(features_path, 'rb') as f:
            features = pkl.load(f)
        return np.array(features)
    
    def preprocess_features(self, features):
        """预处理特征数据"""
        # 标准化
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled
    
    def train_and_detect_pyod(self, features, method_name='iforest'):
        """使用PyOD进行异常检测"""
        if not PYOD_AVAILABLE:
            print("PyOD not available")
            return None
            
        features_scaled = self.preprocess_features(features)
        
        # 选择模型
        if method_name == 'iforest':
            model = IForest(contamination=0.1, random_state=42)
        elif method_name == 'lof':
            model = LOF(contamination=0.1)
        elif method_name == 'cblof':
            model = CBLOF(contamination=0.1, random_state=42)
        elif method_name == 'knn':
            model = KNN(contamination=0.1)
        elif method_name == 'ocsvm':
            model = OCSVM(contamination=0.1)
        elif method_name == 'copod':
            model = COPOD(contamination=0.1)
        elif method_name == 'ecod':
            model = ECOD(contamination=0.1)
        elif method_name == 'rod':
            model = ROD(contamination=0.1)
        elif method_name == 'sos':
            model = SOS(contamination=0.1)
        else:
            print(f"Unknown method: {method_name}")
            return None
        
        # 训练模型
        model.fit(features_scaled)
        
        # 预测异常分数
        scores = model.decision_function(features_scaled)
        labels = model.predict(features_scaled)
        
        self.models[method_name] = model
        self.scores[method_name] = scores
        
        return scores, labels
    
    def train_and_detect_sklearn(self, features, method_name='isolation_forest'):
        """使用Scikit-learn进行异常检测"""
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn not available")
            return None
            
        features_scaled = self.preprocess_features(features)
        
        # 选择模型
        if method_name == 'isolation_forest':
            model = IsolationForest(contamination=0.1, random_state=42)
        elif method_name == 'local_outlier_factor':
            model = LocalOutlierFactor(contamination=0.1, novelty=False)
        elif method_name == 'one_class_svm':
            model = OneClassSVM(nu=0.1)
        elif method_name == 'elliptic_envelope':
            model = EllipticEnvelope(contamination=0.1, random_state=42)
        else:
            print(f"Unknown method: {method_name}")
            return None
        
        # 训练模型
        model.fit(features_scaled)
        
        # 预测异常分数
        if method_name == 'local_outlier_factor':
            scores = -model.negative_outlier_factor_
            labels = model.fit_predict(features_scaled)
        else:
            scores = -model.decision_function(features_scaled)
            labels = model.predict(features_scaled)
        
        self.models[method_name] = model
        self.scores[method_name] = scores
        
        return scores, labels
    
    def train_and_detect_autoencoder(self, features, method_name='autoencoder'):
        """使用自编码器进行异常检测"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available")
            return None
            
        features_scaled = self.preprocess_features(features)
        
        # 使用PyOD的AutoEncoder
        if PYOD_AVAILABLE:
            model = AutoEncoder(
                hidden_neurons=[64, 32, 16, 32, 64],
                contamination=0.1,
                random_state=42
            )
            
            model.fit(features_scaled)
            scores = model.decision_function(features_scaled)
            labels = model.predict(features_scaled)
            
            self.models[method_name] = model
            self.scores[method_name] = scores
            
            return scores, labels
        else:
            print("PyOD not available for AutoEncoder")
            return None
    
    def train_and_detect_vae(self, features, method_name='vae'):
        """使用变分自编码器进行异常检测"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available")
            return None
            
        features_scaled = self.preprocess_features(features)
        
        # 使用PyOD的VAE
        if PYOD_AVAILABLE:
            model = PyOD_VAE(
                encoder_neurons=[64, 32, 16],
                decoder_neurons=[16, 32, 64],
                contamination=0.1,
                random_state=42
            )
            
            model.fit(features_scaled)
            scores = model.decision_function(features_scaled)
            labels = model.predict(features_scaled)
            
            self.models[method_name] = model
            self.scores[method_name] = scores
            
            return scores, labels
        else:
            print("PyOD not available for VAE")
            return None
    
    def get_all_methods_results(self, features):
        """运行所有可用的异常检测方法"""
        results = {}
        
        # PyOD方法
        pyod_methods = ['iforest', 'lof', 'cblof', 'knn', 'ocsvm', 'copod', 'ecod', 'rod', 'sos']
        for method in pyod_methods:
            try:
                scores, labels = self.train_and_detect_pyod(features, method)
                if scores is not None:
                    results[method] = {
                        'scores': scores,
                        'labels': labels,
                        'threshold': np.percentile(scores, 90)  # 使用90%分位数作为阈值
                    }
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        # Scikit-learn方法
        sklearn_methods = ['isolation_forest', 'local_outlier_factor', 'one_class_svm', 'elliptic_envelope']
        for method in sklearn_methods:
            try:
                scores, labels = self.train_and_detect_sklearn(features, method)
                if scores is not None:
                    results[method] = {
                        'scores': scores,
                        'labels': labels,
                        'threshold': np.percentile(scores, 90)
                    }
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        # 深度学习方法
        deep_methods = ['autoencoder', 'vae']
        for method in deep_methods:
            try:
                if method == 'autoencoder':
                    scores, labels = self.train_and_detect_autoencoder(features, method)
                elif method == 'vae':
                    scores, labels = self.train_and_detect_vae(features, method)
                
                if scores is not None:
                    results[method] = {
                        'scores': scores,
                        'labels': labels,
                        'threshold': np.percentile(scores, 90)
                    }
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        return results
    
    def save_results(self, results, output_path):
        """保存结果"""
        with open(output_path, 'wb') as f:
            pkl.dump(results, f)
        print(f"Results saved to {output_path}")
    
    def load_results(self, input_path):
        """加载结果"""
        with open(input_path, 'rb') as f:
            results = pkl.load(f)
        return results

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 初始化异常检测器
    detector = AnomalyDetectionMethods(feature_dim=64)
    
    # 加载FASTtext特征
    features_path = './dataset/optc_day23-flow/train_features.pkl'
    features = detector.load_fasttext_features(features_path)
    
    print(f"Loaded features shape: {features.shape}")
    
    # 运行所有方法
    results = detector.get_all_methods_results(features)
    
    # 保存结果
    detector.save_results(results, './anomaly_detection_results.pkl')
    
    # 打印结果摘要
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print(f"  Scores range: {result['scores'].min():.4f} - {result['scores'].max():.4f}")
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  Anomalies detected: {np.sum(result['labels'] == 1)}")
        print(f"  Normal samples: {np.sum(result['labels'] == 0)}") 