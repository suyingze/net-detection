#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成多种异常检测方法，替换原有VAE
适配现有代码结构
"""

import numpy as np
import pickle as pkl
import json
import time
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========== 异常检测库导入 ==========
try:
    import pyod
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.cblof import CBLOF
    from pyod.models.knn import KNN
    from pyod.models.ocsvm import OCSVM
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE as PyOD_VAE
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    PYOD_AVAILABLE = True
    print(f"PyOD {pyod.__version__} is available")
except ImportError as e:
    print(f"PyOD not available: {e}")
    PYOD_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Scikit-learn not available")
    SKLEARN_AVAILABLE = False

class IntegratedAnomalyDetector:
    """集成异常检测器，替换原有VAE"""
    
    def __init__(self, method='iforest', contamination=0.1, **kwargs):
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # AutoEncoder特定参数
        self.autoencoder_params = {
            'hidden_neuron_list': kwargs.get('hidden_neuron_list', [64, 32, 16, 32, 64]),
            'epoch_num': kwargs.get('epoch_num', 10),
            'batch_size': kwargs.get('batch_size', 32),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'dropout_rate': kwargs.get('dropout_rate', 0.2)
        }
        
        # IForest特定参数
        self.iforest_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_samples': kwargs.get('max_samples', 'auto'),
            'max_features': kwargs.get('max_features', 0.4),
            'bootstrap': kwargs.get('bootstrap', False),
            'random_state': kwargs.get('random_state', 42)
        }
        
    def fit(self, features):
        """训练异常检测模型"""
        print(f"Training {self.method} anomaly detector...")
        
        # 预处理特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 选择并训练模型
        if self.method == 'iforest':
            if PYOD_AVAILABLE:
                self.model = IForest(
                    contamination=self.contamination,
                    n_estimators=self.iforest_params['n_estimators'],
                    max_samples=self.iforest_params['max_samples'],
                    max_features=self.iforest_params['max_features'],
                    bootstrap=self.iforest_params['bootstrap'],
                    random_state=self.iforest_params['random_state']
                )
            elif SKLEARN_AVAILABLE:
                self.model = IsolationForest(
                    contamination=self.contamination,
                    n_estimators=self.iforest_params['n_estimators'],
                    max_samples=self.iforest_params['max_samples'],
                    max_features=self.iforest_params['max_features'],
                    bootstrap=self.iforest_params['bootstrap'],
                    random_state=self.iforest_params['random_state']
                )
            else:
                raise ImportError("Neither PyOD nor scikit-learn available")
                
        elif self.method == 'lof':
            if PYOD_AVAILABLE:
                self.model = LOF(contamination=self.contamination)
            elif SKLEARN_AVAILABLE:
                self.model = LocalOutlierFactor(contamination=self.contamination, novelty=False)
            else:
                raise ImportError("Neither PyOD nor scikit-learn available")
                
        elif self.method == 'cblof':
            if PYOD_AVAILABLE:
                self.model = CBLOF(contamination=self.contamination, random_state=42)
            else:
                raise ImportError("PyOD not available for CBLOF")
                
        elif self.method == 'knn':
            if PYOD_AVAILABLE:
                self.model = KNN(contamination=self.contamination)
            else:
                raise ImportError("PyOD not available for KNN")
                
        elif self.method == 'ocsvm':
            if PYOD_AVAILABLE:
                self.model = OCSVM(contamination=self.contamination)
            elif SKLEARN_AVAILABLE:
                self.model = OneClassSVM(nu=self.contamination)
            else:
                raise ImportError("Neither PyOD nor scikit-learn available")
                
        elif self.method == 'copod':
            if PYOD_AVAILABLE:
                self.model = COPOD(contamination=self.contamination)
            else:
                raise ImportError("PyOD not available for COPOD")
                
        elif self.method == 'ecod':
            if PYOD_AVAILABLE:
                self.model = ECOD(contamination=self.contamination)
            else:
                raise ImportError("PyOD not available for ECOD")
                
        elif self.method == 'autoencoder':
            if PYOD_AVAILABLE:
                self.model = AutoEncoder(
                    hidden_neuron_list=self.autoencoder_params['hidden_neuron_list'],
                    contamination=self.contamination,
                    random_state=42,
                    epoch_num=self.autoencoder_params['epoch_num'],
                    batch_size=self.autoencoder_params['batch_size'],
                    verbose=0
                )
            else:
                raise ImportError("PyOD not available for AutoEncoder")
                
        elif self.method == 'vae':
            if PYOD_AVAILABLE:
                self.model = PyOD_VAE(
                    encoder_neurons=[64, 32, 16],
                    decoder_neurons=[16, 32, 64],
                    contamination=self.contamination,
                    random_state=42
                )
            else:
                raise ImportError("PyOD not available for VAE")
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # 训练模型
        self.model.fit(features_scaled)
        self.is_fitted = True
        print(f"{self.method} training completed.")
        
    def predict(self, features):
        """预测异常分数（类似原VAE的get_MSE）"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # 预处理特征
        features_scaled = self.scaler.transform(features)
        
        # 获取异常分数
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(features_scaled)
        elif hasattr(self.model, 'negative_outlier_factor_'):
            # 对于LocalOutlierFactor
            scores = -self.model.negative_outlier_factor_
        else:
            raise AttributeError("Model does not have decision_function method")
        
        # 确保返回的是正数（异常分数越高表示越异常）
        if self.method in ['lof', 'local_outlier_factor']:
            scores = -scores  # 反转分数，使异常分数为正
        
        return scores
    
    def save_model(self, path):
        """保存模型"""
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'method': self.method,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        import joblib
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.method = model_data['method']
        self.contamination = model_data['contamination']
        self.is_fitted = model_data['is_fitted']
        print(f"Model loaded from {path}")

# ========== 替换原有VAE的接口函数 ==========
def get_anomaly_scores(model, features, device=None):
    """
    替换原有的get_MSE函数
    返回异常检测分数
    """
    # 检查输入类型
    if isinstance(features, list) and all(isinstance(arr, np.ndarray) for arr in features):
        features = np.stack(features)
    
    # 转换为numpy数组
    features = np.asarray(features)
    
    # 获取异常分数
    scores = model.predict(features)
    
    return scores.tolist()

# ========== 使用示例 ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Anomaly Detection')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    parser.add_argument("--method", type=str, default="iforest", 
                       choices=['iforest', 'lof', 'cblof', 'knn', 'ocsvm', 'copod', 'ecod', 'autoencoder', 'vae'])
    parser.add_argument("--contamination", type=float, default=0.1)
    args = parser.parse_args()
    
    dataset = args.dataset
    dataset_path = f'./dataset/{dataset}/'
    
    # 加载训练特征
    print("Loading training features...")
    with open(f"{dataset_path}train_features.pkl", "rb") as f:
        train_features = pkl.load(f)
    
    # 初始化异常检测器
    detector = IntegratedAnomalyDetector(method=args.method, contamination=args.contamination)
    
    # 训练模型
    detector.fit(train_features)
    
    # 保存模型
    model_path = f'./models/{args.method}_anomaly_detector.pkl'
    detector.save_model(model_path)
    
    # 测试预测
    print("Testing prediction...")
    scores = detector.predict(train_features[:100])  # 测试前100个样本
    print(f"Predicted scores range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Mean score: {np.mean(scores):.4f}")
    
    print(f"\nModel trained and saved successfully!")
    print(f"Method: {args.method}")
    print(f"Contamination: {args.contamination}")
    print(f"Model path: {model_path}") 