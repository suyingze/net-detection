# PyOD模型使用指南

## 概述

本指南介绍如何使用PyOD（Python Outlier Detection）库中的异常检测模型来替代原有的VAE模型，并使用现有的阈值比较系统进行评估。

## 环境准备

### 1. 安装依赖

```bash
# 激活conda环境
conda activate anomaly_detection

# 安装PyOD
pip install pyod

# 安装其他依赖
pip install tqdm joblib scikit-learn
```

### 2. 验证安装

```bash
python -c "import pyod; print(f'PyOD version: {pyod.__version__}')"
```

## 可用的PyOD模型

### 1. Isolation Forest (IForest)
- **原理**: 基于随机森林的异常检测
- **适用场景**: 高维数据，计算效率高
- **参数**: contamination=0.1

### 2. COPOD (Copula-based Outlier Detection)
- **原理**: 基于copula理论的异常检测
- **适用场景**: 多变量数据，无需参数调优
- **参数**: contamination=0.1

### 3. Local Outlier Factor (LOF)
- **原理**: 基于局部密度的异常检测
- **适用场景**: 局部异常检测
- **参数**: contamination=0.1

### 4. Cluster-based Local Outlier Factor (CBLOF)
- **原理**: 基于聚类的局部异常因子
- **适用场景**: 聚类数据中的异常检测
- **参数**: contamination=0.1

### 5. k-Nearest Neighbors (KNN)
- **原理**: 基于k近邻的异常检测
- **适用场景**: 基于距离的异常检测
- **参数**: contamination=0.1

### 6. One-Class SVM (OCSVM)
- **原理**: 基于支持向量机的异常检测
- **适用场景**: 非线性异常检测
- **参数**: contamination=0.1

### 7. Empirical Cumulative Distribution (ECOD)
- **原理**: 基于经验累积分布的异常检测
- **适用场景**: 统计方法，无需参数
- **参数**: contamination=0.1

### 8. AutoEncoder
- **原理**: 基于自编码器的异常检测
- **适用场景**: 深度学习异常检测
- **参数**: hidden_neurons=[64, 32, 16, 32, 64], epochs=50

### 9. Variational AutoEncoder (VAE)
- **原理**: 基于变分自编码器的异常检测
- **适用场景**: 生成式异常检测
- **参数**: encoder_neurons=[64, 32, 16], decoder_neurons=[16, 32, 64]

## 使用方法

### 1. 基本使用

```bash
# 测试IForest模型
python pyod_threshold_test.py --method iforest

# 测试COPOD模型
python pyod_threshold_test.py --method copod

# 测试LOF模型
python pyod_threshold_test.py --method lof

# 测试AutoEncoder模型
python pyod_threshold_test.py --method autoencoder
```

### 2. 自定义数据集

```bash
# 使用其他数据集
python pyod_threshold_test.py --method iforest --dataset optc_day23-flow
```

### 3. 批量测试多个模型

```bash
# 测试所有模型
for method in iforest copod lof cblof knn ocsvm ecod autoencoder vae; do
    echo "Testing $method..."
    python pyod_threshold_test.py --method $method
done
```

## 代码结构

### 1. 模型训练

```python
from integrated_anomaly_detection import IntegratedAnomalyDetector

# 创建检测器
detector = IntegratedAnomalyDetector(method='iforest', contamination=0.1)

# 训练模型
detector.fit(train_features)

# 预测异常分数
anomaly_scores = detector.predict(test_features)
```

### 2. 阈值比较

```python
from threshold_comparison import (
    improved_threshold_v2, improved_threshold_v3, improved_threshold_v4,
    improved_threshold_v5, improved_threshold_v6,
    plot_threshold_comparison_with_original_detailed
)

# 使用现有的阈值函数
improved_methods = {
    'GMM': lambda x: improved_threshold_v2(x),
    '局部异常因子': lambda x: improved_threshold_v3(x),
    'IQR': lambda x: improved_threshold_v4(x),
    'Z-score': lambda x: improved_threshold_v5(x),
    '自适应百分位数': lambda x: improved_threshold_v6(x)
}
```

### 3. 性能评估

```python
# 计算性能指标
y_true = [1 if node_id in attack_nodes else 0 for node_id in node_ids]
y_pred = [1 if score > threshold else 0 for score in anomaly_scores]

tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
```

## 输出结果

### 1. 控制台输出

```
=== PyOD IFOREST 阈值比较测试 ===
Loading FastText features...
Loaded 252 attack nodes
Train features: 10033
Test features: 1032
Node IDs: 1032

1. Training PyOD iforest model...
2. Getting anomaly scores...
   Scores range: -0.0648 - 0.1039
   Mean score: 0.0034
   Std score: 0.0436

3. Running threshold comparison...
   GMM: threshold=0.0252, F1=0.5730
   IQR: threshold=0.0351, F1=0.5730
   Z-score: threshold=0.0222, F1=0.5730

4. Generating comparison chart...
5. Model saved to ./models/iforest_pyod_model.pkl

✓ PyOD IFOREST successfully tested with threshold comparison!
```

### 2. 生成的文件

- **模型文件**: `./models/{method}_pyod_model.pkl`
- **对比图表**: `threshold_comparison.png`
- **性能报告**: 控制台输出的详细性能指标

## 模型选择建议

### 1. 快速测试
- **推荐**: IForest, COPOD
- **原因**: 训练速度快，无需调参

### 2. 高精度需求
- **推荐**: AutoEncoder, VAE
- **原因**: 深度学习模型，精度较高

### 3. 实时检测
- **推荐**: IForest, ECOD
- **原因**: 推理速度快

### 4. 多变量数据
- **推荐**: COPOD, CBLOF
- **原因**: 适合多变量异常检测

## 参数调优

### 1. contamination参数
```python
# 调整异常比例
detector = IntegratedAnomalyDetector(method='iforest', contamination=0.05)  # 5%异常
detector = IntegratedAnomalyDetector(method='iforest', contamination=0.2)   # 20%异常
```

### 2. 模型特定参数
```python
# AutoEncoder参数
detector = IntegratedAnomalyDetector(
    method='autoencoder',
    contamination=0.1,
    hidden_neurons=[128, 64, 32, 64, 128],  # 自定义网络结构
    epochs=100,  # 增加训练轮数
    batch_size=64  # 调整批次大小
)
```

## 故障排除

### 1. PyOD安装问题
```bash
# 如果安装失败，尝试
pip install --upgrade pip
pip install pyod --no-cache-dir
```

### 2. 内存不足
```python
# 减少批次大小
detector = IntegratedAnomalyDetector(
    method='autoencoder',
    batch_size=16  # 减小批次大小
)
```

### 3. 训练时间过长
```python
# 使用更快的模型
detector = IntegratedAnomalyDetector(method='iforest')  # 最快的模型
```

## 扩展功能

### 1. 添加新模型
```python
# 在integrated_anomaly_detection.py中添加新模型
elif self.method == 'new_model':
    if PYOD_AVAILABLE:
        from pyod.models.new_model import NewModel
        self.model = NewModel(contamination=self.contamination)
    else:
        raise ImportError("PyOD not available for NewModel")
```

### 2. 自定义评估指标
```python
# 添加新的评估指标
from sklearn.metrics import roc_auc_score, precision_recall_curve

auc_score = roc_auc_score(y_true, anomaly_scores)
precision_curve, recall_curve, _ = precision_recall_curve(y_true, anomaly_scores)
pr_auc = auc(recall_curve, precision_curve)
```

