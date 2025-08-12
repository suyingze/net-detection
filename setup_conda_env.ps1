# Windows PowerShell script for setting up conda environment

Write-Host "=== Setting up Conda Environment for Anomaly Detection ===" -ForegroundColor Green

# 设置环境名称
$ENV_NAME = "anomaly_detection"

Write-Host "Creating conda environment: $ENV_NAME" -ForegroundColor Yellow

# 创建新的conda环境
conda create -n $ENV_NAME python=3.9 -y

# 激活环境
Write-Host "Activating environment..." -ForegroundColor Yellow
conda activate $ENV_NAME

# 安装基础依赖
Write-Host "Installing base dependencies..." -ForegroundColor Yellow
conda install -c conda-forge numpy pandas matplotlib scikit-learn -y

# 安装PyTorch（CPU版本，避免CUDA问题）
Write-Host "Installing PyTorch (CPU version)..." -ForegroundColor Yellow
conda install pytorch cpuonly -c pytorch -y

# 安装其他依赖
Write-Host "Installing additional dependencies..." -ForegroundColor Yellow
pip install pyod joblib gensim

# 验证安装
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "
try:
    import numpy as np
    import torch
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn
    import pyod
    import joblib
    import gensim
    print('✓ All dependencies installed successfully!')
    print(f'  NumPy: {np.__version__}')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Pandas: {pd.__version__}')
    print(f'  scikit-learn: {sklearn.__version__}')
    print(f'  PyOD: {pyod.__version__}')
except ImportError as e:
    print(f'✗ Import error: {e}')
"

Write-Host ""
Write-Host "=== Environment Setup Complete ===" -ForegroundColor Green
Write-Host "To activate the environment:" -ForegroundColor Cyan
Write-Host "  conda activate $ENV_NAME" -ForegroundColor White
Write-Host ""
Write-Host "To test the anomaly detection:" -ForegroundColor Cyan
Write-Host "  python quick_test.py" -ForegroundColor White
Write-Host ""
Write-Host "To run performance tests:" -ForegroundColor Cyan
Write-Host "  python anomaly_detection_performance_test.py" -ForegroundColor White 