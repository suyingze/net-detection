#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖检查脚本
验证现有环境与新异常检测库的兼容性
"""

import sys
import importlib

def check_dependency(package_name, import_name=None):
    """检查依赖包是否可用"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name} (version: {version})")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT AVAILABLE")
        return False

def main():
    print("=== Dependency Compatibility Check ===")
    print()
    
    # 检查现有依赖
    print("Existing Dependencies:")
    existing_deps = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('gensim', 'gensim'),
        ('sklearn', 'sklearn'),
    ]
    
    existing_available = []
    for package, import_name in existing_deps:
        if check_dependency(package, import_name):
            existing_available.append(package)
    
    print()
    
    # 检查新依赖
    print("New Dependencies:")
    new_deps = [
        ('scikit-learn', 'sklearn'),
        ('joblib', 'joblib'),
        ('PyOD', 'pyod'),
    ]
    
    new_available = []
    for package, import_name in new_deps:
        if check_dependency(package, import_name):
            new_available.append(package)
    
    print()
    
    # 检查PyOD的具体模块
    if 'PyOD' in new_available:
        print("PyOD Modules Check:")
        pyod_modules = [
            'pyod.models.iforest',
            'pyod.models.copod', 
            'pyod.models.lof',
            'pyod.models.auto_encoder',
            'pyod.models.vae'
        ]
        
        for module in pyod_modules:
            try:
                importlib.import_module(module)
                print(f"  ✓ {module}")
            except ImportError:
                print(f"  ✗ {module}")
    
    print()
    
    # 兼容性评估
    print("=== Compatibility Assessment ===")
    
    if len(existing_available) >= 5:  # 至少需要numpy, torch, pandas, matplotlib, gensim
        print("✓ Existing environment is well-equipped")
    else:
        print("⚠ Some existing dependencies are missing")
    
    if 'PyOD' in new_available:
        print("✓ PyOD is available for new anomaly detection methods")
    else:
        print("✗ PyOD is required but not available")
    
    if 'scikit-learn' in new_available:
        print("✓ scikit-learn is available for enhanced functionality")
    else:
        print("⚠ scikit-learn is recommended but not required")
    
    print()
    
    # 建议
    print("=== Recommendations ===")
    
    if 'PyOD' not in new_available:
        print("1. Install PyOD: pip install pyod")
    
    if 'scikit-learn' not in new_available:
        print("2. Install scikit-learn: pip install scikit-learn")
    
    if 'joblib' not in new_available:
        print("3. Install joblib: pip install joblib")
    
    if len(existing_available) >= 5 and 'PyOD' in new_available:
        print("✓ Environment is ready for anomaly detection testing")
        print("  Run: python quick_test.py")
    else:
        print("⚠ Please install missing dependencies before testing")

if __name__ == "__main__":
    main() 