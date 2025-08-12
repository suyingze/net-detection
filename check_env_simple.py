#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的环境检查脚本
避免torch CUDA加载问题
"""

import sys

def check_simple_deps():
    """检查基础依赖，避免torch CUDA问题"""
    print("=== Simple Environment Check ===")
    print()
    
    deps_to_check = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('sklearn', 'sklearn'),
        ('joblib', 'joblib'),
        ('gensim', 'gensim'),
    ]
    
    available_deps = []
    
    for package, import_name in deps_to_check:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package} (version: {version})")
            available_deps.append(package)
        except ImportError:
            print(f"✗ {package} - NOT AVAILABLE")
        except Exception as e:
            print(f"⚠ {package} - Error: {e}")
    
    print()
    
    # 检查PyOD（主要的新依赖）
    try:
        import pyod
        print(f"✓ PyOD (version: {pyod.__version__})")
        available_deps.append('PyOD')
        
        # 检查PyOD模块
        pyod_modules = [
            'pyod.models.iforest',
            'pyod.models.copod',
            'pyod.models.lof'
        ]
        
        print("  PyOD modules:")
        for module in pyod_modules:
            try:
                __import__(module)
                print(f"    ✓ {module}")
            except ImportError:
                print(f"    ✗ {module}")
                
    except ImportError:
        print("✗ PyOD - NOT AVAILABLE")
    except Exception as e:
        print(f"⚠ PyOD - Error: {e}")
    
    print()
    
    # 检查torch（避免CUDA问题）
    try:
        import torch
        print(f"✓ PyTorch (version: {torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        available_deps.append('PyTorch')
    except ImportError:
        print("✗ PyTorch - NOT AVAILABLE")
    except Exception as e:
        print(f"⚠ PyTorch - Error: {e}")
    
    print()
    
    # 总结
    print("=== Summary ===")
    if len(available_deps) >= 6:  # 至少需要numpy, pandas, matplotlib, sklearn, joblib, PyOD
        print("✓ Environment is ready for anomaly detection")
        print("  You can run: python quick_test.py")
    else:
        print("⚠ Some dependencies are missing")
        print("  Consider creating a new conda environment")
    
    return available_deps

if __name__ == "__main__":
    check_simple_deps() 