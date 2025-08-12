# Net Detector

检测网络数据异常。

## 实现 (Implement)

以下是运行各个脚本的步骤和说明：

* **训练 FastText 模型**
    ```bash
    python train_fasttext.py --dataset mydata-flow
    ```

* **数据预处理**
    ```bash
    python preprocess.py --dataset mydata-flow
    ```

* **训练 VAE 模型**
    ```bash
    python train_vae.py --dataset mydata-flow
    ```

* **性能评估**
    在运行以下脚本之前，请注意：`train_vae.py` 脚本会输出一个阈值。你需要用这个值来修改 `eval.py` 文件中的相应参数。
    
    ```bash
    python eval.py --dataset mydata-flow
    python eval_performance.py
    ```

* **阈值函数改进**
    ```bash
    python threshold_comparison.py --dataset mydata-flow
    ```

## 环境要求 (Prerequisites)

为了成功运行本项目的脚本，请按照以下步骤配置你的 Python 环境。

1.  **创建 Conda 环境**
    使用 Conda 创建一个名为 `provenance` 的新环境，并指定 Python 版本为 3.12。
    
    ```bash
    conda create --name provenance python=3.12
    ```

2.  **安装依赖包**
    在激活上述 Conda 环境后，使用 pip 安装所有必需的库。
    
    ```bash
    pip install pandas
    pip install tqdm
    pip install git+[https://github.com/casics/nostril.git](https://github.com/casics/nostril.git)
    pip install gensim
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    pip install matplotlib
    pip install seaborn
    pip install streamz
    pip install schedule
    pip install nearpy
    pip install pydot
    pip install graphviz
    ```
