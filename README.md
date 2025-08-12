# Net Detector

## Implement
```
Detect net data anomalies. 
* test on yourdataset
python train_fasttext.py --dataset  mydata-flow
python preprocess.py --dataset mydata-flow
python train_vae.py --dataset  mydata-flow
这里输出阈值，修改eval.py
python eval.py --dataset  mydata-flow
python eval_performance.py

阈值函数改进
python threshold_comparison.py --dataset mydata-flow
```

```

## Prerequisites

```
conda create --name provenance python=3.12
pip install pandas
pip install tqdm
pip install git+https://github.com/casics/nostril.git
pip install gensim
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
pip install seaborn
pip install streamz
pip install schedule
pip install nearpy
pip install pydot
pip install graphviz
...
```
