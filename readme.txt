当前运行方式与逐文件说明请先阅读 README.md 和 docs/project_files.md。
以下是历史环境安装笔记；末尾旧版处理命令已不适用于当前代码。

历史记录：conda remove -n giakt-main --all
1.创建conda虚拟环境
conda create -n giakt-main python=3.9 -y	
conda activate giakt-main

2.安装Pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
3.pip安装相关依赖
//pytorch内置了numpy2.0.2，但不支持scikit-learn,scipy,matplotlib
pip install numpy==1.26.4 --force-reinstall
pip install pandas matplotlib scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyyaml tqdm

4.测试环境
import torch
import numpy
import scipy
import sklearn
print(numpy.__version__)
print(scipy.__version__)
print(sklearn.__version__)
print(torch.cuda.is_available())
print(torch.__version__)

5.数据集预处理，以assistment09为例
(1)字段清洗:仅保留我们所需要的字段，
python scripts/assistment09/cleaned.py
(2)排序，映射，计算全局统计信息
 python scripts/assistment09/preprocessing.py            












