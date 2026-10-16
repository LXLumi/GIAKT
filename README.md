# DiffuQKT 本地复现

本项目提供基础 DiffuQKT 的数据处理、训练和评估。题目表示为 `Q = C + sigmoid(D) * V`，C、D、V 均随机初始化并参与学习；完整保留条件扩散、AKT 知识检索和对比学习。

## 运行

在项目根目录、已激活的 `giakt-main` 环境中运行：

```powershell
# 默认 ASSIST09，最多 100 轮，自动选择设备和保存目录
python main.py

# ASSIST17，最多 50 轮
python main.py assistment2017 --epochs 50
```

默认设备为 auto，优先使用可用的 CUDA。训练结果自动保存在 `experiments/<数据集名称>_<最大轮数>epochs_<编号>/`，例如 `experiments/assistment2017_50epochs_1/`；相同组合再次运行时编号递增。实际完成轮数受早停控制，记录在 summary.json 中。

每次训练完成自动更新 `experiments/results.csv`。每次实验占 train、valid、test 三行，记录最佳验证轮次对应的指标；train 的 loss 为 `(KT, LC, Diff)` 三元组，其余两行为预测 loss。名称和总耗时只写在首行。计时从训练数据加载开始，包含模型迁移到设备、训练/验证与最终测试，不含静态预处理；旧实验没有完整耗时记录时留空。可用 `python -m src.training.results` 手动汇总已有实验。

**所有训练、评估、预处理、诊断与检查命令统一记录在 [src/training/commands.md](src/training/commands.md)。** 命令行参数解析集中在 src/training/cli.py，支持原有 `--config` 写法及可选参数覆盖。

当前配置每次启动训练前重建预处理数据与学生划分；同一次训练的各轮使用同一份数据。固定数据与种子会得到相同划分。每轮评估验证集并选取最佳权重，最后使用该权重评估测试集，同时输出验证与测试结果。验证/测试不计算训练辅助损失，因此这两项报告为 null。

## 文件职责

- `scripts/preprocessing.py`：共用静态索引、学生划分、序列打包和预处理分发。
- `scripts/assistment09/`、`scripts/assistment17/`：数据集专属清洗和格式适配。
- `src/data/interactions.py`：训练时的 NPZ 读取、掩码生成与批次拼接。
- `src/training/`：训练与评估控制、命令行解析、统一命令说明。
- `src/utils/`：配置、设备、指标与文件校验工具。
