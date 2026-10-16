# DiffuQKT 统一运行命令

所有命令均在项目根目录、已激活的 `giakt-main` 环境中运行。命令行参数定义集中在同目录的 `cli.py`，根目录 `main.py` 是启动入口。

## 1. 默认运行与常用训练命令

```powershell
# 默认：ASSIST09，最多 100 轮，自动选择设备，自动创建实验目录
python main.py

# 默认数据集 ASSIST09，最多 50 轮
python main.py --epochs 50

# 显式指定 ASSIST09
python main.py assistment2009 --epochs 50

# ASSIST17，最多 50 轮
python main.py assistment2017 --epochs 50

# 明确要求 CUDA；不可用时报告错误
python main.py assistment2017 --epochs 50 --device cuda

# 使用 CPU
python main.py assistment2009 --epochs 50 --device cpu

# 指定训练/划分种子和批大小
python main.py assistment2017 --epochs 50 --seed 43 --batch-size 32

# 查看全部选项
python main.py --help
```

数据集默认映射为 `configs/assistment2009.yaml`；`assistment2017` 映射为 `configs/assistment2017.yaml`。也接受简写 `assistment09`、`assistment17`。此前出现的拼写 `assistmnet2017` 兼容映射到 ASSIST17；建议使用规范名称。

默认最大轮数为 base.yaml 中的 100，批大小为 64，设备为 auto（优先 CUDA）。`--epochs` 表示最大轮数，现有早停机制仍生效：patience 当前为 10，因此实际完成轮数可能更少。需要固定完成 50 轮时，可将配置中的 patience 调整为至少 50。

## 2. 自动保存目录

默认根目录为 `experiments/`。每次启动训练，自动创建：

```text
experiments/<dataset.name>_<最大轮数>epochs_<编号>/
```

示例：

| 命令                                          | 首次对应目录                                |
| --------------------------------------------- | ------------------------------------------- |
| `python main.py`                            | `experiments/assistment2009_100epochs_1/` |
| `python main.py --epochs 50`                | `experiments/assistment2009_50epochs_1/`  |
| `python main.py assistment2017 --epochs 50` | `experiments/assistment2017_50epochs_1/`  |

相同数据集与轮数再次运行时，目录编号递增为 `_2`、`_3` 等。编号取该组合当前已占用编号的最大值加 1；中断后留下的目录也占用编号。不同数据集、不同轮数组合分别计数。创建目录时会处理同时启动多个进程的编号冲突。

名称中的轮数是本次请求的最大轮数。实际训练轮数与最佳轮次记录在 summary.json。终端启动时打印 `Run directory`，结束时打印 `Saved run`。

每个实验目录保存：

- `config.yaml`：合并后的实际运行配置，包含解析后的 output_dir。
- `history.csv`：逐轮训练与验证指标。
- `best.pt`、`last.pt`：验证最佳轮次和最后一轮的训练状态。
- `valid_metrics.json`：最佳轮次的验证指标。
- `test_metrics.json`、`test_predictions.npz`：最终测试指标与逐目标结果。
- `summary.json`：最佳轮次、完成轮数、该轮训练/验证与最终测试结果、总耗时秒数 duration_seconds 及实际输出目录。

数据预处理产物继续保存到 `dataset.paths.processed`，与实验目录分开。

### 多次实验的统一结果表

每次训练正常完成后，自动更新 `experiments/results.csv`，同时纳入该目录下已有完整 summary.json 的实验。每次实验占三行：

| experiment | split | AUC | ACC | loss | time |
| ---------- | ----- | --- | --- | ---- | ---- |
| assistment2017_100epochs_1 | train | 0.81000000 | 0.73000000 | (0.50000000, 2.00000000, 0.08000000) | 1h65s |
| | valid | 0.78000000 | 0.71000000 | 0.55000000 | |
| | test | 0.77000000 | 0.70000000 | 0.56000000 | |

以上数值仅为格式示例。第一列的实验名称取实际文件夹名称，作为该组三行的标识；time 也只填写在首行。CSV 不支持真正合并单元格，因此其余两行的名称和时间留空。

- train 的 loss 按 `(KT loss, LC loss, Diff loss)` 存储三个未加权分量。LC 对应代码中的 contrastive_loss，Diff 对应 diffusion_loss；联合目标 train_loss 继续保留在 history.csv 和 summary.json，不作为这个三元组的第一项。
- valid、test 的 loss 是各自的预测损失，不填训练辅助损失。AUC、ACC 都使用 0～1 的数值，显示八位小数；未定义的 AUC 留空，缺失的训练损失分量以 NA 表示。
- 三行统一对应验证指标选出的 best_epoch：train 是该轮训练过程中记录的指标，valid 是该轮验证指标，test 是加载 best.pt 后的最终测试指标。
- 总耗时从预处理完成后、开始加载训练 NPZ 数据之前计时，包括数据读取、模型初始化与加载到所选设备、各轮训练/验证、权重保存和最终测试。静态预处理不计入。格式为小时与余下秒数，例如 3665 秒写作 `1h65s`，不另设分钟字段，秒数四舍五入。
- 旧 summary.json 没有 duration_seconds 时仍汇总其指标，time 留空；逐轮时间不用于推算完整耗时。没有 summary.json 的未完成实验跳过。
- 汇总表每次从完整记录重建，同一实验不会重复追加；文件以 UTF-8 BOM 编码保存，便于 Windows Excel 打开。自定义 --output-root 时，results.csv 位于所选父目录；手动 --output-dir 则位于该实验文件夹的父目录。

手动汇总已有结果，无需重新训练：

```powershell
python -m src.training.results

# 指定其他实验根目录
python -m src.training.results --root experiments/comparison
```

本功能更新前已启动的训练进程，需要在完成后执行上述手动命令。如果 CSV 正在被 Excel 占用而无法更新，先关闭文件再执行此命令；该实验的 summary.json 和权重仍会正常保存。

## 3. 可选配置与输出位置

```powershell
# 使用其他 YAML 配置文件；也支持把 YAML 路径作为位置参数
python main.py --config configs/assistment2017.yaml --epochs 50
python main.py configs/assistment2017.yaml --epochs 50

# 改变自动编号目录的父目录
# 结果示例：experiments/comparison/assistment2017_50epochs_1/
python main.py assistment2017 --epochs 50 --output-root experiments/comparison

# 兼容旧写法：精确指定某次实验目录
python main.py --config configs/assistment2009.yaml --epochs 50 --device cuda --output-dir experiments/manual_assist09

# 允许覆盖上述手动指定目录中的训练记录
python main.py assistment2009 --epochs 50 --output-dir experiments/manual_assist09 --overwrite
```

位置参数与 `--config` 二选一；`--output-root` 与 `--output-dir` 二选一。自动编号模式始终创建新目录，正常运行无需指定输出位置或使用 `--overwrite`。

参数覆盖顺序为：命令行 > 数据集配置 > base.yaml。自定义配置中的 `train.output_dir` 仍作为精确输出路径使用；采用自动编号时使用 `train.output_root`，默认 experiments。

## 4. 小规模流程检查

```powershell
# 两轮小规模检查：d=32、batch=8，每个集合最多 64 个窗口、序列最多 32 次交互
python main.py --smoke

# 在 smoke 设置上明确改为一轮，并要求 CUDA
python main.py assistment2017 --smoke --epochs 1 --device cuda

# 保留正式模型尺寸，限制数据量与批次数
python main.py assistment2017 --epochs 1 --max-windows 32 --max-seq-len 32 --max-batches 2
```

显式命令行参数覆盖 smoke 默认值。smoke 仍使用配置中的扩散步数，默认 100。小规模结果也遵循同一命名规则，并在 summary.json 中记录 limited_run=true。

## 5. 单独评估最佳权重

```powershell
# 评估 ASSIST17 的测试集；默认把结果写入所选 checkpoint 所在目录
python main.py assistment2017 --evaluate experiments/assistment2017_50epochs_1/best.pt

# 评估验证集
python main.py assistment2017 --evaluate experiments/assistment2017_50epochs_1/best.pt --split valid

# 评估训练集，并明确要求 CUDA
python main.py assistment2009 --evaluate experiments/assistment2009_50epochs_1/best.pt --split train --device cuda

# 可选：把评估结果写入指定目录
python main.py assistment2017 --evaluate experiments/assistment2017_50epochs_1/best.pt --output-dir experiments/evaluation_assist17
```

将示例中的编号替换为实际实验编号。评估不新建编号训练目录，不触发预处理；使用原有 processed 数据，并校验 metadata 与 checkpoint 的匹配关系。`--evaluate` 不能与 `--preprocess` 或训练专用的 `--output-root` 同时使用。

## 6. 数据预处理命令

当前 `train.preprocess_each_run=true`，因此每次训练前会重新执行预处理并覆盖标准产物；同一次训练内保持划分固定。固定种子和相同输入产生相同划分。

```powershell
# 训练入口显式要求重建数据
python main.py assistment2017 --preprocess --epochs 50

# ASSIST09：只检查前 10000 行，不写数据
python scripts/assistment09/preprocess.py --inspect --rows 10000

# 单独预处理两个数据集
python scripts/assistment09/preprocess.py
python scripts/assistment17/preprocess.py

# 需要从 ASSIST17 原始数据重新生成初步清洗文件时，先执行此命令
python scripts/assistment17/cleaned.py

# 可选：调整耗时截取及短耗时策略
python scripts/assistment09/preprocess.py --max-time 300 --short-time-policy drop
python scripts/assistment17/preprocess.py --max-time 300 --short-time-policy clip
```

预处理脚本的其他可选参数：

| 参数                        | ASSIST09                | ASSIST17                |
| --------------------------- | ----------------------- | ----------------------- |
| `--config PATH`           | 指定数据集配置          | 指定数据集配置          |
| `--input PATH`            | 覆盖原始 CSV 路径       | 覆盖已清洗 CSV 路径     |
| `--output DIRECTORY`      | 覆盖 processed 输出位置 | 覆盖 processed 输出位置 |
| `--clean-output PATH`     | 覆盖清洗五元组输出位置  | 不提供该选项            |
| `--inspect --rows N`      | 仅检查前 N 行           | 不提供该选项            |
| `--no-recover-chronology` | 不提供该选项            | 显式采用清洗文件行序    |

单独指定预处理的输出目录后，需要同步更新训练配置的 dataset.paths.processed，才能让训练读取该目录。使用 ASSIST17 现有清洗文件时，可直接运行 preprocess.py。

## 7. 辅助损失诊断与回归检查

```powershell
# 在 checkpoint 副本上独立检查 CL 和 Diff 的梯度与参数更新
python scripts/diagnose_losses.py --checkpoint experiments/assistment2017_50epochs_1/best.pt --device cuda --output-dir experiments/gradient_audit_assist17

# 全部回归检查
python -B -m unittest discover -s tests -v

# 检查当前 Python 的 PyTorch 与 CUDA 状态
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

diagnose_losses.py 使用自己的 `--output-dir` 保存 gradient_audit.json，不覆盖原 checkpoint，也不参与普通训练。

## 8. 主入口参数速查

| 参数                    | 默认或作用                                                 |
| ----------------------- | ---------------------------------------------------------- |
| `dataset`（位置参数） | 默认 assistment2009；可选 assistment2017、简写或 YAML 路径 |
| `--config`            | 指定 YAML 路径，兼容原有写法                               |
| `--epochs`            | 最大轮数，公共配置默认 100                                 |
| `--batch-size`        | 每批序列窗口数，公共配置默认 64                            |
| `--seed`              | 训练与预处理种子，默认 42                                  |
| `--device`            | 默认 auto，可设 cpu、cuda、cuda:0                          |
| `--output-root`       | 自动编号实验的父目录，默认 experiments                     |
| `--output-dir`        | 可选的精确输出路径                                         |
| `--overwrite`         | 允许覆盖手动指定的实验目录                                 |
| `--preprocess`        | 显式启用训练前预处理                                       |
| `--smoke`             | 设置小规模模型与数据范围                                   |
| `--max-windows`       | 限制每个集合的序列窗口数                                   |
| `--max-seq-len`       | 限制每个窗口的交互长度                                     |
| `--max-batches`       | 限制每个集合的批次数                                       |
| `--evaluate`          | 读取指定 checkpoint 进行评估                               |
| `--split`             | 单独评估的集合，默认 test，可选 valid、train               |
| `-h / --help`         | 显示帮助                                                   |
