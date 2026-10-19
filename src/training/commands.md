# DiffuQKT / LinearQ / NewQ / NewQ V2 / DHA-KT 统一运行命令

所有命令均在项目根目录、已激活的 `giakt-main` 环境中运行。命令行参数定义集中在同目录的 `cli.py`，根目录 `main.py` 是启动入口。

模型源码按 `src/model/diffuqkt/`、`src/model/linearq/`、`src/model/newq/`、`src/model/newq_v2/`、`src/model/dha_kt/` 管理，实验模型按各自边界复用基线。新增模型的复用与注册规则见 [模型目录说明](../../docs/model_architecture.md)。

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

### 模型选择

默认使用 diffuqkt；支持 `--model linearq`、`--model newq`、`--model newq_v2`，名称不区分大小写。数据集和模型分别选择：

```powershell
# LinearQ，ASSIST17 / ASSIST09 各自训练
python main.py assistment2017 --model linearq --epochs 50 --device cuda
python main.py assistment2009 --model linearq --epochs 50 --device cuda

# NewQ：仅替换初始 Q，完整保留条件扩散与原有下游
python main.py assistment2009 --model newq --epochs 50 --device cuda
python main.py assistment2017 --model newq --epochs 50 --device cuda

# NewQ V2：按专属 YAML 选择残差路由、融合方式、拼接维度与条件 detach
python main.py assistment2009 --model newq_v2 --epochs 100 --device cuda
python main.py assistment2017 --model newq_v2 --epochs 100 --device cuda
python main.py assistment2009 --model newq_v2 --smoke --device cuda

# 自动恢复 V2，数据集配置须与训练一致
python main.py assistment2009 --evaluate experiments/newq_v2/assistment2009/newq_v2_assistment2009_100epochs_1/best.pt --device cuda

# V2 保存权重的只读诊断；默认完整采样，只限制数据范围
python -m src.model.newq_v2.diagnostics --config configs/assistment2009.yaml --checkpoint experiments/newq_v2/assistment2009/newq_v2_assistment2009_100epochs_1/best.pt --device cuda --max-batches 2 --max-windows 32 --max-seq-len 32 --output experiments/newq_v2_diagnostics.json

# NewQ 小规模流程检查
python main.py assistment2009 --model newq --smoke --epochs 5 --device cuda

# 从权重自动恢复 NewQ
python main.py assistment2017 --evaluate experiments/newq/assistment2017/newq_assistment2017_50epochs_1/best.pt --device cuda

# 显式选择原始 DiffuQKT
python main.py assistment2017 --model diffuqkt --epochs 50 --device cuda

# LinearQ 小规模流程检查
python main.py assistment2017 --model linearq --smoke --device cuda

# 根据 checkpoint 自动识别模型；无需 --model，数据集必须与训练时一致
python main.py assistment2017 --evaluate experiments/linearq/assistment2017/linearq_assistment2017_50epochs_1/best.pt --device cuda
```

LinearQ 的 `Q=C+DV`、`Q*=(C+A)+D(V+B)` 中，D 沿用 sigmoid 难度标量，A、B 由题目 ID 向量经独立线性层产生；训练目标为 KT + 加权 LC + 加权的按题等权残差约束；每次更新对完整训练题目集合计算 L_item=mean_i,dim((A_i+D_i*B_i)^2)，复用 diffusion_loss / diffusion_weight 字段，保留当前配置权重 0.03。完整公式、参数和恢复规则见 [docs/linearq.md](../../docs/linearq.md)。

公共参数位于 configs/base.yaml，模型专属默认值位于 configs/models/ 下对应的 diffuqkt.yaml、linearq.yaml、newq.yaml。选择 LinearQ 不会合并扩散默认值；NewQ 默认保留与基线一致的 100 步扩散及 0.01 的扩散损失权重，新增 r=16 和概念注意力温度 1。NewQ 的公式及条件适配见 [docs/newq.md](../../docs/newq.md)。新增 checkpoint 保存模型名称；无模型标签的历史 checkpoint 按 DiffuQKT 加载，显式 --model 与保存的模型不符时拒绝评估。

NewQ 使用 --smoke 时，主干按公共规则改为 d=32，题目向量同步改为 r=4，避免题目表在小主干下超过 4 倍参数上限；正常训练默认 d=128、r=16。

NewQ V2 的专属配置为 configs/models/newq_v2.yaml，默认 d=128、r_A=r_R=8、gamma=2、tau=1；--smoke 使用 d=32、r_A=r_R=2，不改变插件模式。residual_diffusion_enabled=false 恢复 off；true / condition 使用独立 R 条件，true / target 仅生成完整目标，true / both 同时提供独立条件和完整目标。target/both 生成后不再加 R。详细切换表和诊断字段见 [docs/newq_v2.md](../../docs/newq_v2.md)。

无需新增模型名或更换命令；在该 YAML 的 model 节修改这两个字段后，仍运行 --model newq_v2，自动目录编号会创建新实验。启动日志、summary 和 checkpoint 记录有效模式；评估按保存的 model_kwargs 恢复，当前 YAML 不覆盖旧 checkpoint 的模式。

条件梯度对照使用新增 `residual_condition_detach`（bool，默认 false）。需要执行 `f(...,sg(R))+R` 时，在该文件 model 节设置下面三项，再运行相同训练命令；本次保留用户当前的 enabled=false，没有自动启用实验：

```yaml
residual_diffusion_enabled: true
residual_diffusion_role: condition
residual_condition_detach: true
```

只将条件输入 detach，输出端和原始对比分支仍用正常 R；生成器仍可使用其数值并学习条件权重。condition 下 Diff 无残差参数梯度，KT/CL 保留正常残差路径；both 下完整目标仍可向残差反传，off/target 下标志无效。请求值和实际生效状态写入 summary/checkpoint/诊断，旧权重缺字段时按 false 恢复。

Q 与 R 的融合另由下面两项控制，命令保持相同：

```yaml
residual_fusion: concat          # 默认 add：相加；concat：按 [Q, R] 拼接
residual_concat_keep_2d: true    # true/1 保留 2d；false/0 通过共享 Linear(2d,d) 映射回 d
```

第二项只在 concat 时生效，接受 true/false 或整数 1/0，保存时统一为 bool。映射回 d 的线性层初始为 [I,I]，原始/增强分支共享且参与训练；保留 2d 时自动适配本模型答案嵌入、检索和预测宽度，target/both 的完整扩散目标也为 2d。默认仍为 add/false，缺少字段的历史权重按原相加模型恢复。改变设置后重新运行 `python main.py assistment2009 --model newq_v2 --epochs 50 --device cuda` 会分配新的实验目录。

NewQ V2 的 D 条件由同一 YAML 的 `model.diffusion_difficulty_condition` 控制，默认 true。设为 false 后训练去噪及每一步逆扩散均不接收显式 D 条件；`Q_sem=C_bar+D*V_bar` 中的 D 仍正常学习，R 路由、detach 和融合设置保持各自独立。命令不变，例如 `python main.py assistment2009 --model newq_v2 --epochs 50 --device cuda`。字段保存到实际配置、checkpoint、summary 和诊断；旧权重缺字段时按 true 恢复，评估不受当前 YAML 的开关覆盖。详见 [NewQ V2 说明](../../docs/newq_v2.md)。

### DHA-KT

专属配置为 configs/models/dha_kt.yaml，当前两数据集的完整训练参考库已建立，可直接执行前两条训练命令。参考构建、运行范围、两后端互斥配置及详细验收见 [docs/dha_kt_usage.md](../../docs/dha_kt_usage.md)。

```powershell
# 正式 profile 后端；50 总轮次包含 5 轮 raw 预热、5 轮扩散预训练
python main.py assistment2009 --model dha_kt --epochs 50 --device cuda
python main.py assistment2017 --model dha_kt --epochs 50 --device cuda

# 更换划分/处理范围后重建参考库；已有目录需显式 --overwrite
python -m scripts.dha_kt.build_reference --config configs/assistment2009.yaml --output-dir data/assistment09/dha_kt/reference --overwrite
python -m scripts.dha_kt.build_reference --config configs/assistment2017.yaml --output-dir data/assistment17/dha_kt/reference --overwrite

# 独立评估恢复全部参考 buffer，不要求原 artifact 存在
python main.py assistment2009 --evaluate experiments/dha_kt/assistment2009/dha_kt_assistment2009_50epochs_1/best.pt --device cuda

# 继续至总轮数 100；从保存阶段/Adam/RNG 恢复并自动分配新目录
python main.py assistment2009 --model dha_kt --resume experiments/dha_kt/assistment2009/dha_kt_assistment2009_50epochs_1/last.pt --epochs 100 --device cuda

# 范围匹配的 joint smoke；不改现有完整参考库
python -m scripts.dha_kt.build_reference --config configs/assistment2009.yaml --output-dir data/assistment09/dha_kt/reference_smoke --max-windows 64 --max-seq-len 32
python main.py assistment2009 --model dha_kt --reference-artifact data/assistment09/dha_kt/reference_smoke --smoke --max-batches 2 --device cuda

# 保存权重的只读诊断
python -m src.model.dha_kt.diagnostics --config configs/assistment2009.yaml --checkpoint experiments/dha_kt/assistment2009/dha_kt_assistment2009_50epochs_1/best.pt --device cuda --max-batches 2 --output experiments/dha_kt_diagnostics.json
```

训练默认自动查找对应数据集目录下的 dha_kt/reference；可用 `--reference-artifact` 或 YAML 的 artifact_path 指定其他库。参考库必须匹配当前 max_windows/max_seq_len 和学生划分，缺失或校验失败时明确报错。DHA 无 CL，CSV 保留三元组 `(KT, NA, Diff)`，pilot、分歧和阶段成本另存 history/summary。默认 H=8、M=3、100 步 cosine 日程、8 点可微 DDIM，三种融合及 r→pi 对照由专属 YAML 控制。阶段结束切换选模组，不用 warmup 指标替代 joint 最佳轮次。

## 2. 自动保存目录

默认根目录为 `experiments/`。每次启动训练，自动创建：

```text
experiments/<模型目录>/<dataset.name>/<实验名>/
```

示例：

| 命令                                          | 首次对应目录                                |
| --------------------------------------------- | ------------------------------------------- |
| `python main.py`                            | `experiments/diffuqk/assistment2009/assistment2009_100epochs_1/` |
| `python main.py --epochs 50`                | `experiments/diffuqk/assistment2009/assistment2009_50epochs_1/`  |
| `python main.py assistment2017 --epochs 50` | `experiments/diffuqk/assistment2017/assistment2017_50epochs_1/`  |

相同模型、数据集与轮数再次运行时，目录编号递增为 `_2`、`_3` 等。编号取该组合当前已占用编号的最大值加 1，同时检查分组目录与尚未迁移的根目录旧实验；中断后留下的目录也占用编号。不同模型、不同数据集、不同轮数组合分别计数。创建目录时会处理同时启动多个进程的编号冲突。

基线模型的文件夹命名为 `diffuqk`，配置与 `--model` 仍使用 `diffuqkt`，单次实验沿用没有模型前缀的旧名称。LinearQ 的分组目录是 `linearq`，单次实验保留 `linearq_` 前缀，例如 `experiments/linearq/assistment2017/linearq_assistment2017_50epochs_1/`。后续注册的新模型使用模型名作为分组目录，并在单次实验名前添加同名模型前缀。

NewQ 对应 `experiments/newq/<数据集>/newq_<数据集>_<轮数>epochs_<编号>/`，例如 `experiments/newq/assistment2017/newq_assistment2017_50epochs_1/`。

NewQ V2 对应 `experiments/newq_v2/<数据集>/newq_v2_<数据集>_<轮数>epochs_<编号>/`，不复用 V1 的目录或权重结构。

根目录结构如下，所有分组共用根目录的 results.csv：

```text
experiments/
├── results.csv
├── diffuqk/
│   ├── assistment2009/
│   │   └── assistment2009_100epochs_1/
│   └── assistment2017/
│       └── assistment2017_100epochs_1/
└── linearq/
    ├── assistment2009/
    │   └── linearq_assistment2009_100epochs_1/
    └── assistment2017/
        └── linearq_assistment2017_100epochs_1/
```

已有实验只移动整个目录，实验名与文件内容保持不变。历史 config.yaml、summary.json 与 checkpoint 中的 output_dir 是当时的运行路径快照；评估时使用移动后的实际 checkpoint 路径即可，无需改写这些快照。

使用历史 config.yaml 重新训练时，可追加 `--output-root experiments`，覆盖快照里的旧 output_dir，并按当前分组与编号规则创建新实验。

名称中的轮数是本次请求的最大轮数。实际训练轮数与最佳轮次记录在 summary.json。终端启动时打印 `Run directory`，结束时打印 `Saved run`。

每个实验目录保存：

- `config.yaml`：合并后的实际运行配置，包含解析后的 output_dir。
- `history.csv`：逐轮训练与验证指标。
- `best.pt`、`last.pt`：验证最佳轮次和最后一轮的训练状态。
- `valid_metrics.json`：最佳轮次的验证指标。
- `test_metrics.json`、`test_predictions.npz`：最终测试指标与逐目标结果。
- `summary.json`：最佳轮次、完成轮数、该轮训练/验证与最终测试结果、总耗时秒数 duration_seconds、含时区的结束时间 finished_at、汇总表小数位数 results_decimal_places 及实际输出目录。

数据预处理产物继续保存到 `dataset.paths.processed`，与实验目录分开。

每轮终端日志将 train、valid 分成两行，loss、AUC、ACC 的标签与数值按列对齐，统一显示四位小数；训练辅助损失放在训练行的末尾，耗时放在验证行的末尾。验证行的 loss 为模型返回的预测损失，与汇总 CSV 的 loss 含义相同。最终指标与独立评估的终端 JSON 也只对展示副本格式化，指标值显示为四位小数字符串。未定义的 AUC 在逐轮日志中显示 NA，在终端 JSON 中保留 null。

计算和验证选模始终使用完整精度；history.csv、valid_metrics.json、test_metrics.json、summary.json 和 checkpoint 中的指标也保留原始精度。只有终端展示及下述 results.csv 使用格式化后的数值。

### 多次实验的统一结果表

每次训练正常完成后，自动更新 `experiments/results.csv`，递归纳入各模型、数据集分组与旧平铺目录下已有完整 summary.json 的实验。每次实验占三行：

| experiment | split | AUC | ACC | loss | 备注 |
| ---------- | ----- | --- | --- | ---- | ---- |
| assistment2017_100epochs_1 | train | 0.8100 | 0.7300 | (0.5000, 2.0000, 0.0800) | |
| 2026-10-18 15:30:45 | valid | 0.7800 | 0.7100 | 0.5500 | |
| 61m5s | test | 0.7700 | 0.7000 | 0.5600 | |

以上数值仅为格式示例。experiment 的三行依次为实际实验文件夹名称、该次训练的结束时间和总耗时。结束时间以训练机器记录的本地时间显示，格式为 `YYYY-MM-DD HH:MM:SS`，在最终测试结束后采集并以含时区的 ISO 时间写入 summary.json 的 finished_at。旧实验没有 finished_at 时，使用 summary.json 文件的修改时间近似；复制或修改过该文件时，该近似值可能与实际结束时间不同。

“备注”列的三行默认留空，可在训练结束后手动填写超参数变化等内容。保存为 UTF-8 CSV 后，下次训练结束或运行手动汇总命令时，会按实验目录名和 split 保留已有备注，包括逗号、引号和单元格内换行。编辑时保持 experiment 与 split 的分组结构。旧版 time 列的耗时转移到 experiment 的第三行，不作为备注保留。

- train 的 loss 按 `(KT loss, LC loss, Diff loss)` 存储三个未加权分量。LC 对应代码中的 contrastive_loss，Diff 对应 diffusion_loss；联合目标 train_loss 继续保留在 history.csv 和 summary.json，不作为这个三元组的第一项。
- LinearQ 的 train 行保留相同三元组位置，显示 `(KT loss, LC loss, L_item)`，第三项按训练题目去重、等权，日志按步骤平均。训练前自动提取训练题目集合并存入 checkpoint，当前权重保留为 0.03。
- valid、test 的 loss 是各自的预测损失，不填训练辅助损失。AUC、ACC 都使用 0～1 的数值；后续新训练的 AUC、ACC、loss 与训练损失三元组均显示四位小数，未定义的 AUC 留空，缺失的训练损失分量以 NA 表示。
- 新训练的 summary.json 记录 results_decimal_places=4，手动重建时也沿用四位格式；旧实验没有该字段，继续使用原来的八位格式，不批量改写已有记录。原始完整精度指标保留在逐轮记录、JSON 和权重中。
- 三行统一对应验证指标选出的 best_epoch：train 是该轮训练过程中记录的指标，valid 是该轮验证指标，test 是加载 best.pt 后的最终测试指标。
- 总耗时从预处理完成后、开始加载训练 NPZ 数据之前计时，包括数据读取、模型初始化与加载到所选设备、各轮训练/验证、权重保存和最终测试。静态预处理不计入。格式为 `ambs`，a 为总分钟数，b 为余下秒数（0～59）；例如 3665 秒写作 `61m5s`，超过一小时也继续累计分钟数，秒数四舍五入。
- 旧 summary.json 没有 duration_seconds 时仍汇总其指标，experiment 第三行留空；逐轮时间不用于推算完整耗时。没有 summary.json 的未完成实验跳过。
- 汇总表每次从完整记录重建指标，并保留当前表中已有实验的备注，同一实验不会重复追加；当前 CSV 中已有完整三行、但实验目录已经移除的历史记录继续保留。损坏的 summary.json 仍会告警并跳过。文件以 UTF-8 BOM 编码保存，便于 Windows Excel 打开。
- 自定义 --output-root 时，模型/数据集分组与 results.csv 都位于所选根目录。精确指定 --output-dir 时，如果该目录位于配置的 output_root 内，继续更新根目录的汇总表；在该根目录外时，沿用写入实验父目录的行为。实验名保留模型和数据集前缀，移动时保持名称不变，以便对应已有备注。

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

# 改变模型/数据集分组及统一汇总表的根目录
# 结果示例：experiments/comparison/diffuqk/assistment2017/assistment2017_50epochs_1/
python main.py assistment2017 --epochs 50 --output-root experiments/comparison

# 兼容旧写法：精确指定某次实验目录
python main.py --config configs/assistment2009.yaml --epochs 50 --device cuda --output-dir experiments/manual_assist09

# 允许覆盖上述手动指定目录中的训练记录
python main.py assistment2009 --epochs 50 --output-dir experiments/manual_assist09 --overwrite
```

位置参数与 `--config` 二选一；`--output-root` 与 `--output-dir` 二选一。自动编号模式始终创建新目录，正常运行无需指定输出位置或使用 `--overwrite`。

参数覆盖顺序为：命令行 > 数据集或自定义配置 > 所选模型专属配置 > base.yaml。自定义配置中的 `train.output_dir` 仍作为精确输出路径使用；采用自动编号时使用 `train.output_root`，默认 experiments。

## 4. 小规模流程检查

```powershell
# 两轮小规模检查：d=32、batch=8，每个集合最多 64 个窗口、序列最多 32 次交互
python main.py --smoke

# 在 smoke 设置上明确改为一轮，并要求 CUDA
python main.py assistment2017 --smoke --epochs 1 --device cuda

# 保留正式模型尺寸，限制数据量与批次数
python main.py assistment2017 --epochs 1 --max-windows 32 --max-seq-len 32 --max-batches 2
```

显式命令行参数覆盖 smoke 默认值。DiffuQKT 的 smoke 仍使用配置中的扩散步数，默认 100；LinearQ 不使用扩散。小规模结果也遵循同一命名规则，并在 summary.json 中记录 limited_run=true。

## 5. 单独评估最佳权重

```powershell
# 评估 ASSIST17 的测试集；默认把结果写入所选 checkpoint 所在目录
python main.py assistment2017 --evaluate experiments/diffuqk/assistment2017/assistment2017_50epochs_1/best.pt

# 评估验证集
python main.py assistment2017 --evaluate experiments/diffuqk/assistment2017/assistment2017_50epochs_1/best.pt --split valid

# 评估训练集，并明确要求 CUDA
python main.py assistment2009 --evaluate experiments/diffuqk/assistment2009/assistment2009_50epochs_1/best.pt --split train --device cuda

# 可选：把评估结果写入指定目录
python main.py assistment2017 --evaluate experiments/diffuqk/assistment2017/assistment2017_50epochs_1/best.pt --output-dir experiments/evaluation_assist17
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
python scripts/diagnose_losses.py --checkpoint experiments/diffuqk/assistment2017/assistment2017_50epochs_1/best.pt --device cuda --output-dir experiments/gradient_audit_assist17

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
| `--model`             | 默认 diffuqkt，可选 linearq；评估时默认自动识别 checkpoint 模型 |
| `--epochs`            | 最大轮数，公共配置默认 100                                 |
| `--batch-size`        | 每批序列窗口数，公共配置默认 64                            |
| `--seed`              | 训练与预处理种子，默认 42                                  |
| `--device`            | 默认 auto，可设 cpu、cuda、cuda:0                          |
| `--output-root`       | 模型/数据集分组与统一汇总表的根目录，默认 experiments       |
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
