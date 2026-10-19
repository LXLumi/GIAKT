# DiffuQKT、LinearQ、NewQ、NewQ V2 与 DHA-KT

本项目提供基础 DiffuQKT 的数据处理、训练和评估。题目表示为 `Q = C + sigmoid(D) * V`，C、D、V 均随机初始化并参与学习；完整保留条件扩散、AKT 知识检索和对比学习。

LinearQ 使用 `Q* = (C+A) + sigmoid(D) * (V+B)` 替代扩散增强，其中独立线性层把可学习题目向量 E_q 映射为 A、B。两种模型共用相同的知识检索、状态对比学习与预测；LinearQ 的训练目标为 KT + 加权 LC + 加权的按训练题目等权残差约束；diffusion_loss 字段记录 L_item，当前权重保留为 0.03。详细公式与设计见 [docs/linearq.md](docs/linearq.md)。

NewQ 使用题目条件权重融合概念与变化方向，构造 `Q=sum_k alpha(q,k)*(C_k+D_q*V_k)+R_q`。默认新增 16 维题目向量，残差线性映射到原有嵌入空间；以 `C_bar+R_q`、D_q 作为原维度扩散条件，完整复用基线的扩散、知识检索、对比学习和预测。损失保持 KT + 加权 LC + 加权 Diff。具体选择见 [docs/newq.md](docs/newq.md)。

NewQ V2 使用独立的 8 维注意力代码和 8 维残差代码，以余弦匹配学习概念权重。专属配置中的 residual_diffusion_enabled / residual_diffusion_role 控制 off、condition、target、both；默认相加时，condition 将 R 作为独立扩散条件，生成语义后再加 R。target 只重建完整 Q、无 R 条件；both 同时作为条件与完整目标，二者生成后不再融合 R。residual_condition_detach（构造默认 false）只停止条件输入向残差参数的梯度，正常输出/原始分支保留。实际训练设置以当前 YAML 为准；条件 detach 对照使用 true/condition，并将 detach 标志设为 true。完整语义见 [docs/newq_v2.md](docs/newq_v2.md)。

NewQ V2 的残差融合可用 `residual_fusion: add/concat` 切换，默认相加。concat 时，`residual_concat_keep_2d: true`（或 1）保留 `[Q,R]` 的 2d 维并适配本模型后续宽度；false（或 0）使用原始/增强分支共享的可学习 Linear(2d,d) 映射回 d，初始权重为 [I,I]。两项均在 configs/models/newq_v2.yaml 中设置，训练命令和旧相加权重恢复方式不变。

DHA-KT 是独立的 `dha_kt` 模型：从训练集内部交叉拟合的固定行为轮廓生成候选，用已观察答案的责任权重修正历史记忆，再通过共享检索器进行第二遍未来预测。正式后端采用可微 DDIM，目标为 KT + 0.5×pilot + 0.1×轮廓去噪，不使用身份 InfoNCE；另保留 legacy 题目扩散对照。完整训练参考库已放在各数据集的 dha_kt/reference 下。代码、阶段训练、恢复及检查范围见 [docs/dha_kt_usage.md](docs/dha_kt_usage.md)。

## 运行

在项目根目录、已激活的 `giakt-main` 环境中运行：

```powershell
# 默认 ASSIST09，最多 100 轮，自动选择设备和保存目录
python main.py

# ASSIST17，最多 50 轮
python main.py assistment2017 --epochs 50

# ASSIST17 的 LinearQ，最多 50 轮
python main.py assistment2017 --model linearq --epochs 50 --device cuda

# NewQ，最多 50 轮
python main.py assistment2009 --model newq --epochs 50 --device cuda

# NewQ V2，最多 100 轮
python main.py assistment2009 --model newq_v2 --epochs 100 --device cuda

# DHA-KT，使用已构建的完整训练参考库；最多 50 轮含 5 轮预热和 5 轮扩散预训练
python main.py assistment2009 --model dha_kt --epochs 50 --device cuda
python main.py assistment2017 --model dha_kt --epochs 50 --device cuda
```

默认设备为 auto，优先使用可用的 CUDA。训练结果按“模型 / 数据集 / 单次实验”保存，例如 `experiments/diffuqk/assistment2017/assistment2017_50epochs_1/`；相同模型、数据集和轮数组合再次运行时编号递增。基线文件夹按项目约定命名为 diffuqk，命令行与配置中的模型名仍为 diffuqkt。实际完成轮数受早停控制，记录在 summary.json 中。

默认模型为 diffuqkt，支持 --model linearq。LinearQ 保存在 `experiments/linearq/<数据集>/` 下，单次实验名称保留 linearq_ 前缀，例如 `experiments/linearq/assistment2017/linearq_assistment2017_50epochs_1/`。新增模型完成注册后使用各自的模型名目录。评估时自动读取 checkpoint 中的模型名称；原始旧权重继续兼容。LinearQ 的 CSV 训练损失显示为 `(KT, LC, L_item)`，第三项对完整训练题目集合等权计算，避免交互频率影响正则权重。

`--model newq` 的实验自动保存在 `experiments/newq/<数据集>/newq_<数据集>_<轮数>epochs_<编号>/`，统一 CSV 中仍记录真实的 `(KT, LC, Diff)`。

`--model newq_v2` 使用 `experiments/newq_v2/<数据集>/newq_v2_<数据集>_<轮数>epochs_<编号>/`；CSV 第三项为实际目标的扩散重建损失，summary 保存有效模式及目标名称。

`--model dha_kt` 默认自动读取对应数据集目录下的 `dha_kt/reference`，也可通过 `--reference-artifact` 指定其他库。结果使用 `experiments/dha_kt/<数据集>/dha_kt_<数据集>_<轮数>epochs_<编号>/`。训练 CSV 保留 `(KT, NA, Diff)`，pilot 独立保存在 history/summary；参考 buffer 自包含于 checkpoint，独立评估不需要原 artifact 路径。`--resume <last.pt>` 支持恢复 DHA 阶段、Adam 与随机数状态。

每次训练完成自动更新根目录的 `experiments/results.csv`，递归汇总各模型、数据集分组以及尚未迁移的平铺目录；已有 CSV 中没有对应目录的历史实验也继续保留。每次实验占 train、valid、test 三行，记录最佳验证轮次对应的指标；train 的 loss 为 `(KT, LC, Diff)` 三元组，其余两行为预测 loss。experiment 列的三行依次存储实验目录名、训练结束时间（`YYYY-MM-DD HH:MM:SS`）和总耗时（如 `61m5s`）。最后一列为“备注”，三行均可手动填写，后续训练或手动重建汇总表时按实验名和 split 保留。新训练在 summary.json 中保存含时区的 finished_at；旧实验以 summary.json 的修改时间近似结束时间，缺少完整耗时记录时耗时仍留空。计时从训练数据加载开始，包含模型迁移到设备、训练/验证与最终测试，不含静态预处理。可用 `python -m src.training.results` 手动汇总已有实验。

每轮终端日志将 train、valid 分成两行，loss、AUC、ACC 按列对齐并显示四位小数。后续新训练在 results.csv 中的指标和损失分量也使用四位小数；旧实验仍保留八位格式及已有备注。计算、选模和 history.csv、JSON、checkpoint 中的原始指标保持完整精度，格式化只作用于终端与汇总表。

**所有训练、评估、预处理、诊断与检查命令统一记录在 [src/training/commands.md](src/training/commands.md)。** 命令行参数解析集中在 src/training/cli.py，支持原有 `--config` 写法及可选参数覆盖。

训练是否重新预处理由 `train.preprocess_each_run` 控制，当前为 false，复用已有 processed；缺少 metadata.json 或显式指定 `--preprocess` 时重建数据与学生划分。同一次训练的各轮使用同一份数据。固定数据与种子会得到相同划分。每轮评估验证集并选取最佳权重，最后使用该权重评估测试集，同时输出验证与测试结果。验证/测试不计算训练辅助损失，因此这两项报告为 null。

## 文件职责

- `scripts/preprocessing.py`：共用静态索引、学生划分、序列打包和预处理分发。
- `scripts/assistment09/`、`scripts/assistment17/`：数据集专属清洗和格式适配。
- `src/data/interactions.py`：训练时的 NPZ 读取、掩码生成与批次拼接。
- `src/model/diffuqkt/`：完整基线，包括供实验模型复用的基础嵌入、下游知识追踪与注意力，以及扩散模块。
- `src/model/linearq/`：LinearQ 的线性增强、训练题目集合初始化与按题等权残差约束，直接复用基线的下游实现。
- `src/model/newq/`：NewQ 的概念加权与题目残差，只改初始 Q 及其条件输入，继承完整 DiffuQKT 后续流程。
- `src/model/newq_v2/`：独立代码、余弦概念权重、扩散后的残差绕行，以及保存权重的只读诊断。
- `src/model/dha_kt/`：固定行为轮廓扩散、候选/责任/两遍历史同化、参考 buffer 与只读诊断。
- `scripts/dha_kt/`：从当前训练划分离线构建交叉拟合参考库；不修改已有清洗与划分。
- `src/model/registry.py`：模型名称注册、创建与 checkpoint 模型识别。
- `src/training/`：训练与评估控制、命令行解析、统一命令说明。
- `src/utils/`：配置、设备、指标与文件校验工具。

详细文件说明、数据格式和模型流程见 [docs/project_files.md](docs/project_files.md)。

模型继承边界、模型注册方式及 `src` 各目录的管理约定见 [docs/model_architecture.md](docs/model_architecture.md)。
