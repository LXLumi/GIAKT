# DHA-KT 实现、运行与验收

实现日期：2026-10-19。模型名 `dha_kt`，公开类 `src.model.DHAKT`。设计依据为 [dha_kt.md](dha_kt.md) 与 [dha_kt_implementation.md](dha_kt_implementation.md)；本文件描述实际代码与检查范围。现有四个模型的源码及 YAML 保留，特别是用户当前 NewQV2 的 both/concat/保留 2d 设置未修改。

## 1. 当前实现

正式后端 `evidence_profile` 使用训练学生内部交叉拟合的固定行为轮廓。无扩散的题目编码器构造 S、R、Q0，ProfileDiffusion 从独立噪声生成轮廓，经小幅非零初始化的无偏置 H→d 投影得到 DeltaR，默认构造 `Qm=F(S,R+DeltaR)`。三种融合为 add、concat→d、concat→2d，原始及生成候选共享融合参数。关闭基础 R 时只支持 add，且不创建残差代码表和 W_R；生成修正仍可保留。

第一遍使用原始 Q0 历史得到作答前候选概率。先验固定 raw=0.5，其余 0.5 均分给默认三个生成候选；通过概率混合得到 pilot 预测。观察答案后用稳定 log-likelihood/softmax 得到责任，默认只 detach 责任，保留候选 payload 的梯度。责任均值/方差通过零初始化的无偏置适配器修正历史交互。第二遍保持 Q0 查询/key，只替换历史 values，严格检索 j<t，最终仍使用作答前先验混合。两遍共享同一检索器和预测头，候选维共享每个位置的预测头 dropout 掩码。

默认 joint 目标为 `KT + 0.5*pilot + 0.1*denoising`。不使用身份 InfoNCE；contrastive_weight 非零报错，contrastive_loss 为 null。pilot_loss 独立保存，不能填入 CL 列。扩散监督从全训练集合的唯一 `(排除折视图,题目,当前去重概念集合)` 键中均匀抽样，默认每步最多 128 个键，对每个键的已观测目标维度按有界计数可靠度加权，最后对键取均值；不会按当前 batch 的重复交互次数计算去噪目标。

## 2. 完整参考库与训练命令

本次已经基于当前完整训练划分建立以下参考库，可直接使用训练命令，无需重新清洗/划分数据：

| 数据集 | 参考目录 | 唯一训练事件 | 移除的窗口上下文重复 |
| --- | --- | ---: | ---: |
| ASSIST09 | `data/assistment09/dha_kt/reference/` | 181623 | 387 |
| ASSIST17 | `data/assistment17/dha_kt/reference/` | 562706 | 2277 |

每库使用 3 个学生折、8 个抽样 draw、8 个等宽概率分箱。参考预测器是固定 prefix_beta_skill，先预测后更新；全局 Beta(1,1) 平滑，各概念用强度 5 的全局前缀先验，多概念交互只更新一次全局计数。轮廓为 `4*sum(y-p_ref)/(n+5)`。draw0 的 dense 使用全部合法参考学生、partial 抽其中 50%；其他 draw 按文档的嵌套学生子集协议构造。

从 `sourceRow` 追踪真实事件，仅移除同学生同 sourceRow 的重叠上下文；相同题号/答案的真实重复练习保留。训练学生读取排除所属折的库，即使对训练集执行 eval 诊断也不切为 full_train；验证/测试只读取训练集全库，并核对学生互斥性。当前/未来答案、学生 ID 和 dense 目标不作为生成器输入。

```powershell
# 当前完整参考库已存在，可直接训练
python main.py assistment2009 --model dha_kt --epochs 50 --device cuda
python main.py assistment2017 --model dha_kt --epochs 50 --device cuda

# 重新预处理/改变划分后，明确重建对应参考库
python -m scripts.dha_kt.build_reference --config configs/assistment2009.yaml --output-dir data/assistment09/dha_kt/reference --overwrite
python -m scripts.dha_kt.build_reference --config configs/assistment2017.yaml --output-dir data/assistment17/dha_kt/reference --overwrite
```

model.profile.artifact_path 默认 null：根据实际训练文件 `<数据集目录>/processed/train.npz` 自动查找 `<数据集目录>/dha_kt/reference`。因此 `python main.py assistment2009 --model dha_kt` 可直接使用当前已生成的 ASSIST09 参考库，ASSIST17 同理。仍可在 YAML 填入目录或使用 `--reference-artifact <目录或manifest>` 覆盖，显式路径优先且不会在路径失效时静默回退。实际路径和 manifest SHA256 写入运行配置与 checkpoint。缺失参考库仍明确报错，不自动构建、改成 legacy 或使用全零假库；范围、元信息和哈希校验保留。

输出自动进入 `experiments/dha_kt/<dataset>/dha_kt_<dataset>_<epochs>epochs_<n>/`。根 results.csv 沿用三行布局，训练 loss 为 `(KT, NA, Diff)`；pilot、候选分歧、责任指标和阶段信息保存在 history.csv/summary/checkpoint，不更改旧 CSV 的列含义。

## 3. 阶段训练、评估与恢复

默认 5 轮 base_warmup、5 轮 diffusion_pretrain，剩余轮次 joint。因此 `--epochs 50` 是总共最多 50 轮，包含 10 轮前置阶段；joint 仍受现有早停控制。

| 阶段 | 更新 | 优化目标/输入 |
| --- | --- | --- |
| base_warmup | 题目编码、答案、共享检索及预测 | 仅原始 Q 的预测 BCE；不使用同化修正 |
| diffusion_pretrain | 仅去噪器 | 固定 partial→dense 去噪监督；结构条件来自已冻结的预热编码器 |
| joint | 所有模型参数 | 实际可微采样、两遍编码和三项损失 |

Adam 从开始即包含所有未来可训练参数，阶段只控制 requires_grad；不会解冻后漏掉参数。阶段策略在 `dha_kt_stages.py`，公共 trainer 仅调用可选 hook。切换阶段重置验证选模和早停计数，进入 joint 后不会拿 warmup 的 raw 指标与 joint 混合预测竞争；前置阶段不提前停止。配置总轮数若只覆盖前置阶段，保存及报告的是该阶段结果，不能当成完整 joint 实验。

```powershell
# 独立评估：不需要 --model 或 --reference-artifact
python main.py assistment2009 --evaluate experiments/dha_kt/assistment2009/dha_kt_assistment2009_50epochs_1/best.pt --device cuda

# 从 last 继续至总轮数 100；自动创建新目录，沿用保存结构、阶段、参考库和 Adam/RNG
python main.py assistment2009 --model dha_kt --resume experiments/dha_kt/assistment2009/dha_kt_assistment2009_50epochs_1/last.pt --epochs 100 --device cuda
```

checkpoint 自包含参考条件/目标/计数/mask、学生折/划分映射、监督键及 manifest，buffer 形状写入 model_kwargs。删除原 artifact 路径后仍可评估或恢复训练，当前 YAML 不覆盖保存结构；恢复训练还核对 runtime scope，并需要 checkpoint 同目录的 best.pt 保留既有最佳验证结果。保存阶段、epoch、global_step、完整 history、Adam、CPU/CUDA/NumPy/Python RNG 和 DataLoader shuffle 状态。CPU 检查已验证分阶段中断续训与连续训练逐值一致；跨硬件/库版本不作逐位一致承诺。

duration_seconds 从数据加载开始，包含参考 buffer 载入、模型迁移、训练/验证、保存及最终测试。stage_seconds 按各阶段训练/验证 batch 的执行时间求和，不包含保存 I/O；sampling_seconds 按批次统计。CUDA 候选计时同步设备，包含按键构造/采样和融合；不将离线 artifact 构造混入模型训练时间。诊断里的额外梯度与修正探测不计入其候选采样耗时。

## 4. 小范围检查必须匹配参考范围

`--smoke` 使用 d=32、2+2 代码、64 窗口/32 长度，并显式跳过前两阶段检查 joint；仍保留 100 步日程与 8 点 DDIM。应为该范围单独构建 artifact：

```powershell
python -m scripts.dha_kt.build_reference --config configs/assistment2009.yaml --output-dir data/assistment09/dha_kt/reference_smoke --max-windows 64 --max-seq-len 32
python main.py assistment2009 --model dha_kt --reference-artifact data/assistment09/dha_kt/reference_smoke --smoke --max-batches 2 --device cuda
```

若修改 max_windows/max_seq_len，重建匹配范围；不允许静默使用完整参考库冒充同范围 smoke。max_batches 只限制本次优化/评估遍历的批次数，不改变可加载数据集及参考库范围，应在报告中说明。参考 builder 缺少可验证 sourceRow 时终止，不用 `(student,question,answer)` 伪造身份。

## 5. 结构开关与 legacy 对照

- `residual_fusion: add/concat` 与 `residual_concat_keep_2d: false/true` 保留三融合；2d 时答案/适配器/检索/预测一起扩宽。
- `profile_injection: residual_update/semantic_update` 决定修正槽，只选一个。
- `profile_residual_condition: none/detach/trainable` 决定独立基础 R 条件；detach 不冻结基础 R 的其他预测路径。
- `responsibility_mode: posterior/prior` 提供 r→pi 机制消融；默认 posterior，温度 1 才对应原指定混合观测模型后验。
- `generated_correction_enabled: false` 让候选重复原始 Q，测试同化退化；不将此开关误称为删除整个扩散模块。
- `surprise_input: true` 为显式的 surprise 实验，默认 false，增加适配器一个输入维度。

legacy 后端不读 profile artifact，不训练 profile loss，采用原题目 live 目标和完整可微 DDPM；不能与固定行为轮廓结果混写。需同时覆盖这些专属设置，例如 d=128：

```yaml
model:
  candidate_backend: legacy_question
  profile_injection: null
  profile_residual_condition: none
  legacy_residual_mode: both
  legacy_residual_condition_detach: false
  diffusion:
    prediction_type: x0
    schedule: linear
    sampler: ddpm
    steps: 100
    sampling_steps: 100
    ddim_eta: 0.0
    time_embedding_dim: 128
    hidden_dim: 256
```

off/condition 生成 S 后执行一次 F(S_hat,R)，target/both 直接使用完整输出；target 的生成候选不直接读取当前 R，但 Q0/查询/原始历史仍使用 R。legacy 复用基线逆扩散循环的未装饰可微函数，不修改旧模型的 no_grad 评估接口。两后端的互斥字段不合法时立即报错。

## 6. 只读诊断与逐文件职责

```powershell
python -m src.model.dha_kt.diagnostics --config configs/assistment2009.yaml --checkpoint experiments/dha_kt/assistment2009/dha_kt_assistment2009_50epochs_1/best.pt --device cuda --max-batches 2 --output experiments/dha_kt_diagnostics.json
```

报告候选概率方差、责任熵/先验距离、候选范数、R/S 与 DeltaR/S 比例、raw 预测质量、最终及 pilot 的 NLL/Brier/ECE、训练参考计数分组、阶段/采样耗时，以及 KT/pilot/Diff 对各模块的单批梯度。比例按唯一参考/题目/概念键统计，其他统计注明对应预测/交互范围。legacy 的完整输出不能唯一拆 R 时，不伪造 DeltaR/S。诊断不更新参数、不重写 checkpoint；有限批次不是全训练平均。

| 文件 | 作用 |
| --- | --- |
| `src/model/dha_kt/model.py` | 独立构造/配置校验、两遍 forward、阶段 hook、损失、准备/恢复与实验元数据。 |
| `src/model/dha_kt/item_encoder.py` | NewQV2 公式的受控无扩散适配：独立代码、集合余弦融合、基础 R 开关和统一 F；无闲置旧扩散或下游网络。 |
| `src/model/dha_kt/diffusion.py` | 轮廓 x0 去噪、cosine/linear 日程、正确 DDIM 跳步和可微采样，以及 legacy 的原循环适配。 |
| `src/model/dha_kt/candidates.py` | 去重条件键、按稳定 SHA256 派生局部噪声、两个后端候选组装、forward 内 gather 与参数快照评估缓存。 |
| `src/model/dha_kt/evidence.py` | 稳定概率混合、责任、候选矩及候选共享 dropout 的预测头。 |
| `src/model/dha_kt/retriever.py` | 继承基线 attention 层，仅分离 Q 查询与历史交互输入；两遍共享实例。 |
| `src/model/dha_kt/reference.py` | 哈希/集合/坐标校验、持久 buffer 与 manifest、split/fold 查询；不启动教师或静态构建。 |
| `src/model/dha_kt/diagnostics.py` | 保存权重的独立只读诊断及命令行。 |
| `src/model/dha_kt/__init__.py` | 导出 DHAKT。 |
| `scripts/dha_kt/build_reference.py` | 静态事件去重、前缀参考预测、学生折/抽样、固定轮廓和 manifest 构建。 |
| `scripts/dha_kt/__init__.py` | 静态构建包标识。 |
| `src/training/dha_kt_stages.py` | DHA 专属 epoch→阶段策略和冻结/解冻规则。 |
| `configs/models/dha_kt.yaml` | 独立模型、后端、参考、采样、阶段和损失配置，参数逐项中文注释。 |
| `tests/dha_kt_helpers.py` | 有真实事件 provenance 与互斥学生的微型检查数据，不是正式训练数据。 |
| `tests/test_dha_kt_reference.py` | 学生折隔离、前缀时序、事件身份、嵌套子集、哈希/范围及 buffer 恢复。 |
| `tests/test_dha_kt_diffusion.py` | DDIM 数值、终端 SNR、固定目标、缺失监督及条件 detach。 |
| `tests/test_dha_kt_evidence.py` | 手算责任、共享 dropout、因果性、首事件、融合、payload 梯度、稳定键和 legacy 路由。 |
| `tests/test_dha_kt_training.py` | 阶段/Adam 更新、严格恢复、独立评估及连续/中断训练一致性。 |
| `tests/test_dha_kt_integration.py` | 注册/CLI、只读诊断、完整参数预算及显式 split 协议。 |

通用接入仅涉及 registry、公开导出、MODEL_CONFIGS、CLI，以及 trainer 的可选上下文/阶段/额外标量/恢复协议。其他模型未实现这些 hook 时保持原路径。旧 results.csv 格式、现有数据静态清洗、所有旧模型代码和 YAML 未改。

每个参考目录的 `manifest.json` 保存来源、范围、学生折/抽样、固定坐标和文件哈希；`profiles.npz` 保存 partial/dense 轮廓、计数、mask、学生归属及唯一监督键；`prefix_predictions.npz` 保存训练事件的作答前参考预测，供复核。三个文件均非模型可训练参数。

## 7. 本次实际验收

`python -B -m unittest discover -s tests -v` 共 142 项通过，其中新增 28 项 DHA 检查。保持 90 个旧模型/配置/静态处理/数据元信息/正式实验文件的 SHA256 不变。58 份现有权重严格加载成功：每类 best/last 分别为 DiffuQKT 5、LinearQ 8、NewQ 1、NewQV2 15。

真实数据检查环境为 giakt-main、PyTorch 2.4.1+cu124、NVIDIA GeForce RTX 3050 Ti Laptop GPU。两数据集分别运行 add、concat→d、concat→2d，共 6 次、每次 3 轮（预热/扩散预训/joint 各一轮）。使用 d=128、8+8 代码、H=8、M=3、100 步 cosine 日程及 8 点 DDIM；每划分 64 窗口、长度 32、batch=8、每轮最多两批，参考库同范围单独构建。不是完整数据收敛训练。

| 数据集 | add 参数量/基线倍率 | concat→d | concat→2d |
| --- | --- | --- | --- |
| ASSIST09 | 754269 / 1.381028 | 787037 / 1.441024 | 1839965 / 3.368881 |
| ASSIST17 | 486242 / 0.926002 | 519010 / 0.988406 | 1571938 / 2.993609 |

六次均完成训练、验证选模、best/last 严格恢复、最终测试、独立恢复评估及只读诊断；固定权重恢复后的 AUC/ACC/BCE/loss 与保存测试一致。KT 和 pilot 经实际采样传到生成器的梯度均非零。另完成一次真实 CUDA 3→4 轮断点续训，以及一次无参考库的 legacy both/concat→2d、完整 100 步 DDPM 联合训练与恢复测试。

这些有限检查只证明代码、数据隔离、因果性、梯度和保存恢复流程。短程检查的候选概率分歧仍很小，不能声称已经获得有效教育语义、不确定性校准或性能增益。多种子完整实验、第三数据集及扩散必要性/等成本替代候选的研究对照尚未开展。临时权重/脚本/报告清理，完整参考库保留供正式训练。

## 8. 默认启动路径修复（2026-10-19）

旧实现在 profile.artifact_path=null 时直接报必填错误，导致已有默认参考库仍不能通过 `python main.py assistment2009 --model dha_kt` 启动。现在根据实际 dataset.path 自动定位对应数据集的 dha_kt/reference；显式 YAML/命令行路径优先。实际路径持久化，已恢复的 bank 不重新访问原目录，所有参考内容/范围校验保留。

新增通过真实 CLI→train→prepare_training 的无路径参数回归检查，覆盖两数据集目录；另覆盖显式路径优先、显式路径失效不回退、默认库不存在时明确报错。本次 30 项 DHA 检查通过。ASSIST09/17 各以省略 --reference-artifact 的真实命令完成一轮 CUDA 启动/训练/验证/保存/测试，每划分最多一个 batch，保留完整数据加载范围，与默认完整参考库匹配。临时实验已清理，默认库和正式实验保留。
