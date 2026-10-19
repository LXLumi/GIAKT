# NewQ V2：独立题目代码与可配置残差扩散

实现日期：2026-10-18。依据 [q_structure_v1.md](q_structure_v1.md) 第 13 节实现独立代码，并按第 15 节加入残差扩散插件。模型标识仍为 `newq_v2`，类名仍为 `NewQV2`。现有 `newq` 继续表示 V1，模型及 checkpoint 分开管理。

构造函数默认关闭插件，以保持缺少路由字段的历史 checkpoint 的 off 行为。当前 YAML 的 residual_diffusion_enabled=false 是用户已有设置，本次保留。插件只控制 R 在扩散中的路径，关闭时仍保留残差、扩散与原始状态分支。2026-10-19 新增独立的 residual_condition_detach，默认 false；需要条件梯度对照时显式启用 condition 并打开该标志。

同日新增 `residual_fusion: add/concat` 与 `residual_concat_keep_2d: false/true`，默认 add。concat 时 false/0 用共享线性层映射回 d，true/1 保留 2d 并适配本模型的扩散输出和下游宽度。下文以 F(Q,R) 表示统一融合函数，具体规则见末尾“相加与拼接控制”；旧验收章节保留当时 add 的结果。

## 表示构造

```text
a_q = attention_question_embedding[q]     # r_A=8
z_q = residual_question_embedding[q]      # r_R=8，两张独立参数表
key_k = key_projection(C_k)               # 无偏置 Linear(d, r_A)
score_qk = (gamma/tau) * dot(normalize(a_q), normalize(key_k))
alpha_qk = softmax_K(score_qk)            # gamma=2，tau=1，不除 sqrt(r_A)
C_bar = sum_k alpha_qk * C_k              # 使用原始 value，不对 C/V 做 L2 归一化
V_bar = sum_k alpha_qk * V_k
D_q = sigmoid(difficulty[q])
Q_sem = C_bar + D_q * V_bar
R_q = residual_projection(z_q)            # 无偏置 Linear(r_R, d)
Q_orig = F(Q_sem, R_q)                   # 默认相加；可选拼接及维度控制
```

两个题目表分别以标准差 0.02 的正态分布初始化，padding_idx=0 且第 0 行清零；两个投影沿用 PyTorch 默认初始化。gamma/tau 是固定配置值，不注册为可学习参数。L2 归一化仅作用于最后一个特征维，epsilon=1e-8，保留计算图。

兼容单概念和多概念输入，对当前记录的概念排序去重，padding 不参与权重归一化。有有效概念时权重和为 1；空概念集合的权重、C_bar、V_bar、Q_sem 均为零，有效题目的残差仍保留。无效事件的语义、残差、原始/增强表示和权重全部为零。单概念权重恒为 1，打分梯度为零。不同记录的标签视图各自处理，同题共享残差，不读取答案或学生状态。

## 扩散与下游

NewQV2 直接继承 DiffuQKT，重写 `question_components` 和 `enhance_questions`，主 forward 直接复用基线。训练、评估与诊断共用 `diffusion_route` 和 `generate_questions`，按下面的模式选择目标、条件和输出：

| enabled | role | 有效模式 | 加噪/重建目标 | 独立 R 条件 | 训练/采样结束后的 Q_star |
| --- | --- | --- | --- | --- | --- |
| false | 三种合法值任意 | off | Q_sem | 无 | F(generated, R) |
| true | condition | condition，独立条件 | Q_sem | 有 | F(generated, R) |
| true | target | target | Q_orig=F(Q_sem,R) | 无 | generated |
| true | both | both | Q_orig=F(Q_sem,R) | 有 | generated |

target 严格为仅目标，不能解释为 both。固定其他参数、C_bar、D 和采样 generator 时，只改变当前 R 不会改变 target 的评估主分支；这是本次实验选择的信息路径。训练阶段仍通过含 R 的加噪目标与 MSE 目标获得梯度。both 则在每个去噪步骤额外提供 R。

扩散输出为 d 时，condition/both 将未加噪的 d 维 R 追加到原输入 `[C_bar, noisy, D, time_rep]` 后，宽度为 `4*d+1`；不将 R 与 C_bar 相加。off/target 保持原扩散对象与 `3*d+1` 宽度。保留 2d 的 target/both 改用末节说明的输入/输出适配。初始噪声始终为标准高斯，不人为叠加 R；采样日程、posterior/fixed_large 方差规则和 100 步均保留。插件复用基线加噪、MSE 及采样循环，通过每次调用独立的适配器将同一份已掩码 R 传入每个去噪步骤，不维护跨调用的可变条件状态。

构造顺序先完成原 V2 的全部参数，再创建扩展模块，复制时间 MLP、日程和共有去噪权重。第一层前 `3*d+1` 列复制原值，最后 d 列清零，第二层与偏置保持原值。扩展构造放在保留 CPU RNG 状态的上下文中，避免临时随机初始化改变后续训练噪声序列。新增列为零时，初次 Diff 对 R 的条件梯度可为零；条件列自身可获得梯度，正常学习后该路径可以起作用。

原始状态对比分支始终使用 Q_orig=F(Q_sem,R)，增强分支使用模式规定的 Q_star。总损失仍为 `KT + 0.001*CL + 0.01*Diff`，Diff 对表中所选目标计算有效位置/输出维度归一化的 MSE；目标保持正常梯度，独立条件是否 detach 由下述新标志决定。验证/测试的辅助损失仍为 null，不新增 L_R 或熵损失。

| 有效模式 | KT/CL 对残差参数 | Diff 对残差参数 |
| --- | --- | --- |
| off | 保留直接输出及原始状态路径 | 无依赖 |
| condition | 直接输出和原始状态路径，也可通过条件 | 条件列非退化时可通过条件 |
| target | 训练的加噪输入与原始状态路径 | 加噪输入和完整重建目标 |
| both | 训练的加噪输入、条件与原始状态路径 | 条件、加噪输入和完整目标 |

上表描述 residual_condition_detach=false 的原路径。设为 true 后，只删除表中的“条件”梯度路径；condition 的 Diff 不再依赖残差参数，KT/CL 仍有正常输出与原始状态路径。both 仍可经完整目标及加噪输入训练残差参数，不能据此要求其 Diff 梯度为零。off/target 无独立 R 条件，标志不影响这些模式。

## 参数与接口

默认 add 时，off/target 的题目结构新增参数为 `(N+1+d)*(r_A+r_R)`；condition/both 另外增加去噪器第一层的 `2*d*d` 个条件权重，不增加偏置或第二套去噪器。concat 的额外参数见末节；预算始终包含完整扩散、KT 主干和预测头，超过基线 3 倍警告，超过 4 倍报错。默认总代码维度仍为 16，默认输出仍为 d=128。

`question_components` 提供 `concept`/`fused_concept`、`variation`、`scale`/`conditioning`、`semantic_original`、`residual`、`original`、两种代码、`concept_ids`、`concept_weights` 和 `concept_mask`。其中 concept 严格为纯 d 维 C_bar，original 为融合后的 Q；诊断时按 concept_ids 与 concept_mask 对齐权重。

| 文件 | 作用 |
| --- | --- |
| `src/model/newq_v2/model.py` | 独立代码、余弦概念融合、相加/拼接和共享投影、宽度适配、集中路由与预算检查。 |
| `src/model/newq_v2/diffusion.py` | 可选独立 R 条件及 2d 完整目标的输入/输出适配；复用基线的训练和采样循环。 |
| `src/model/newq_v2/__init__.py` | 导出 NewQV2。 |
| `src/model/newq_v2/diagnostics.py` | 保存权重的只读诊断，默认输出到 stdout，可显式保存报告。 |
| `configs/models/newq_v2.yaml` | 两种代码维度、gamma/tau 和原有扩散参数，所有参数带中文说明。 |
| `tests/test_newq_v2.py` | 第 13 节的数值、掩码、调用路由、残差精确绕行、梯度、基线退化、因果性、预算和诊断验收。 |
| `tests/test_newq_v2_plugin.py` | 第 15 节的四模式路由、逐步条件、数值依赖、初始化等价、预算、保存模式恢复与只读诊断验收。 |
| `tests/test_newq_v2_detach.py` | 条件 detach 的 bool 校验、四模式前向/参数不变、梯度隔离、正常残差路径、条件数值可见性、保存恢复与诊断验收。 |
| `tests/test_newq_v2_fusion.py` | 两种拼接维度、共享投影、完整目标/采样公式、detach 梯度兼容、因果性、参数预算、训练保存恢复和只读诊断。 |

插件没有增加模型标识、注册项或另一份 YAML。公共训练器只增加可选的 experiment_metadata 接口，用于记录模式到启动日志、checkpoint 和 summary；没有该接口的模型沿用原行为。数据预处理、原 DiffuQKT、LinearQ 与 NewQ V1 保留。

V2 的 prepare_training 返回路由、条件 detach 与融合/维度标志，不做数据转换。即使直接使用缺少新字段的旧配置启动训练，也会将构造默认值显式写入新 model_kwargs 和实际运行配置；历史 checkpoint 的只读加载不调用此准备步骤。

## 命令与版本恢复

在 configs/models/newq_v2.yaml 中修改路由参数即可切换模式；下面是启用 condition 的示例，不代表覆盖用户当前文件值：

```yaml
residual_diffusion_enabled: true
residual_diffusion_role: condition
residual_condition_detach: false
```

enabled 与 residual_condition_detach 必须为 Python/YAML bool，拒绝 0/1、字符串或 null；role 必须为精确小写 condition/target/both，关闭时也校验。四种路由模式仍只由 enabled/role 推导，detach 是独立的梯度对照因素。

```powershell
python main.py assistment2009 --model newq_v2 --epochs 100 --device cuda
python main.py assistment2017 --model newq_v2 --epochs 100 --device cuda

# 小规模流程：d=32、r_A=r_R=2，完整保留 100 步扩散
python main.py assistment2009 --model newq_v2 --smoke --device cuda

# 从保存的名称和构造参数自动恢复 V2
python main.py assistment2009 --evaluate experiments/newq_v2/assistment2009/newq_v2_assistment2009_100epochs_1/best.pt --device cuda
```

输出仍进入 `experiments/newq_v2/<dataset>/newq_v2_<dataset>_<epochs>epochs_<n>/`，根目录 results.csv 继续统一汇总。运行配置与 model_kwargs 保存路由字段和 residual_condition_detach，checkpoint/summary 还保存有效模式、effective_residual_condition_detach、目标/条件/输出路径和实际参数量。层宽相同的模式或梯度对照也按保存字段恢复，不从层宽或预测值推断语义。

构造函数默认 enabled=false、role=condition、residual_condition_detach=false；因此缺少路由字段的历史 V2 按 off 严格恢复，只有缺少 detach 字段时保留原路由并按 false 恢复，不受当前 YAML 影响。评估从 checkpoint.model_kwargs 创建结构，不做 strict=False、列裁剪/填充或静默迁移。旧 V1 仍按旧类恢复。

融合字段缺失时按 add/false 恢复，原参数键与层形状保持一致；不能将拼接权重作为相加模型静默加载。

## 只读诊断

```powershell
python -m src.model.newq_v2.diagnostics --config configs/assistment2009.yaml --checkpoint experiments/newq_v2/assistment2009/newq_v2_assistment2009_100epochs_1/best.pt --device cuda --max-batches 2 --max-windows 32 --max-seq-len 32 --output experiments/newq_v2_diagnostics.json
```

省略 --output 时只打印 JSON。诊断加载模型副本、不创建优化器、不更新参数、不改写 checkpoint 或原有实验指标；显式输出路径不得覆盖所加载的 checkpoint。

报告包含：

- 多概念事件的归一化熵和相对均匀权重最大偏差；排除 K<=1，并提供最多 8 条题目/概念 ID 与权重对齐示例。
- `||R||/max(||Q_sem||,1e-8)` 的分布，包含 p50/p95/p99。
- 完整反向采样后的生成与最终表示范数 p95/p99/max，以及预测位置的最大绝对 logit。off/condition 使用 semantic_generated_norm，target/both 使用 original_generated_norm，不把完整目标叫作语义生成。
- `stable_bce_from_logits` 与 `probability_bce` 两个独立字段。前者使用 float64 logits 的稳定 BCE；后者沿用当前概率指标的计算规则，不将概率裁剪后的 BCE 冒充 logit BCE。非有限值单独计数，不通过表示截断掩盖。
- 单个非打乱训练 batch 分别反传 KT、CL、Diff 的各模块梯度范数，明确批次与预测范围；不进行优化器更新。梯度范数只反映该批次，不表示完整训练平均。
- 请求的路由字段、条件 detach 标志及实际生效状态、有效模式、真实重建目标、条件/后加残差路径及完整参数量。

分布按有效交互计权，包含首个上下文事件；同题多次出现会多次计数。梯度探测使用同一加载范围下训练集的首个 batch，推断统计使用 --split 指定的划分（默认 valid）。诊断执行 checkpoint 中的完整采样步数，max-batches 只限制数据批次。

## 第 13 节 off 初版验收记录（历史）

以下保留原 V2、即插件 off 的验收，不能套用为 condition/target/both 的梯度断言。代码验收与完整训练性能分别报告；各模式的 Diff 目标可能不同，不能仅以 MSE 大小认定生成或预测更好。

2026-10-18 实际完成 87 项回归检查。包括单独反传 Diff 后，采用当前 Adam 与 weight_decay 设置执行更新时，残差表/投影保持逐值不变；V1 相关测试及配置仍保留。

### CUDA 小范围训练与参数实测

两个当前真实 processed 数据集分别训练 5 轮，使用 d=128、r_A=r_R=8、100 步扩散、batch=8；每个划分最多前 32 个窗口、序列长度最多 32，每轮最多两个 batch。复用已有数据与划分，没有重建预处理。环境为 giakt-main，PyTorch 2.4.1+cu124，设备 NVIDIA GeForce RTX 3050 Ti Laptop GPU。

| 数据集 | 基线参数 | V2 参数 | 倍率 | 最佳轮次 | 该轮训练 KT | 该轮训练 CL | 该轮训练 Diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSIST09 | 546165 | 830549 | 1.520692 | 4 | 0.618752 | 2.442726 | 0.980335 |
| ASSIST17 | 525098 | 562522 | 1.071271 | 3 | 0.703551 | 2.536534 | 0.870445 |

实际构造后的参数量与规格表一致。两次均完成验证选模、保存 best/last、加载最佳权重测试，再通过独立 main.py 自动识别 V2 重新评估，AUC、ACC、BCE 与保存的最终测试一致，统一 CSV 共六行。另通过真实 smoke 命令完成 ASSIST09 一轮 CUDA 检查，d=32、2+2 代码、每划分最多 8 个窗口、每轮最多一个 batch，仍执行 100 步采样。这些均是流程检查，不代表完整训练性能。

### 最佳临时权重的只读诊断实测

诊断采用 valid 的前两个 batch，加载范围同上。统计按交互计权，ASSIST09 为 401 个有效交互、385 个预测目标，其中 87 个多概念事件涉及 85 个题目；ASSIST17 为 491 个有效交互、475 个预测目标，当前数据没有多概念事件，因此其注意力熵/偏差留空。

| 诊断量 | ASSIST09 | ASSIST17 |
| --- | ---: | ---: |
| 多概念归一化熵均值 | 0.944524 | 不适用 |
| 相对均匀权重的最大绝对偏差 | 0.329787 | 不适用 |
| 残差/语义范数比 p50 | 0.006917 | 0.008364 |
| 残差/语义范数比 p95 | 0.015392 | 0.015073 |
| 残差/语义范数比 p99 | 0.019756 | 0.016355 |
| 完整生成语义范数 p95 | 3.295629 | 2.889429 |
| 完整生成语义范数 p99 | 3.344881 | 2.976912 |
| 完整生成语义范数 max | 3.346716 | 2.979541 |
| 最终 Q_star 范数 p95 | 3.296936 | 2.893477 |
| 最终 Q_star 范数 p99 | 3.343266 | 2.975311 |
| 最终 Q_star 范数 max | 3.356948 | 2.984310 |
| 最大绝对 logit | 0.847480 | 0.229477 |
| 稳定 logit BCE | 0.5385187459 | 0.6718293803 |
| 原概率 BCE | 0.5385187439 | 0.6718293818 |

本次范围没有非有限表示或 logit；不将局部检查中未出现放大推断为完整训练已解决生成稳定性问题。

梯度探测分别使用相同加载范围内训练集的第一个未打乱 batch，共 8 个窗口。以下为各项**未加权损失**单独反传的模块梯度 L2 范数，不是完整训练平均，也不是优化后的性能指标。

| 数据集 | 参数组 | KT | CL | Diff |
| --- | --- | ---: | ---: | ---: |
| ASSIST09 | E_A | 0.006745 | 0.073188 | 0.076368 |
| ASSIST09 | W_K | 0.001940 | 0.023409 | 0.026366 |
| ASSIST09 | E_R | 0.005622 | 0.014253 | None |
| ASSIST09 | W_R | 0.000742 | 0.001614 | None |
| ASSIST17 | E_A | 0 | 0 | 0 |
| ASSIST17 | W_K | 0 | 0 | 0 |
| ASSIST17 | E_R | 0.004073 | 0.016823 | None |
| ASSIST17 | W_R | 0.000279 | 0.001420 | None |

ASSIST17 的单概念结构使打分梯度为零，符合规格。两个数据集的 Diff 对残差参数均没有计算图依赖。诊断前后 checkpoint SHA256 一致。

当前 14 份既有 best.pt 严格加载成功：DiffuQKT 5 份、LinearQ 8 份、NewQ V1 1 份。旧模型实现/配置、V1 测试、主入口、公共训练与汇总、数据 metadata 及正式 summary/history/results 共 50 个受保护文件的 SHA256 保持一致。临时检查脚本、配置、日志、权重与报告在核验后清理，数值摘要保留于本节。

## 第 15 节插件实现与验收（2026-10-18）

98 项回归检查通过，原 off 验收保留。重排前保存的独立固定输入结果与新 off 实现逐值比较：三个合法 role 的初始化参数、state_dict 键名、训练/评估输出和全部参数梯度均与旧实现完全一致。新增条件模块的共有参数与 off 逐值一致，新增列全零；构造后 CPU RNG 状态也一致。零条件下扩展去噪器与原网络在 float32 容差内等价，不能把这一性质误解为不同目标/输出模式的最终表示等价。

已验证独立条件的实际数值作用、所有 steps 都收到同一已掩码 R、标准高斯初始噪声、两种方差公式、target 的评估预测对当前 R 不变而训练目标随 R 改变。非退化条件权重下验证四模式的梯度表；另验证新条件列全零时 Diff 到 R 的零梯度与断开条件通路不同。对比对象、因果性、padding、参数预算、模式往返和诊断只读性均通过。

### 两个真实数据集的四模式流程检查

使用原 processed，不重建数据。环境仍为 giakt-main / PyTorch 2.4.1+cu124 / NVIDIA GeForce RTX 3050 Ti Laptop GPU，d=128、8+8 代码、100 步扩散、batch=8。各划分最多前 32 个窗口、长度最多 32，每轮最多两个 batch。off/condition 各运行 5 轮，target/both 各运行 1 轮；后两者只检查流程，没有采用相同训练预算做性能比较。

| 数据集 | 模式 | 实际参数 | 完成轮数 | 最佳轮次 | 训练 KT | 训练 CL | 训练 Diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSIST09 | off | 830549 | 5 | 4 | 0.618752 | 2.442726 | 0.980335 |
| ASSIST09 | condition | 863317 | 5 | 4 | 0.618972 | 2.444602 | 0.980012 |
| ASSIST09 | target | 830549 | 1 | 1 | 0.633006 | 3.380278 | 0.796012 |
| ASSIST09 | both | 863317 | 1 | 1 | 0.633003 | 3.380268 | 0.796012 |
| ASSIST17 | off | 562522 | 5 | 3 | 0.703551 | 2.536534 | 0.870445 |
| ASSIST17 | condition | 595290 | 5 | 2 | 0.692186 | 2.971385 | 0.838627 |
| ASSIST17 | target | 562522 | 1 | 1 | 0.683985 | 3.289996 | 0.804175 |
| ASSIST17 | both | 595290 | 1 | 1 | 0.683992 | 3.289982 | 0.804173 |

condition/both 恰多 32768 个参数，分别为基线约 1.581/1.134 倍。全部 8 次运行均完成完整采样验证、保存 best/last、最终测试、独立恢复评估及只读诊断；恢复后的 AUC、ACC、BCE 与保存结果一致。实际 YAML、model_kwargs、checkpoint 和 summary 中的请求开关与有效模式已核对，不从层宽推断 off/target 或 condition/both。

另通过当时默认 true/condition 的真实 --smoke 命令完成 ASSIST09 一轮 CUDA 检查：d=32、2+2 代码、100 步、每划分最多 8 个窗口、每轮最多一个 batch；实际参数量 130381，日志与保存记录均为 condition，训练、选模和测试完整通过。

### 按真实模式执行的诊断

下表仍为 valid 前两个 batch、按交互计权的局部检查。生成向量在 off/condition 下为语义表示，在 target/both 下为完整题目表示；所有报告保留最大绝对 logit、稳定 logit BCE 与概率 BCE。探测梯度来自加载范围内第一个训练 batch，未加权 Diff 单独反传，不代表全训练平均。

| 数据集 | 模式 | 生成范数 max | max绝对logit | 稳定 logit BCE | 概率 BCE | Diff 到 E_R 的梯度范数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ASSIST09 | off | 3.346716 | 0.847480 | 0.5385187459 | 0.5385187439 | None |
| ASSIST09 | condition | 3.350309 | 0.834994 | 0.5404168221 | 0.5404168177 | 0.001732 |
| ASSIST09 | target | 1.940470 | 1.195539 | 0.5106519588 | 0.5106519619 | 0.009521 |
| ASSIST09 | both | 1.943629 | 1.195121 | 0.5106401592 | 0.5106401588 | 0.009687 |
| ASSIST17 | off | 2.979541 | 0.229477 | 0.6718293803 | 0.6718293818 | None |
| ASSIST17 | condition | 2.353937 | 0.267513 | 0.7107401266 | 0.7107401248 | 0.001215 |
| ASSIST17 | target | 1.519594 | 0.699522 | 0.6572067326 | 0.6572067323 | 0.008220 |
| ASSIST17 | both | 1.523870 | 0.696057 | 0.6572049487 | 0.6572049466 | 0.008252 |

本次八组生成结果均没有非有限值；不据此声称完整训练稳定性或预测性能已改善。历史 16 份 best.pt 与 16 份 last.pt 全部严格加载，包含 2 组缺少插件字段的旧 V2，均按 off 恢复。原基线、LinearQ、V1、入口/注册/配置工具/CLI、已有公共及其他模型配置、数据元信息和正式实验记录等 57 个受保护文件哈希保持一致。公共训练器的必要变更由原回归检查覆盖；没有模型元数据接口的基线行为保留。

全部临时权重、日志、配置及报告在核验后清理，本节保留摘要；后续正式实验仍用现有命令与自动编号目录。插件的性能结论需另做同划分、同预算、多种子实验，不能用不同目标的 MSE 或这里的一轮检查比较优劣。

## 条件输入的 stop-gradient 对照（2026-10-19）

新增 `residual_condition_detach: false`，构造默认值同为 false。启用本次对照需在 model 节设置：

```yaml
residual_diffusion_enabled: true
residual_diffusion_role: condition
residual_condition_detach: true
```

此时 `Q_star=f_theta(x_t,t,C_bar,D,R.detach())+R`。仅传给扩散器的条件视图 detach，原始组件 R、输出端加数、原始对比分支以及 target/both 的完整重建目标保持正常梯度。没有新增参数、损失、扩散步数或网络结构变化。

在参数和随机数相同时，开/关标志的训练与采样前向值逐值相同；后续优化轨迹可因梯度不同而分开。生成器仍读取 R 的数值，其条件权重仍正常学习；改变 R 可改变生成结果。condition 的 Diff 不再通过条件更新 E_R/W_R，但 KT/CL 可以经正常残差路径更新它们，因此该标志不是冻结残差模块。

`residual_condition_detach` 保存请求值，`effective_residual_condition_detach` 只有在 condition/both 且标志为 true 时为 true。两者自动进入训练日志、summary、checkpoint 与只读诊断；构造参数也写入实际配置。缺少新字段的历史权重按 false 严格加载，当前 YAML 不覆盖保存值。已有模型标识、路径编号、主干和三项损失均保留。

### 本次对照的实际核验

`python -B -m unittest discover -s tests -v` 的 104 项检查全部通过，新增 6 项覆盖严格布尔校验、四模式前向等价、条件数值依赖、逐项损失梯度、保存/恢复和诊断。另与修改前独立保存的结果比较：省略标志或设为 false 时，四模式的初始化参数、训练/评估输出与全部参数梯度逐值一致。

在 giakt-main / PyTorch 2.4.1+cu124 / NVIDIA GeForce RTX 3050 Ti Laptop GPU 上，两个真实 processed 数据集各执行 condition 模式 detach=false/true，共四次、每次 2 轮。使用 d=128、r_A=r_R=8、100 步扩散；每个划分最多前 32 个窗口、序列长度最多 32、batch=8、每轮最多两个 batch，复用原数据划分。开关不增加参数，ASSIST09/17 均分别为 863317/595290。均完成训练、验证选模、best/last 保存、最佳权重测试、独立 main.py 恢复评估和只读诊断；恢复 AUC/ACC/BCE/loss 与保存结果一致。

以下为各自最佳临时权重在第一个训练 batch 上，未加权 Diff 单独反传的梯度 L2 范数。条件列经过训练后已非零，因而 false 组的梯度可与 true 组的断开路径区分：

| 数据集 | detach | Diff 到 E_R | Diff 到 W_R |
| --- | --- | ---: | ---: |
| ASSIST09 | false | 0.0014213385 | 0.0000463198 |
| ASSIST09 | true | None | None |
| ASSIST17 | false | 0.0012150181 | 0.0000195087 |
| ASSIST17 | true | None | None |

两组 true 的 KT/CL 仍对 E_R/W_R 有非零梯度；生成器条件映射继续参与学习。`None` 表示该损失没有到残差参数的计算图路径，不是零初始化造成的数值为零。诊断前后 checkpoint 哈希一致。

现有 19 份 best 和 19 份 last 全部严格加载（每类分别为 DiffuQKT 5、LinearQ 8、NewQ 1、NewQ V2 5）；缺少新字段的旧 V2 按 false 恢复。其他模型、V2 扩散/诊断、公共训练/入口、数据元信息及正式实验记录等 65 个受保护文件哈希保持一致。临时脚本、配置、权重、日志和结果均已清理。以上是功能与梯度核验，未进行完整训练性能比较。

## 相加与拼接控制（2026-10-19）

新增两个独立字段，保留其他路由和 detach 设置：

```yaml
model:
  residual_fusion: add             # add 或 concat，默认 add
  residual_concat_keep_2d: false    # true/1 保留 2d；false/0 映射回 d，仅 concat 生效
```

`residual_fusion` 严格接受小写 add/concat。`residual_concat_keep_2d` 接受布尔值及整数 0/1，拒绝浮点数、字符串、其他整数和 null；保存时统一转换为 bool。add 时仍校验并保存第二个字段，但它不改变维度或创建新参数。

设 X 是待融合的 d 维语义题目表示，R 仍由原来的残差代码及投影生成，维度为 d。拼接顺序固定为 `[X,R]`：

| residual_fusion | residual_concat_keep_2d | F(X,R) | 最终题目/状态维度 |
| --- | --- | --- | --- |
| add | 任意合法值 | X+R | d |
| concat | false/0 | W_f [X;R] | d |
| concat | true/1 | [X;R] | 2d |

W_f 为共享无偏置 Linear(2d,d)，初始权重 `[I,I]`，随后正常学习两个输入块的混合。原始分支 Q_orig=F(Q_sem,R) 与 off/condition 的 Q_star=F(generated,R) 使用同一个投影参数；没有原始/增强分支各自独立的投影。target/both 以融合后的完整 Q_orig 为加噪和重建目标，生成后直接使用，避免再次融合 R。

保留 2d 时，C_bar、V_bar、Q_sem、R 和独立 R 条件仍为 d；答案嵌入、知识检索器和预测头按 2d 创建，基线 forward、对比公式和预测算法继续复用。扩散宽度如下：

| 模式 | 扩散目标/输出宽度 | 去噪器输入 | 最终 Q_star |
| --- | --- | --- | --- |
| off | d | 3d+1 | [generated(d);R(d)] |
| condition | d | 4d+1 | [generated(d);R(d)] |
| target | 2d | 4d+1 | generated(2d) |
| both | 2d | 5d+1 | generated(2d) |

完整 2d 目标的去噪输入为 `[C_bar(d),noisy_Q(d),noisy_R(d),D(1),time_rep(d)]`，both 再追加独立 R(d)。时间 MLP 维度与去噪隐藏层 2d 保留；输出层扩为 2d。共有权重按语义列复制，新增带噪 R 输入、独立条件列以及 R 输出行置零。专属的 ConcatenatedTargetDiffusion 只适配层宽及初始噪声形状，复用基线的训练、MSE、加噪和完整逆扩散循环；posterior/fixed_large、标准高斯初始噪声、步数及损失系数不变。MSE 对真实输出维度取均值。

`residual_condition_detach` 与两种融合方式兼容，仍只作用于独立条件 R。off 的 Diff 无残差依赖；condition+detach 的 Diff 无残差依赖；target/both 的完整目标及加噪输入保持正常梯度。投影回 d 时，W_f 通过 KT/CL 学习，在 target/both 下也通过完整扩散目标学习。保留 2d 时不创建 W_f。

配置、model_kwargs、checkpoint、summary 和只读诊断保存融合/维度请求值；元数据额外报告 `effective_residual_concat_keep_2d`、`question_representation_dim`、`diffusion_output_dim` 与 `fuse_residual_after_generation`。原 `add_residual_after_generation` 仅在确实相加时为 true。投影存在时，诊断单独报告该模块的梯度。历史缺字段权重按 add/false 严格恢复，当前 YAML 不覆盖保存的结构。

完整参数预算继续生效，包含新增投影及扩大的答案、检索、预测和扩散层；不能通过仅计算题目表绕过 4 倍上限。默认 d=128、8+8 代码时的实测参数：

| 数据集 | 拼接结果 | off | condition | target | both |
| --- | --- | ---: | ---: | ---: | ---: |
| ASSIST09 | 映射回 d | 863317 | 896085 | 863317 | 896085 |
| ASSIST09 | 保留 2d | 1817941 | 1850709 | 1883605 | 1916373 |
| ASSIST17 | 映射回 d | 595290 | 628058 | 595290 | 628058 |
| ASSIST17 | 保留 2d | 1549914 | 1582682 | 1615578 | 1648346 |

本次 114 项回归检查通过，新增 10 项覆盖两种拼接宽度、手工融合、共享投影训练、三项梯度/条件 detach、完整 2d 目标的两种方差采样公式、因果性/padding、完整参数预算，以及八种模式/宽度的训练、保存恢复和只读诊断。与变更前独立保存的八组结果比较，默认 add 的四路由×detach 开关在初始化、state_dict、训练/评估输出和全部梯度上逐值一致。

另在 giakt-main / PyTorch 2.4.1+cu124 / RTX 3050 Ti Laptop GPU 上完成两个真实数据集×两种拼接宽度×四路由，共 16 次一轮 CUDA 流程检查，均使用 d=128、8+8 代码和完整 100 步扩散，每划分最多 8 窗口、长度 16、batch=8、最多一个 batch，条件 detach=true。均完成验证选模、best/last 严格恢复、最终测试、独立恢复评估和只读诊断，恢复指标与保存测试一致。20 份既有 best 和 20 份 last 严格加载（每类 DiffuQKT 5、LinearQ 8、NewQ 1、V2 6）。这些属于有限数据的流程核验，未比较完整训练性能。

公共代码、基线/其他模型、公共及其他模型配置、数据元信息和正式 summary/history/results 等 70 个受保护文件哈希不变。临时脚本、配置、权重和结果在核验后清理；正式实验仍使用原命令与自动编号目录。

## 可选的 D 去噪条件（2026-10-19）

在 `configs/models/newq_v2.yaml` 的 model 节增加独立布尔开关：

```yaml
model:
  diffusion_difficulty_condition: true  # false 关闭显式 D 条件
```

默认 true，保留已有去噪条件。令 D_cond=D（开启）或 0（关闭），训练的 x0 重建及评估的每一步逆扩散统一使用 `f_theta(x_t,t,C_bar,D_cond[,R_cond])`。R_cond 仍由残差路由和 residual_condition_detach 单独决定；此开关兼容 off/condition/target/both 以及相加、拼接回 d、保留 2d 三种融合。

关闭只表示不把题目难度作为独立条件交给去噪器，不是冻结 D 或从题目表示移除 D。`Q_sem=C_bar+D*V_bar`、原始分支、重建目标、加噪输入和三项损失保留，因此 D 仍可经这些路径参与学习。尤其训练时带噪目标仍含 D 的信息，不能把该实验解释为生成过程完全没有难度信息。

实现集中在 NewQV2.generate_questions：关闭时把传入扩散器的标量条件置零，不删除网络列、不添加参数或损失，也不改变参数初始化及采样日程。共享扩散类和 DiffuQKT、DHA-KT 等其他模型无需修改。开关严格接受 YAML true/false；不接受字符串和整数 0/1。

prepare_training 将该字段写入实际配置和 model_kwargs，experiment_metadata 将其写入日志、summary、checkpoint 和只读诊断。评估按 checkpoint 中的构造参数恢复；当前 YAML 不覆盖保存值。缺少字段的旧 checkpoint 按 true 恢复并可严格加载，无需重训或迁移权重。切换开关后开始新训练，仍用原来的命令和自动编号目录。

新增 `tests/test_newq_v2_difficulty.py` 六项检查，覆盖每个实际去噪步骤的条件、四路由和三种融合、隔离后的条件数值依赖、显式条件梯度与表示中 D 的正常学习、参数/RNG 一致性，以及训练保存恢复和诊断。另与修改前代码独立比较 12 组路由/融合，默认 true 的参数、训练/评估输出及全部梯度逐值相同。

本次 `python -B -m unittest discover -s tests -v` 全部 150 项检查通过，包含微型数据的一轮 CPU 训练、验证选模、测试及保存恢复。44 个其他源码/配置文件哈希保持一致。未运行完整数据集重训练，未比较预测性能。
