# DHA-KT：供 Codex 使用的实现指南

> 状态：首版已按本规格完成；实际代码、运行命令、验收及研究阶段边界见 [dha_kt_usage.md](dha_kt_usage.md)。下文保留原开发契约，不将未开展的性能实验写为已完成。
> 日期：2026-10-19。模型定义与研究边界见 [dha_kt.md](dha_kt.md)。
> 本文约束仅作用于未来 DHA-KT 实现；现有 DiffuQKT、LinearQ、NewQ、NewQV2 的行为、用户配置和历史 checkpoint 必须保留。

## 1. 开始编码前的阅读顺序

1. [model_architecture.md](model_architecture.md)：包目录、模型注册、prepare_training、checkpoint 约定。
2. [references/diffuqkt/model.py](../references/diffuqkt/model.py)：作者原版加噪、x0 去噪、反向采样与共享 KT。
3. [src/model/diffuqkt/diffusion.py](../src/model/diffuqkt/diffusion.py)：当前可运行的数学实现及 no_grad 采样限制。
4. [src/model/newq_v2/model.py](../src/model/newq_v2/model.py)、[diffusion.py](../src/model/newq_v2/diffusion.py)：题目组件、独立 R 条件、三种 fusion、完整 2d 目标。
5. [newq_v2.md](newq_v2.md) 与现有 `tests/test_newq_v2*.py`：真实模式语义、梯度与恢复要求。
6. [src/model/diffuqkt/attention.py](../src/model/diffuqkt/attention.py)：inclusive 编码与 strict 历史检索。
7. `src/model/registry.py`、`src/utils/config.py`、`src/training/trainer.py`、`src/data/interactions.py`：实际接入接口。

本文不是要求复刻作者全部工程代码。references 只读，不导入其 run.py/model.py 作为新训练依赖。

## 2. 本次核对的现有能力与新增边界

### 2.1 已有能力

- NewQV2：两张独立题目代码表、去重多概念余弦融合、S=C_bar+D*V_bar、R=W_R*z_q。
- fusion：add / concat→d / concat→2d。
- legacy R 路由：off / condition / target / both。
- residual_condition_detach：仅独立 R 条件 detach。
- 通用字典 batch、训练/评估、注册、checkpoint、experiment_metadata。

### 2.2 必须新写的能力

- 学生级交叉拟合参考预测、行为轮廓 artifact 与真实事件去重。
- 轮廓维度 H 的扩散器、可微少步采样、按条件键稳定的候选库。
- DHA 独立 forward、pilot/final mixture、责任与历史记忆适配。
- 安全的题目查询/交互记忆分离接口。
- 分阶段训练控制及其恢复状态、诊断和评估缓存。

当前不存在可直接调用的旧 IRT/GCN/item_behavior 模块。不要根据早期聊天中的目录假定这些能力仍在，也不要为本文恢复无关旧分支。

## 3. 建议文件布局与复用方式

```text
src/model/dha_kt/
    __init__.py
    model.py              # DHAKT 组装、forward、loss、prepare_training、元数据
    item_encoder.py       # NewQV2 组件/融合的无扩散编码适配
    diffusion.py          # ProfileDiffusion：x0、日程、DDIM、可微采样
    candidates.py         # profile/legacy 后端、唯一键、局部噪声与候选生成
    evidence.py           # 稳定混合概率、责任、候选矩、记忆修正
    retriever.py          # 复用现有 attention 层，分离 Q 与 interaction 输入
    reference.py          # artifact 验证/读取、buffer、fold/view 查询，不启动教师训练
    diagnostics.py        # 模式/分歧/梯度/校准只读诊断
scripts/dha_kt/
    build_reference.py    # 训练集事件、学生折、参考预测、轮廓和 manifest
src/training/
    dha_kt_stages.py       # 如采用统一阶段调度，在此隔离策略，不复制整个 trainer
configs/models/dha_kt.yaml
tests/test_dha_kt_*.py
```

这是建议布局，允许合并小文件，但不把模型专属数学放进通用 utils。静态训练事件与 artifact 构造放 scripts；运行时载入放模型/数据适配，不改动现有清洗逻辑。

优先组合无扩散题目编码器与公共 attention 层。不要实例化完整 NewQV2 再闲置它的扩散器、对比支路和下游参数，也不要复制一套旧基线代码。若抽取公共题目编码函数，必须用回归检查证明 NewQV2 的参数名、初始化顺序、前向和旧 checkpoint 未变；无法保证时使用范围受控的适配实现，并明确来源。

DHA 需要独立 forward。仅覆盖 enhance_questions 后调用 QuestionKTBase.forward 无法完成本设计。

## 4. 题目与 R 的统一接口

推荐组件协议：

```python
ItemComponents = {
    'concept': Tensor[N, L, d],
    'variation': Tensor[N, L, d],
    'difficulty': Tensor[N, L, 1],
    'semantic': Tensor[N, L, d],
    'residual': Tensor[N, L, d],       # 关闭基础R时为零，不创建对应可训练表
    'original': Tensor[N, L, d_q],
    'canonical_skill_ids': LongTensor[N, L, K],
}
```

统一实现 `fuse(semantic, residual)`，原始与生成候选共用它。轮廓修正默认先作用于 R，再融合：

概念去重、padding处理与NewQV2一致。某有效题目若概念集合为空，S可以为零而R仍有效；对全mask概念集合须显式处理，禁止softmax后产生NaN，不能把“无概念”误判为整条padding。

```python
q0 = fuse(S, R)
delta_r = profile_to_residual(u_sample)   # H->d，无偏置
qm = fuse(S, R + delta_r)
```

不要在融合后再一律 `qm += R`；该规则对 concat 没有正确语义。不要将 d 维修正直接加到 2d 的完整向量上。三个 fusion 应分别按以下公式验收：

```text
add:        qm = S + R + delta_r
concat d:   qm = W_f [S; R+delta_r]
concat 2d:  qm = [S; R+delta_r]
```

### 4.1 配置语义，必须互不混用

| 字段 | 作用 |
| --- | --- |
| learned_residual_enabled | 是否创建/使用自由基础 R；false 不代表关闭生成行为修正 |
| generated_correction_enabled | 是否将采样轮廓投影到题目候选；false 时 profile 后端返回重复原始候选 |
| residual_fusion | add 或 concat |
| residual_concat_keep_2d | concat 时保留 2d，或经共享 W_f 回 d |
| profile_injection | residual_update 为默认；semantic_update 为独立实验 |
| profile_residual_condition | none/detach/trainable，控制 R 是否作为轮廓去噪条件 |
| candidate_backend | evidence_profile 或 legacy_question |
| legacy_residual_mode | 仅 legacy 后端：off/condition/target/both |
| legacy_residual_condition_detach | 仅 legacy 条件支路，不等于固定目标 |

首版 `learned_residual_enabled=false` 只支持 add；非 add 报错。去掉所有残差的实验需要同时关闭生成修正。`semantic_update` 定义为 F(S+DeltaS,R)，不是把同一个修正同时加到 S 和 R。

profile 后端要求 legacy_residual_mode=null，legacy_residual_condition_detach=false 为无效的中性兼容值；legacy 后端要求 profile_residual_condition=none、profile_injection=null，跳过 profile artifact 和 profile loss。基础 R 关闭时 profile_residual_condition 必须为 none。切换后端的专属配置必须显式覆盖这些字段；不合法时立即报错，不能静默忽略冲突。请求值与有效值分别记录。

### 4.2 legacy 候选适配的准确契约

| 路由 | 目标 | R 条件 | 候选组装 |
| --- | --- | --- | --- |
| off | S | 无 | fuse(S_hat,R) |
| condition | S | R 或 detached R | fuse(S_hat,R) |
| target | fuse(S,R) | 无 | 直接使用完整输出 |
| both | fuse(S,R) | R 或 detached R | 直接使用完整输出 |

对 target/both 不允许再融合 R。add 或 concat→d 的完整目标无法唯一拆出“生成 S/生成 R”；只有 concat→2d 有显式两块。

legacy target 中 m>=1 的生成候选在固定结构条件/噪声下不直接读取当前 R；R 通过训练目标影响生成器参数。不要给该局部生成路径增加“评估保留当前 R”的新行为。DHA 仍保留 Q0=F(S,R)，且查询/历史使用 Q0，因此 DHA 整体预测可以随 R 改变；旧 NewQV2 target 整模型的 R 不敏感测试不能直接套到 DHA。

legacy 适配用于验证同化机制，可先用完整 DDPM、训练/评估同采样路径。若另写少步适配，明确使用正确 DDIM 公式。旧 x0 目标仍为 live 题目张量，除非独立启用 teacher_target 并定义教师；不能将其结果称为固定参考轮廓版本。

## 5. 输入 batch 与 mask

现有 batch 是字典：

```python
problem_ids:    LongTensor[N,L]
skill_ids:      LongTensor[N,L] or LongTensor[N,L,K]
answers:        LongTensor[N,L]
student_ids:    LongTensor[N]
valid_mask:    BoolTensor[N,L]
prediction_mask: BoolTensor[N,L]
```

第一位置 prediction_mask=false，但它是有效历史交互：仍要构造该位置的答案条件记忆，供之后位置检索。不要用 prediction_mask 删除第一个历史事件或其责任；两项 KT loss 才使用 prediction_mask。

padding 的 Q、交互记忆与状态必须为零，不进行 profile 监督；责任在无效位置可显式归零，不能对全 -inf 行 softmax 后留下 NaN。

当前 collate_sequences 对所有 ndim>=2 的字段按时间截断。因此不要直接往 batch 加 `[N,M,d]` 这类非时间库张量。候选/profile 在模型内查询，或明确扩展 collate 的字段规则。

同一学生的所有窗口固定属于同一参考折。学生 ID 仅用于 fold 查询与评估分组，不送入预测器。

## 6. 参考 artifact 的构造与数据契约

### 6.1 构建流程

1. 读取现有训练划分与映射，确认真实事件 provenance，去除窗口重叠造成的重复。
2. 按学生固定分 F 折，保存映射与种子。
3. 首版使用下述固定 prefix_beta_skill 参考器，无教师拟合；若换成可训练参考器，只能在其余折训练并在合法参考集内部OOF预测，禁止使用看过 f 答案的 warm-start 权重。
4. 使用作答前输入得到参考 p_ref，固定等宽概率分箱。
5. 在合法参考学生中抽取 nested partial/dense 子集，计算 u、计数、mask，保存多个抽样视图。
6. 为验证/测试再构建 full_train 参考库，仅使用训练学生。
7. 写 manifest、校验和与可复现构造参数，验证集合互斥后才允许 DHA 训练。

事件去重不能采用 `(student,question,answer)`；重复练习可有同一三元组。若 processed 无唯一事件标识，builder 应明确读取训练学生过滤后的事件级来源，并核对与当前 split/过滤/长度策略的一致性；缺少可验证来源时终止 artifact 构建，不伪造事件身份。

严格复现时 reference scope 应与训练运行范围匹配。若 smoke/max_windows/max_seq_len 截断训练范围，要重建对应 artifact，或显式标记“使用完整训练参考库”的非标准 smoke 设置；不能隐瞒范围不同。

### 6.2 建议持久化字段

```text
manifest:
  schema_version, dataset_metadata_sha256, train_events_sha256
  split_seed, fold_seed, fold_count, student_fold_assignment_sha256
  training_scope, event_identity_scheme, deduplication_report
  excluded_students_per_view / reference_students_per_view
  predictor_type, predictor_config, predictor_checkpoint_sha256
  prefix_prediction_protocol, bin_edges, fixed_profile_scale, shrinkage_kappa
  partial_dense_sampling_protocol, draw_count, draw_seeds
  profile_table_shapes, tensor_file_sha256
  canonical_skill_policy, builder_version

tensors:
  partial_profile[view,draw,question,H]
  partial_counts[view,draw,question,H]
  partial_observed[view,draw,question,H]
  dense_profile[view,draw,question,H]
  dense_counts[view,draw,question,H]
  dense_observed[view,draw,question,H]
  student_fold_ids[max_student_index+1]
```

实际存储可以稀疏化。全训练库视图与训练排除折视图必须有明确的整数编号，不能让未知学生默认进入某个训练折。参考库选择独立于 model.train()/eval()：训练学生始终使用 exclude_fold，即便正在做 eval 诊断；valid/test 使用 full_train。可根据学生是否在保存的训练fold表中自动选择，并以显式 split context 校验：训练split有未知学生或valid/test与训练学生相交均报错。对未知题目使用无证据掩码与结构先验，并记录数量。

计数和 mask 是条件，dense 轮廓及其完整答案统计不能被原样塞回条件。去噪 loss 使用固定 dense 数据，生成的初始噪声与目标无关。

### 6.3 首版 builder 的确定默认

```yaml
fold_count: 3
draw_count: 8
predictor_type: prefix_beta_skill
predictor_beta_alpha: 1.0
predictor_beta_beta: 1.0
predictor_skill_prior_strength: 5.0
probability_bins: 8
profile_fixed_scale: 4.0
profile_shrinkage_kappa: 5.0
draw_0_dense_fraction: 1.0
draw_0_partial_fraction: 0.5
other_draw_dense_fraction_range: [0.8, 1.0]
other_draw_partial_fraction_of_dense_range: [0.25, 0.75]
eval_condition_draw: 0
```

prefix_beta_skill 对参考学生 i 的事件 t，先计算全局前缀先验 `p_global=(past_correct+1)/(past_count+2)`；各当前唯一概念 k 的预测为 `(past_correct_k+5*p_global)/(past_count_k+5)`，再对概念取均值。空概念用 p_global。预测后才更新该学生的全局及概念计数；多概念事件对全局计数只加一次。固定常数不拟合标签，不使用题号参数或校准器。这保证每个 p_ref 的输入只依赖该参考学生过去的事件，也避免高容量教师在训练样本上残差接近零的捷径；它的校准与分箱支持仍需诊断，不能当真实能力。

所有子集按学生抽样，A是B的子集；取整规则固定为 floor，再对非空参考池保证dense至少1人、partial至少1人且不超过dense。dense仅1人时允许A=B并记录退化视图，不伪造另一人。合法参考池为空则全missing，执行raw_only；训练学生总数少于fold_count时构造失败，用户可显式选择更小且至少2的fold_count。随机子集、比例和计数写入manifest。

训练每epoch固定一个合法condition draw；eval固定draw=0且仍读取partial一侧。不要为eval临时改成dense全部证据；若做full-evidence条件实验，必须单独配置并确保训练包含该条件范围。三个fold的所有轮廓使用相同固定bins与缩放。

### 6.4 缺失监督

- 目标 missing 不当作零真值；loss 只在 dense_observed 上计算。
- 全 missing 的条目跳过 L_diff，输出 `num_supervised_profile_keys` 供诊断。
- partial missing 可用零填充值，但必须同时输入 mask/count。
- 未知/无参考证据的生成是共享结构条件下的外推，不能称“已恢复真值”。
- artifact 文件缺失或哈希不匹配必须报错，不默默回退到 legacy。
- 有 artifact 但某题无证据时允许明确的 prior_only 或 raw_only 策略，保存有效模式；首版建议 raw_only，并在该题把生成修正置零。

## 7. ProfileDiffusion 的实现契约

### 7.1 参数化

采用条件 x0 预测，与 DiffuQKT 原模块一致：

```python
predict_x0(noisy_profile, diffusion_step, structure_condition,
           partial_profile, counts, observed, residual_condition=None)
```

profile_dim=H、condition_dim、time_embedding_dim、hidden_dim 独立声明。不能为了复用类而把 H 伪装为概念宽度 d，也不能照抄 `3*d+1`。

注册 float32 日程 buffer，计算日程可使用 float64。posterior 与作者 fixed_large 方差属于完整 DDPM 的明确选择；DDIM eta=0 的路径不使用同一方差公式，需分开记录。

默认建议 cosine schedule，确保 alpha_bar 在所有步骤合法且终端接近零但不为引起除零的精确零。记录终端 SNR。保留 linear 作为对照，不修改旧模型日程。

### 7.2 DDIM 跳步公式

给当前步骤 s 与下一步骤 r<s，a_s=alpha_bar[s]，a_r=alpha_bar[r]；最后输出对应 a_r=1：

```text
eps_hat = (x_s-sqrt(a_s)*x0_hat)/sqrt(1-a_s)
sigma   = eta * sqrt((1-a_r)/(1-a_s) * (1-a_s/a_r))
x_r     = sqrt(a_r)*x0_hat
          + sqrt(max(1-a_r-sigma^2,0))*eps_hat
          + sigma*z
```

最后一步直接返回 x0_hat。时间网格包含最高噪声步，不产生重复索引。训练与 eval 共用一个 sampler 实现，仅由调用上下文决定是否构建梯度，不在核心函数上永久加 no_grad。

不得把 s→s-1 的 DDPM 相邻系数用于 s→r 的跳步。应为跳步公式编写独立数值检查。

### 7.3 两条训练用途

- L_diff：对固定 dense x0 抽一个 s，加解析高斯噪声，预测 x0；固定目标和 noisy-input 源均来自无梯度 artifact。
- KT 候选：从独立起始噪声执行实际反向采样，不读取本次监督的干净目标；joint 阶段梯度经过实际采样步骤。

不要从 dense target 加噪后做一次恢复，便称之为与部署相同的 KT 候选。若因显存采用冻结生成器实验，应记录 `generator_frozen`，不能声称 end-to-end。

## 8. 候选键、随机数与缓存

建议规范键：

```text
(reference_view, profile_condition_draw, question_id,
 sorted_unique_skill_set, candidate_id,
 backend, route, fusion, sampler_version, candidate_seed)
```

训练中固定同一 epoch 的 condition draw 和起始噪声标识；同一 forward 按唯一键生成一次再 gather 到事件。参数更新后同一键的输出可变，身份约定不意味着权重更新后向量固定。

评估键还包含 checkpoint/manifest/config 指纹。每题候选噪声从稳定哈希派生，不使用 Python 随机 hash，不依赖 batch 出现顺序或全局 RNG 消耗数。传入 generator 必须被尊重：可使用其确定的初始 seed 作为 seed 派生输入，而不是无条件 torch.randn 消耗全局状态。

同题在不同事件携带不同概念集合时，必须用概念集合区分完整候选；保留当前事件视图语义，不默默改为全题概念并集。计数/profile 可以按题共享，结构条件仍按事件概念视图。

| 可缓存内容 | 训练跨 step | 固定 checkpoint 评估 |
| --- | --- | --- |
| 参考轮廓/计数/mask、稳定噪声 | 可以 | 可以 |
| 含当前参数的 C/S/R/投影 | 不可以 | 可以 |
| 完整生成候选及其图 | 不可以 | 可以，仅 no_grad 值 |

训练缓存范围默认仅单个 forward。冻结扩散器不足以允许缓存完整 Q*，因为题目条件、R、W_f、W_u 仍可能更新。

## 9. 安全的两遍检索接口

现有 KnowledgeRetriever 内部会构造 interaction=question+answer，并把 question 送入当前位置查询。DHA 需要独立接口：

```python
forward_with_interactions(question_prior, interaction_input, valid)
```

推荐在 DHA 自己的 retriever.py 组合/继承已有层；若增加公共方法，现有 forward 应继续委托相同数学路径，且旧输出不变。

```python
def forward_with_interactions(self, q_prior, x_history, valid):
    questions = self.question_encoder(q_prior, q_prior, q_prior, valid)
    interactions = self.interaction_encoder(x_history, x_history, x_history, valid)
    return self.retrieval(questions, questions, interactions, valid, strict=True)
```

第一遍：q_prior=Q0，x_history=Q0+answer_embedding。

第二遍：q_prior 仍为 Q0，x_history 为经过候选责任修正的 X1。

两遍引用同一个 retriever 实例和同一个 predictor。不能分别构造两套 F 后仅叫“双通道共享”；不能把 mu、X1、责任或当前答案放进 question_encoder/query/key 输入。当前 strict retrieval 已隔离当前位置答案，不额外 shift。

## 10. 稳定混合概率、责任与证据

设候选 logits 为 ell `[N,L,M+1]`，固定 log_pi 同形可广播。

```python
log_pc = F.logsigmoid(ell)
log_pw = F.logsigmoid(-ell)
log_p = torch.logsumexp(log_pi + log_pc, dim=-1)
log_1mp = torch.logsumexp(log_pi + log_pw, dim=-1)
mixture_logit = log_p - log_1mp
```

对 mixture_logit 做 BCEWithLogits，可与返回 probabilities=exp(log_p) 保持一致。禁止用平均 logits 代替概率混合。

责任：

```python
log_lik = torch.where(answers[...,None].bool(), log_pc, log_pw)
r = (log_pi + log_lik).softmax(dim=-1)
r = r.detach()                               # 默认
r = r * valid[...,None]
mu = (r[...,None] * candidates).sum(dim=-2)
v = (r[...,None] * (candidates-mu[...,None,:]).square()).sum(dim=-2)
x0 = q0 + answer_embedding(answers)
x1 = x0 + adapter(torch.cat([mu-q0, v], dim=-1))
x1 = x1 * valid[...,None]
```

责任 SG 不等于把 mu/v/candidates 全部 detach。默认 adapter 是无偏置 Linear(2*d_q,d_q)，零初始化。若以后加入 surprise，需要独立字段、维度声明和控制组；默认关闭。

`score_all_candidates` 训练时必须在候选维共享预测头的 dropout mask，例如对 `[N,L,M+1,hidden]` 广播 `[N,L,1,hidden]` 掩码。pilot 与 final 都采用这一规则，二者可各抽一组掩码。不能直接 flatten 后调用旧 Sequential 中普通 Dropout，让同一位置不同候选遭遇独立随机扰动。可在 DHA 内实现共享候选掩码的头并复用同一实例，不修改旧模型预测头。候选分歧与退化验收使用eval/dropout=0。

对被配置为 raw_only 的无参考题目，候选可全部置为 Q0，r 仍可形式化计算，mu=Q0、v=0，使适配器修正为零。

## 11. forward 伪代码与输出

以下表达 evidence_profile 后端 joint 阶段的时序，函数名不是当前已有 API；legacy后端通过其自身候选/表示损失接口分派，不查询不存在的profile库：

```python
def forward(self, batch, generator=None):
    valid = batch['valid_mask']
    targets = valid & batch['prediction_mask']
    components = self.item_encoder(batch)
    q0 = components['original']

    # 按学生的训练fold归属和显式数据split取参考，不以self.training判断。
    # 对train数据做eval诊断时，仍然使用exclude_fold。
    refs = self.reference_bank.lookup(batch, data_split=self.data_split)
    candidates = self.candidate_builder(components, refs, generator=generator)
    # candidates[...,0,:] == q0；生成不读取batch当前/未来answers。

    x0 = (q0 + self.answer(batch['answers'])) * valid[...,None]
    h0 = self.retriever.forward_with_interactions(q0, x0, valid)
    ell0 = self.score_all_candidates(h0, candidates)
    pilot_logit = mixture_logit(ell0, self.log_prior)

    r = responsibilities(ell0, batch['answers'], self.log_prior,
                         stop_gradient=True, valid=valid)
    mu, variance = candidate_moments(candidates, r)
    x1 = x0 + self.adapter(cat(mu-q0, variance))
    x1 = x1 * valid[...,None]

    h1 = self.retriever.forward_with_interactions(q0, x1, valid)
    ell1 = self.score_all_candidates(h1, candidates)
    final_logit = mixture_logit(ell1, self.log_prior)  # 使用pi，不使用r
    kt = bce_with_logits(final_logit[targets], answers[targets].float())
    pilot = bce_with_logits(pilot_logit[targets], answers[targets].float())

    if self.training:
        den = self.profile_diffusion_loss(refs)       # 按唯一监督键
        total = kt + self.pilot_weight*pilot + self.diffusion_weight*den
    else:
        den, total = None, kt

    return {
        'logits': final_logit,
        'probabilities': final_logit.sigmoid(),
        'states': h1,
        'loss': total,
        'kt_loss': kt,
        'contrastive_loss': None,
        'diffusion_loss': den,
        'pilot_loss': pilot if self.training else None,
        # 可另返回标量diagnostics，不要默认存整批大候选图。
    }
```

当前第一位置不计损失；可无历史预测用于构造下一步证据。第二遍的当前责任不能进入 final_logit 当前位置：时间隔离由安全 query/value 接口和 strict 检索共同保证。

模型评估时 loss=kt，辅助 loss=None。不能将 pilot 值塞进 contrastive_loss 列。无有效 profile 监督时返回图安全的零 den，并报告监督条目数；整个 batch 无预测目标应沿用现有明确报错。

## 12. 注册、配置与现有 trainer 的适配

### 12.1 注册

- `src/model/registry.py`：新增独立 `dha_kt`，不替换 newq_v2。
- `src/model/__init__.py`：需要公开类时导出 DHAKT。
- `src/utils/config.py` 的 MODEL_CONFIGS：添加模型 YAML；CLI choices 由这里派生。
- `configs/models/dha_kt.yaml`：新配置，不修改当前 NewQV2 YAML。

构造器实际接收 `num_problems,num_skills,**model,**loss`。因此下面嵌套配置需要由 DHA 构造器解析字典，或配置层明确 flatten；不要写 YAML 后让现有 trainer 意外 TypeError。

### 12.2 建议配置草案

以下是新模型待实现的 schema，不是当前有效命令：

```yaml
model:
  embedding_dim: 128
  num_heads: 2
  dropout: 0.4
  attention_embedding_dim: 8
  residual_embedding_dim: 8
  attention_scale: 2.0
  attention_temperature: 1.0
  learned_residual_enabled: true
  residual_fusion: add
  residual_concat_keep_2d: false
  generated_correction_enabled: true
  candidate_backend: evidence_profile
  profile_injection: residual_update
  profile_residual_condition: none
  legacy_residual_mode: null
  legacy_residual_condition_detach: false
  generated_candidates: 3
  raw_prior_mass: 0.5
  responsibility_stop_gradient: true
  responsibility_temperature: 1.0
  surprise_input: false
  profile:
    artifact_path: null             # null 按训练数据位置自动查找 <数据集目录>/dha_kt/reference；必须有真实参考库
    manifest_sha256: null           # 首次载入确定并保存
    bins: 8
    fixed_scale: 4.0
    shrinkage_kappa: 5.0
    no_evidence_policy: raw_only
  diffusion:
    prediction_type: x0
    schedule: cosine
    steps: 100
    sampler: ddim
    sampling_steps: 8
    ddim_eta: 0.0
    time_embedding_dim: 64
    hidden_dim: 128
  candidate_sampling:
    seed: 2026
    train_draw_policy: epoch_fixed
    eval_condition_draw: 0
    train_cache: forward_only
    eval_cache: checkpoint_keyed
loss:
  contrastive_weight: 0.0
  pilot_weight: 0.5
  diffusion_weight: 0.1
```

配置中的值是起始设计，不是验证后的最佳参数。保留公共 `temperature/contrastive_negatives` 的兼容接收及说明；首版 DHA 对比权重非零时应报错或明确提供另一个实验模式，不能悄悄重用身份负样本。

旧公共顶层 beta/diffusion_steps 若被合并进来，须由 schema 明确识别冲突；不要同时存在两套不同日程但只记录其中一套。所有请求值、有效值、d_q、目标宽度和参数量进入 experiment_metadata。

### 12.3 训练器最小变更

当前 run_epoch 硬编码 loss/kt_loss/contrastive_loss/diffusion_loss 四键。DHA 保留四键，无 CL 返回 None。若要聚合 pilot_loss 和诊断，扩展通用可选标量聚合，保持旧 CSV/历史记录兼容；不能只在 model 中增加字段就假定日志会显示。

prepare_training(dataset) 发生在 optimizer 构造前，可验证并载入 artifact，返回恢复所需的构造参数。不能在 forward 中启动参考教师训练或访问验证数据。复杂阶段逻辑放独立训练策略，避免把所有情况写进公共 trainer 的 model_name 分支。

现有训练开始通常一次性构造 Adam，并按 requires_grad 过滤参数；仅提前建立全部模块不足以覆盖后续解冻。应在阶段冻结前把全部将来训练的参数收进Adam，再用requires_grad/损失控制阶段；或切换阶段时对未纳入参数执行add_param_group并保留已有状态。不得重复添加参数。恢复时重建同样的组及顺序，并验证新解冻参数确实发生数值更新。

显式增加可选的数据split上下文协议，例如训练/评估入口在调用DHA前执行 `set_data_context('train'/'valid'/'test')`，lookup同时核对保存的学生fold归属。普通model.eval()不修改此上下文；训练集只读诊断必须设置train。其他模型无需实现此hook，保持原行为。若采用纯学生归属推断，也必须在入口核对split和artifact的成员集合，禁止仅依赖self.training切库。

## 13. 阶段训练与 checkpoint

建议阶段状态为 base_warmup / diffusion_pretrain / joint；参考构造是离线步骤。每个阶段保存配置、epoch、global_step、候选 draw 与噪声策略。

| 阶段 | 参与更新 | 输入路径 |
| --- | --- | --- |
| base_warmup | 题目编码、共享 KT、预测头 | 原始Q；关闭同化修正，仅raw预测BCE |
| diffusion_pretrain | 轮廓去噪器；结构条件可固定为 warmup 快照 | partial→dense，仅L_diff |
| joint | 题目编码、候选投影、适配器、共享 KT、去噪器 | 实际采样＋两遍编码＋三项损失 |

参考 p_ref 及统计目标不随阶段更新；扩散条件可依赖当前可训练结构嵌入，不影响目标坐标固定这一约定。联合阶段 prior 与责任温度先固定。

现有 checkpoint 恢复会按 model_kwargs 构造模型后立即 load_state_dict，不重新调用 prepare_training。因此：

- buffer 的形状参数（参考视图数、draw 数、H、学生索引表长度等）必须写入 model_kwargs。
- 构造器能以这些参数建立空 buffer，再从 state_dict 恢复，评估不依赖原 artifact 路径存在。
- 参考库的路径解析/存在性校验作用于首次训练prepare阶段：artifact_path=null 时自动查找数据集目录下的 dha_kt/reference；恢复时有完整bank schema与state_dict可不读取旧路径，不能在构造器无条件检查路径存在后阻断自包含恢复。
- 参考条件、计数、mask、fold映射等必须作为buffer或其他明确受checkpoint保护的状态。
- 若大库采用外部不可变 artifact，checkpoint 必须保存 manifest 和文件哈希，恢复缺文件必须明确报错；不能悄悄重建。首版优先自包含必要参考buffer。
- 保存阶段与 optimizer 状态，恢复后不能从 joint 静默回到 warmup。
- 不持久化评估缓存的 autograd 图；缓存重建键必须包含权重快照指纹。
- 历史模型用原来的构造器与默认值严格恢复，DHA 的字段不能覆盖旧权重。

训练split与valid/test split的参考库视图不同是交叉拟合协议的一部分，应保存并报告；模型train/eval模式本身不决定参考视图。候选反向采样算法、结构/轮廓含义及两遍时序保持一致。

## 14. 分步开发计划与验收

### A. 固定候选的同化骨架

先用手工/确定性候选验证两遍共享、稳定 mixture、r、矩、safe retriever。此阶段不声称已完成扩散版。

验收：手算0.8/0.4候选、pi=0.5时，答错责任0.25/0.75，答对2/3与1/3；改当前标签不改当前预测；全部候选相同退化。

### B. NewQV2 题目与 legacy 适配

实现三 fusion、基础 R 开关与四旧路由的候选组装。先完整反向采样保持数学一致，确认 target/both 不重复融合。

验收：按公式检查每条路径、维度、R 条件 detach、当前 R 的 eval 数值依赖、旧模型不受影响。

### C. 参考证据与 profile 生成器

实现学生折、事件去重、固定坐标轮廓、manifest、x0 去噪和可微 DDIM。profile artifact 缺失时拒绝正式模式。

验收：修改被排除折/验证/测试答案不改变对应训练参考artifact；missing监督正确；固定噪声下DDIM数值匹配；真实小范围数据可构建并恢复。

### D. 联合训练与完整保存恢复

接入阶段策略、Adam、日志、checkpoint、自包含buffer、独立评估与只读诊断。

验收：小数据训练→验证→best/last→独立恢复评估完整跑通，loss/概率恢复一致；性能结论仍需完整实验。

### E. 机制与扩散必要性实验

在同划分/同预算上比较 r→pi、普通门控、surprise-only、非扩散候选、R开关与注入位置、M和采样步数。至少多种子，并报告预训练及部署成本。

## 15. 必须实现的检查

1. **因果性**：固定候选噪声、eval或dropout=0；改 y_t 不改 p_t，改未来答案不改过去预测及责任；过去答案允许影响未来。
2. **query/value隔离**：审计第二遍 question/query/key 无当前答案依赖，仅 interaction values含答案条件编码。
3. **首事件**：prediction_mask排除它，但其答案仍影响后续位置。
4. **mask/padding**：无效事件不进入监督、责任、参考计数，输出无NaN；混合单/多概念批次正确。
5. **融合公式**：三fusion共享参数、所有候选同维；先融合再计算矩；2d的答案/状态/预测宽度正确。
6. **旧R路由**：off仍保留R；target/both只融合一次；条件detach只断条件，不改变前向值或冻结R。
7. **梯度**：r无反传，payload保留；Lpilot直接训练pilot；实际sampler可微。evidence_profile的固定目标与noisy源无可训练依赖；legacy_question保留live目标，除非另行定义teacher_target。零初始化导致首步零梯度与None需区分。
8. **稳定概率**：极端logits、pi归一、无效行与混合logit正确，不能平均logits。
9. **采样**：DDIM跳步、终端SNR、时间网格、末步x0、固定按键噪声、CPU/CUDA可复现边界。
10. **缓存**：改变batch顺序/大小不改eval候选；optimizer更新后训练不复用旧向量；条件概念视图不同键不同。
11. **数据隔离**：学生折、重叠去重、reference scope、训练/验证/测试边界，构造manifest可追溯。
12. **恢复**：删去原artifact路径后自包含checkpoint仍可评估；改当前YAML不改保存结构；旧模型state_dict严格恢复。
13. **计量**：实际参数量含题目表/2d层/扩散/适配器，统计预训练和采样耗时。现有NewQV2的4倍预算不能被新模型默默绕过；DHA可设独立显式上限并报告口径。

不要求为纯文档运行训练。未来实施时应按阶段运行必要检查，再进行小范围流程验证；无需在每个小改动后重复所有完整训练。

## 16. 完成代码前不能宣称的结果

- 加入扩散就必然改善题目表示或稀疏题目性能。
- 采样方差已经等于真实认知不确定性。
- r拟合当前答案更好就证明历史追踪更准确。
- profile交叉拟合意味着所有模型参数从未看过当前学生标签。
- legacy_target/both在评估时仍直接保留当前自由R。
- cache或no_grad冻结的候选路径仍是端到端生成器联合学习。
- 通过有限smoke即可证明完整训练稳定或达到顶会水平。

模型研究的判断标准是：答案条件候选责任带来稳定的未来预测增益，并且扩散候选的作用超过更简单、成本匹配的替代方案。
