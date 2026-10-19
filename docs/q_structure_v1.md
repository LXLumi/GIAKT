# Q 结构设计与版本记录：概念加权融合、修正残差及 NewQ V2 编码规格

状态：第 1～11 节保留历史讨论，第 12 节记录 NewQ V1，第 13～14 节保留原 V2 规格与验收。第 15 节保留残差扩散插件规格，第 16 节记录插件实现核验，第 17 节增加独立的条件输入 detach 对照，第 18 节增加残差相加/拼接与维度控制。历史限制以后续用户要求的补充为准，不创建新模型。默认相加且关闭插件时保持第 13 节路径。实际记录见 [newq_v2.md](newq_v2.md)，V1 见 [newq.md](newq.md)。

以 GIAKT 原始 DiffuQKT 为基线。已有扩散效果验证实验与本次结构研究分别管理；后续结构实现使用独立模型文件与配置。

## 1. 已确认方向与解释边界

- 将概念等权平均改为题目条件下的概念加权融合。
- 增加题目修正残差，容纳已标注概念及当前难度机制未解释的题目差异。
- 第一版保留原有标量难度，先检验上述两项变化的作用。
- 下文的权重参数化、残差约束及扩散接法均为候选，继续讨论后再实现。

等权平均对每道题固定了概念等贡献的假设，缺少对题目差异的自适应表达。等权平均仍是合理的对照与新模型的特殊情形；学习得到的权重也不自动成为经过教育学验证的概念重要性。

## 2. 初始 Q 的建议公式

设题目 ID 为 q，本次观测的非重复原子概念集合为 K。若题目与概念集合映射固定，将 Q(q,K) 简写为 Q_q。

当前多概念基线为：

$$
Q^{base}_{q,K}=\frac{1}{|K|}\sum_{k\in K}C_k
+D_q\frac{1}{|K|}\sum_{k\in K}V_k,
\qquad D_q=\sigma(d_q).
$$

第一版建议为：

$$
\boxed{Q_{q,K}=\sum_{k\in K}\alpha_{qk\mid K}(C_k+D_qV_k)+R_q}
$$

其中：

$$
\alpha_{qk\mid K}\geq0,\qquad
\sum_{k\in K}\alpha_{qk\mid K}=1,\qquad R_q\in\mathbb R^d.
$$

第一版候选使用同一组权重融合 C 与 V，使概念身份向量与变化方向具有一致的贡献分配：

$$
\bar C_{q,K}=\sum_k\alpha_{qk\mid K}C_k,\quad
\bar V_{q,K}=\sum_k\alpha_{qk\mid K}V_k,\quad
Q_{q,K}=\bar C_{q,K}+D_q\bar V_{q,K}+R_q.
$$

权重为 1/|K| 且残差为零时，退化为原始 Q。单概念事件的权重恒为 1，其结构改善来自残差。

## 3. 各组成部分的语义

| 组成 | 操作性含义 | 不能自动推出的解释 |
| --- | --- | --- |
| C_k | 跨题目共享的概念表征 | 嵌入维度对应固定认知属性 |
| D_q V_k | 当前机制下的题目相关概念偏移 | D_q 已是经过校准的测量学难度 |
| alpha | 当前题目及标签视图下，各概念对题目表征的相对贡献 | 必要掌握程度、因果贡献或学生薄弱程度 |
| R_q | 共享概念及当前难度机制未解释的题目特异性偏差 | 具体维度已对应题型、语言或推理步骤 |

题型、表述、任务情境、步骤要求、概念组织及遗漏标签都可能影响作答。但仅有题目 ID、标签与答题记录时，残差只能吸收这些因素可能产生的统计效应，不能识别究竟是哪一种因素。

残差也可能补偿概念或难度分支的拟合不足，与概念部分不自动统计独立或几何正交。

给定 C、V、D 后，加权融合位于概念级向量 C_k+D_qV_k 的凸包内；残差允许题目表示偏离这个凸包。两者提供不同的表达能力。

## 4. 权重的候选参数化

建议权重依赖题目与概念，不依赖当前答案或学生状态，使 Q 保持题目属性的含义。

一种简单候选为低维题目向量 e_q 与双线性匹配：

$$
s_{qk}=\frac{(W_Qe_q)^\top(W_KC_k)}{\sqrt{d_a}},\qquad
\alpha_{qk\mid K}=\frac{\exp(s_{qk}/\tau)}{\sum_{j\in K}\exp(s_{qj}/\tau)}.
$$

归一化只包含当前集合中的有效概念，padding 权重为零。概念排列不影响融合结果。

简单线性拼接打分 s(q,k)=w_q^T e_q+w_c^T C_k 不适合实现题目条件权重：在同题 softmax 中，题目项会被抵消。双线性匹配或带非线性的联合打分可避免该问题。

softmax 不保证所有标注概念均被充分使用，也不保证低权重概念不是必要概念。是否引入平滑约束，应依据数据与消融决定。

## 5. 残差的候选参数化与约束

最小候选为：

$$
R_q=W_Re_q.
$$

残差不必由 D_q 再统一缩放，因为未解释的题目差异未必与当前难度参数成比例。共享 e_q 可减少参数，但使权重与残差存在参数耦合。

第一版可使用较低维 e_q 和小幅残差初始化。残差可能绕过共享概念分支、主要记忆训练题目，一种候选收缩目标为：

$$
L_R=\frac{1}{|I_{train}|d}\sum_{q\in I_{train}}\|R_q\|_2^2.
$$

I_train 只包含训练中出现的非 padding 题目。该目标按题目计权，不能用重复交互的均值冒充相同目标；具体计算方式与权重待讨论。

低维来源、小幅初始化与残差惩罚不保证分解唯一。第一版暂不强制残差与概念向量正交，因为尚无证据支持该几何约束。

## 6. 本版保留的建模假设

本版放宽概念等权假设，增加题目特异性的表达空间，但仍采用标量难度与加权加性融合。权重打分的非线性不等于建立了共同必需、相互替代或高阶概念交互机制。

本版研究问题是：题目条件的概念贡献分配与受约束的题目修正，能否提升 KT 预测性能。概念联合要求属于后续扩展。

## 7. 扩散接口与标签视图

本轮先确定初始 Q，扩散接入位置待讨论。当前 DiffuQKT 训练时从 Q 加噪，评估时通过条件从噪声生成。若只将残差加入训练目标，却不使其题目信息进入评估条件或输出通路，就不能保证生成结果保留该修正。

候选接法包括将题目特征引入生成条件、在生成后加入残差，或将加权概念与残差合成为现有维度的条件向量。各接法须说明训练目标、条件与输出的对应关系。

若同一题目 ID 在不同记录中具有不同标签，先按实际观测集合构造 Q(q,K)，不能直接把跨记录标签并集解释成单次作答必须共同使用的概念。R_q 可在同题的不同标签视图间共享。

仅由 ID 学习 e_q 时，对未见新题没有可靠的题目特异性估计；未来若评估冷启动，需要约定共享分支与残差的回退方式。

## 8. 最小结构消融

| 设置 | 概念融合 | 残差 | 验证目的 |
| --- | --- | --- | --- |
| A | 平均 | 无 | 原始 Q 对照 |
| B | 学习权重 | 无 | 概念贡献分配的收益 |
| C | 平均 | 有 | 题目修正的收益 |
| D | 学习权重 | 有 | 两者是否互补 |

使用一致的数据划分、KT 主干、预测头、训练预算与明确的增强策略。B 仍需题目向量用于查询，应记录实际参数量并考虑题目参数容量的对照。

分别检查整体、单概念、多概念与低频题目的表现。单概念样本的改善不能归因于多概念权重；权重可视化不能单独证明教育学解释。

## 9. 待确定选择

1. 权重采用双线性匹配还是非线性联合打分，题目向量维度如何控制。
2. 概念与变化方向是否沿用同一组权重。
3. 残差采用线性投影还是其他简单形式，是否使用独立残差惩罚。
4. 结构信息进入扩散条件还是经过输出通路。

定稿后新增独立模型文件、专属配置与验证，原始 DiffuQKT 作为结构研究基线。

## 10. 依据

- 当前项目基线：src/model/diffuqkt/base.py 的概念均值与 Q=C+D*V；src/model/diffuqkt/model.py 和 diffusion.py 的重构及评估生成路径。
- [Neural Cognitive Diagnosis for Intelligent Education Systems](https://ojs.aaai.org/index.php/AAAI/article/view/6080)：讨论概念、难度、区分度与文本等题目因素。
- [Pre-Training Multi-Concept Question Embeddings for Knowledge Tracing](https://www.mdpi.com/2076-3417/15/7/3654)：包含由概念表征加权构造题目表征的相关工作。

本稿是第一版可验证的设计方向，尚未完成系统新颖性对比，也尚无该结构的预测性能实验。

## 11. 梯度传播、维度与参数预算（2026-10-18 补充讨论）

### 11.1 用户给出的参数约束

新模型总参数量以原始 DiffuQKT 为参照，目标控制在基线的 3 倍以内，4 倍作为不可超过的硬上限。这里比较的是新模型总量与基线总量之比，包含保留的扩散、KT 主干与预测头。

当前阶段继续讨论方案；以下参数化与默认维度均为建议，不代表已经实现或完成性能验证。

### 11.2 权重通过任务梯度端到端学习

alpha 是注意力网络每次前向计算得到的中间张量，不是为每次作答单独注册的模型参数。优化器更新的是题目嵌入表及打分网络参数。

第一版建议保持端到端反向传播：任务损失经 Q 回到 alpha、概念向量、变化向量与残差，再回到注意力网络和题目向量。训练时不对 alpha 使用 detach，也不将其计算置于 no_grad 中。正常反传已能够学习权重，不需要另行手写梯度规则或为 alpha 提供额外人工标签。

设 Z_k=C_k+D_qV_k，S=sum_k alpha_k Z_k，则：

$$
\frac{\partial Q}{\partial s_j}
=\frac{\alpha_j}{\tau}(Z_j-S).
$$

上游任务梯度通过该关系训练打分函数。概念向量若同时用于 key 与融合 value，会同时收到这两条路径的梯度；共享 e_q 会同时收到注意力路径与残差路径的梯度。

第一版可先沿用当前训练器的梯度裁剪及常规优化方式，温度先固定为 1。额外冻结、梯度缩放或模块学习率调整，应以训练信号为依据再引入。

单概念事件的 softmax 恒为 1，其注意力打分梯度为零，这是结构性质。该事件仍能训练概念、变化、难度与残差。单概念数据不能用于证明注意力权重分配的效果。

### 11.3 128 维 Q 与低维题目参数是两件事

当前基线 C_k、V_k 与输出 Q 是 128 维，但没有为每道题保存一个独立的 128 维题目 ID 向量。每题独立的难度表只存一个标量。新增 N_q 行、128 列的嵌入表因此会显著增加参数。

建议保留 C、V、Q、残差输出与 KT 主干的 128 维空间，只将新增的题目特异性参数 e_q 设置为 r 维。e_q 直接在低维表中学习；不是训练完成后才升维，也不是先把完整 Q 压缩到 r 维。

为简化候选打分网络，可以直接将 e_q 作为低维 query：

$$
E\in\mathbb R^{(N_q+1)\times r},\quad
e_q=E[q],\quad
W_K\in\mathbb R^{r\times128},\quad
W_R\in\mathbb R^{128\times r},
$$

$$
s_{qk}=\frac{e_q^\top W_KC_k}{\sqrt r},\qquad
\alpha_{qk\mid K}=\operatorname{softmax}_{k\in K}(s_{qk}),\qquad
R_q=W_Re_q.
$$

两个投影均不设偏置，供所有题目共享。注意力在 r 维空间匹配，残差由共享线性映射生成 128 维输出，最终 Q 仍为 128 维。

线性残差集合受到至多 r 维子空间的约束；完整 Q 还包含共享概念与变化分支，不因此被限制为至多 r 维。

### 11.4 当前基线实测与候选结构估算

使用当前两个数据集的 processed/metadata.json、模型配置和实际 DiffuQKT 构造器，在 giakt-main 的 PyTorch 2.4.1+cu124 中统计所有模型参数：

| 数据集 | 题目数 | 概念数 | 输出维度 | 基线参数量 |
| --- | ---: | ---: | ---: | ---: |
| ASSIST09 | 17,645 | 123 | 128 | 546,165 |
| ASSIST17 | 2,210 | 101 | 128 | 525,098 |

按照上述精简候选，新增参数恰为：

$$
\Delta P=(N_q+1)r+128r+128r=(N_q+257)r.
$$

下面是基线实测量加上述结构参数的估算，不是已实现的新模型实测值；尚未计入待定的扩散条件接口变化。

| r | ASSIST09 新增参数 | ASSIST09 总参数 / 倍率 | ASSIST17 新增参数 | ASSIST17 总参数 / 倍率 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 143,216 | 689,381 / 1.262 | 19,736 | 544,834 / 1.038 |
| 16 | 286,432 | 832,597 / 1.524 | 39,472 | 564,570 / 1.075 |
| 32 | 572,864 | 1,119,029 / 2.049 | 78,944 | 604,042 / 1.150 |
| 64 | 1,145,728 | 1,691,893 / 3.098 | 157,888 | 682,986 / 1.301 |
| 128 | 2,291,456 | 2,837,621 / 5.196 | 315,776 | 840,874 / 1.601 |

建议第一版从 r=16 开始，r=32 作为容量对照，保持所有输出维度为 128。r=64 在 ASSIST09 上已经超过 3 倍目标；r=128 超过该数据集的 4 倍硬上限。

后续实现时，应对含全部条件编码与额外模块的最终模型重新统计参数，不能仅凭本表代替总量校验。

### 11.5 本次核验范围

已实际构造当前基线并统计参数；用独立的小型张量计算核验多概念打分参数、题目向量、残差投影、C、V 均可获得有限非零梯度，同时核验单概念打分梯度为零。该核验只说明候选计算图可微，不表示新模型已经实现或获得预测增益。

反向传播机制参考：[PyTorch 自动微分说明](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

## 12. NewQ 第一版实现记录（2026-10-18）

本节是 V1 历史实现记录，用于复现现有 newq 实验。V2 编码以第 13 节为准，不能继续采用本节的共享代码及 C_bar+R 扩散条件。

本次实现选择第 11.3 节的精简双线性参数化：默认 r=16，attention_temperature=1，共用同一组 alpha 融合 C、V，残差为无偏置线性映射 R_q=W_R*e_q。题目向量以标准差 0.02 的正态分布初始化，任务梯度端到端传播；有效概念按当前观测集合去重，padding 权重为零。

根据本次“初始 Q 之后与基线一致”的范围，保持基线的扩散网络、对比学习、预测、三项联合损失及其默认系数，不增加独立 L_R。为让残差在评估采样中有条件入口，采用第 7 节的同维条件方案：概念条件为 C_bar+R_q，难度条件为 D_q，重建目标为 Q=C_bar+D_q*V_bar+R_q；生成后不再另加一次残差。

模型位于 src/model/newq/model.py，继承 DiffuQKT，只新增参数并重写 question_components；配置为 configs/models/newq.yaml，通过 --model newq 选择。最终模型统计包含所有保留模块，并在构造时拒绝超过同规模基线 4 倍的配置。具体参数、验证范围与命令见 [newq.md](newq.md)。

## 13. NewQ V2：目标 1、2 的确定性编码规格（2026-10-18）

### 13.1 任务范围、命名与实施原则

本轮只落实两项目标：

1. **目标 1：将题目残差移到扩散输出端。** 扩散只对共享概念语义部分加噪、重构和采样；训练及评估均在扩散结束后加一次残差。
2. **目标 2：概念权重与残差使用独立题目代码，并采用尺度明确的归一化注意力。** 默认 8 维注意力代码加 8 维残差代码，保留 128 维输出。

本轮采用独立版本名称：模型标识 `newq_v2`，类名 `NewQV2`。保留 `newq` 的现有模型、专属配置、测试语义与历史 checkpoint。禁止直接改写 V1 类来冒充 V2，也禁止将旧 checkpoint 静默加载进不同结构。

V2 继承 `DiffuQKT`，重写 `question_components` 与 `enhance_questions`。原始 `QuestionKTBase.forward` 可以直接复用。不要继承现有 `NewQ` 后保留其旧题目表、再额外创建新表；这样会留下无用参数并混淆梯度及预算。

本轮默认不引入独立残差损失、熵损失、难度校准、零终端 SNR 日程、额外主干、预训练、梯度截断新规则或 r=32 容量扩展。保留现有三项联合损失及默认系数。上述因素可在后续独立实验中研究，本轮不得顺手混入。

### 13.2 实验依据与可验证假设

现有 V1 的 ASSIST09、seed=42 完整实验得到测试 AUC=0.753005，低于同协议 DiffuQKT 的 0.774305。诊断结果包括：

- 最佳轮注意力归一化熵约 0.999836；推断时强制等权，验证 AUC 几乎不变。
- 探测批次中，共享题目向量从注意力路径获得的 KT 梯度范数仅约为残差路径的 0.249%。这是单批诊断，不是全训练平均。
- 最后轮完整反向生成出现表示和 logit 的异常放大，直接使用初始 Q 可以显著缓解该轮退化。

V2 的可验证假设是：把残差移出扩散训练目标与条件，可能减少两者的耦合；独立题目代码及归一化打分，可能使概念权重获得更有效的学习通路。这些是需要重训验证的假设，不能预先写成已取得的性能改进。

### 13.3 参数、张量形状与确定默认值

记 N 为题目数，M 为概念数，B 为 batch 大小，T 为序列长度，K 为当前张量中的概念槽位数，d 为输出维度。

| 名称 | 实现属性/配置名 | 形状或默认值 | 作用 |
| --- | --- | --- | --- |
| 概念表 C | 继承 `concept` | `[M+1, d]`，默认 d=128 | 共享概念向量 |
| 变化表 V | 继承 `variation` | `[M+1, d]` | 共享概念变化方向 |
| 难度 D | 继承 `difficulty` 后 sigmoid | 每题一个标量 | 原始难度机制 |
| 注意力题目表 E_A | `attention_question_embedding` | `[N+1, r_A]` | 仅用于概念权重查询 |
| 残差题目表 E_R | `residual_question_embedding` | `[N+1, r_R]` | 仅用于题目残差 |
| key 映射 W_K | `key_projection` | `Linear(d, r_A, bias=False)` | 概念投影到匹配空间 |
| 残差映射 W_R | `residual_projection` | `Linear(r_R, d, bias=False)` | 生成 d 维残差 |
| r_A | `attention_embedding_dim` | 8 | 注意力代码维度 |
| r_R | `residual_embedding_dim` | 8 | 残差代码维度 |
| gamma | `attention_scale` | 2.0，固定配置量 | 控制余弦分数幅度，不是可学习参数 |
| tau | `attention_temperature` | 1.0，固定配置量 | softmax 温度 |
| epsilon | 实现常量 | `1e-8` | L2 归一化的安全下界 |

两个题目表必须是不同的 `nn.Embedding` 参数对象，不允许别名共享。两张表独立随机初始化为 `normal_(mean=0, std=0.02)`，然后将 padding 行 0 设为零，使用 `padding_idx=0`。两个线性映射使用 PyTorch 默认初始化，均不设偏置。

维度必须为正整数，显式拒绝 bool。gamma、tau 必须为有限正数。d、heads、dropout、扩散参数沿用基线校验。代码维度之和默认仍为 16，不是把每张表都设成 16。

默认不增加额外 query 投影、独立概念 key 表或多头概念注意力。参数统计按本节所列对象计算。

### 13.4 目标 2：独立代码与余弦注意力的确切公式

对题目 q、当前观测的非重复概念集合 K_q：

$$
a_q=E_A[q]\in\mathbb R^{r_A},\qquad
z_q=E_R[q]\in\mathbb R^{r_R},\qquad
k_k=W_KC_k\in\mathbb R^{r_A}.
$$

仅在最后一个特征维上分别做 L2 归一化：

$$
\widehat a_q=\frac{a_q}{\max(\|a_q\|_2,\epsilon)},\qquad
\widehat k_k=\frac{k_k}{\max(\|k_k\|_2,\epsilon)}.
$$

注意力分数及权重固定定义为：

$$
s_{qk}=\frac{\gamma}{\tau}\widehat a_q^\top\widehat k_k,\qquad
\alpha_{qk\mid K_q}=\operatorname{softmax}_{k\in K_q}(s_{qk}).
$$

**归一化后不再除以 sqrt(r_A)。** gamma 默认 2，tau 默认 1，因此本轮默认有效尺度为 2；不要重复乘除温度，也不要把 gamma 注册为可学习参数。当前默认值是明确的起始配置，不表示已验证为最优。

实现可使用 `F.normalize(..., p=2, dim=-1, eps=1e-8)`。仅归一化用于打分的 query/key，融合 value 仍使用原始 C_k 与 V_k，不能把归一化后的 key 当成 d 维概念 value。计算不得跨 batch、时间或概念集合归一化。

$$
\bar C_{q,K_q}=\sum_{k\in K_q}\alpha_{qk\mid K_q}C_k,\qquad
\bar V_{q,K_q}=\sum_{k\in K_q}\alpha_{qk\mid K_q}V_k,
$$

$$
D_q=\sigma(d_q),\qquad
Q^{sem}_{q,K_q}=\bar C_{q,K_q}+D_q\bar V_{q,K_q},\qquad
R_q=W_Rz_q.
$$

C 与 V 共用同一组 alpha；残差只读取 z_q，不读取 a_q。所有融合结果保持 d 维。

### 13.5 概念集合、掩码与零向量行为

输入兼容 `[B,T]` 单概念 skill_ids 与 `[B,T,K]` 多概念 skill_ids；前者先增加槽位维。K 必须大于零。

按 V1 规则对每次事件的标签排序、去重，忽略 padding 0。有效概念掩码同时包含：非零 ID、当前事件内第一次出现、事件 valid_mask=True。重复标签不增加权重，概念排列不改变融合结果。

注意力应采用以下等价逻辑，保证全掩码事件不会产生 NaN：

```python
masked_scores = scores.masked_fill(~concept_mask, torch.finfo(scores.dtype).min)
weights = masked_scores.softmax(dim=-1)
weights = weights * concept_mask.to(weights.dtype)
```

有至少一个有效概念时，权重和为 1。全掩码时，所有权重为零，不对零和再做除法。单概念事件的唯一有效权重为 1，其打分参数梯度为零。

有效事件若概念集合为空，定义 C_bar、V_bar、Q_sem 为零，仍保留由有效题目 ID 得到的 R；这仅约定安全的数值行为，不声称空标签具有教育学语义。valid_mask=False 时，所有语义向量、残差、原始/增强 Q 与权重均为零，即使 padding 区被填入非零题目/概念 ID 也不得影响有效位置。

query 或 key 恰为零时，归一化结果为零，分数有限。当前流程采用 float32；若以后加入混合精度，应在 float32 中完成归一化和 softmax，再处理输出 dtype，不能让 epsilon 在低精度下失效。

相同题目 ID 的 R 在不同标签视图间相同；alpha 只在当前观测集合归一化。不将跨记录标签并集当作单次作答的共同概念要求。权重与残差计算不得读取当前答案、学生 ID 或未来记录。

### 13.6 目标 1：训练与评估的扩散接口

初始完整题目表示仍定义为：

$$
Q^{orig}=Q^{sem}+R.
$$

扩散部分的接口严格为：

| 扩散入口/目标 | V2 必须使用的量 |
| --- | --- |
| 加噪 x0 | `semantic_original = Q_sem` |
| d 维条件 | `concept = C_bar` |
| 标量条件 | `conditioning = D` |
| 去噪训练目标 | `Q_sem` |
| 反向采样输出 | `Q_sem_star`，尚不包含 R |
| 进入 KT 主分支的表示 | `Q_star = Q_sem_star + R` |

R 及 z_q 不得进入加噪目标、去噪目标、条件拼接、初始噪声或每一步反向迭代。残差仅在本次调用的去噪/完整采样结束后相加一次。禁止沿用 V1 的 `concept=C_bar+R`，禁止在生成结果上重复添加残差，也禁止在训练用输出残差而评估仍用 V1 条件方案。

训练时：

$$
x_t=\sqrt{\bar\alpha_t}Q^{sem}+\sqrt{1-\bar\alpha_t}\varepsilon,
\qquad
\widehat Q^{sem}=G(x_t,t,\bar C,D),
$$

$$
L_{Diff}=\operatorname{MSE}_{valid,d}(\widehat Q^{sem},Q^{sem}),\qquad
Q^*=(\widehat Q^{sem}+R)\odot valid.
$$

训练的随机时间步、噪声、有效位置/嵌入维度归一化与 target 梯度策略均沿用当前 DiffuQKT；本轮不额外 detach Q_sem 目标。

验证/测试时：

$$
Q^{sem,*}=\operatorname{Sample}(\bar C,D;\varepsilon_{seed}),\qquad
Q^*=(Q^{sem,*}+R)\odot valid.
$$

评估保留完整 100 步反向采样、当前 beta 日程及 variance 规则；使用相同外部 generator 接口。评估不计算扩散损失，返回 `None`。

推荐的 `enhance_questions` 伪代码如下；变量名及三个入参对应关系是接口要求：

```python
def enhance_questions(self, components, batch, generator=None):
    semantic = components["semantic_original"]
    concept = components["concept"]       # 只有 C_bar
    condition = components["conditioning"]  # 只有 D
    residual = components["residual"]
    valid = batch["valid_mask"]
    if self.training:
        semantic_star, diffusion_loss = self.diffusion.training_representation(
            semantic, concept, condition, valid)
    else:
        semantic_star = self.diffusion.sample(concept, condition, valid, generator)
        diffusion_loss = None
    enhanced = (semantic_star + residual) * valid.unsqueeze(-1)
    return enhanced, {"diffusion_loss": diffusion_loss}
```

去噪网络输入宽度仍为 `3*d+1`；不增加新的条件投影或扩散模块。

### 13.7 question_components 字段契约与对比学习

| 字段 | V2 含义/形状 |
| --- | --- |
| `concept` | 纯 C_bar，`[B,T,d]` |
| `fused_concept` | 同 C_bar，供诊断使用 |
| `variation` | V_bar，`[B,T,d]` |
| `scale` / `conditioning` | D，`[B,T,1]` |
| `semantic_original` | Q_sem，`[B,T,d]` |
| `residual` | R，`[B,T,d]` |
| `original` | Q_sem+R，`[B,T,d]` |
| `attention_question_embedding` | a，`[B,T,r_A]`，有效位置掩码已应用 |
| `residual_question_embedding` | z，`[B,T,r_R]`，有效位置掩码已应用 |
| `concept_weights` | alpha，`[B,T,K]` |
| `concept_ids` | 与 weights 一一对齐的排序后 ID，`[B,T,K]` |
| `concept_mask` | 非 padding、去重且有效的槽位，`[B,T,K]` |

原始状态对比分支继续读取 `components["original"]`，因此它使用 Q_sem+R；增强分支使用 Q_sem_star+R。两个分支均包含同一题目的同一残差，**只在扩散重构中排除残差，不在状态对比分支中删除残差**。

继续复用基线答案嵌入、知识检索、严格历史掩码、主预测头和损失返回字段。不得把 `original` 改成 Q_sem 后直接复用 base.forward，那会悄悄改变对比学习的比较对象。

### 13.8 损失与梯度的验收契约

默认总损失为：

$$
L=L_{KT}+0.001L_{CL}+0.01L_{Diff}.
$$

所有参数默认正常参与端到端优化，不对 alpha、a、key、z、R 额外 detach。L2 归一化本身保留计算图。独立代码实现的是参数路径分离，不是停止注意力学习。

| 参数 | KT 梯度 | CL 梯度 | Diff 梯度 |
| --- | --- | --- | --- |
| E_A、W_K | 多概念且非退化输入时可有 | 多概念且非退化输入时可有 | 多概念且非退化输入时可有 |
| E_R、W_R | 可有 | 可有 | **必须无依赖，梯度为 None 或全零** |
| C、V、D、扩散网络、KT 主干 | 按原有计算图决定 | 按原有计算图决定 | 按原有计算图决定，不强求与损失无关的主干有梯度 |

Diff 不更新 E_R、W_R 是目标 1 的正确结果。不要复制 V1 中“每一个损失都必须对所有新增参数给出非零梯度”的测试期望。

另外应直接验证：`R.square().sum()` 不依赖 E_A、W_K；只依赖权重或语义部分的诊断目标不依赖 E_R、W_R。这表示专属代码分离，不表示 C 等共享参数与残差训练在长期统计上完全独立。

### 13.9 参数预算与默认配置

两个投影无偏置、gamma/tau 固定且无额外模块时，新增参数必须满足：

$$
\Delta P=(N+1)(r_A+r_R)+dr_A+dr_R
=(N+1+d)(r_A+r_R).
$$

| 数据集，d=128，8+8 | 基线 | 新增 | V2 总量 | 倍率 |
| --- | ---: | ---: | ---: | ---: |
| ASSIST09，N=17,645 | 546,165 | 284,384 | 830,549 | 1.521 |
| ASSIST17，N=2,210 | 525,098 | 37,424 | 562,522 | 1.071 |

这两个总量比 V1 的对应默认总量各少 2,048 个参数。表格须在实际模型构造后再次核验，不能用公式替代最终模型计数。以完整同规模 DiffuQKT 为参照，默认方案必须在 3 倍以内；非默认配置超过 3 倍但不超过 4 倍时给出明确警告，超过 4 倍直接报错。不要把用于统计的基线模型注册为 V2 的子模块。

新增 `configs/models/newq_v2.yaml`，专属参数应为：

```yaml
model:
  attention_embedding_dim: 8
  residual_embedding_dim: 8
  attention_scale: 2.0
  attention_temperature: 1.0
  diffusion_steps: 100
  beta_start: 0.0001
  beta_end: 0.02
  reverse_variance: posterior
loss:
  diffusion_weight: 0.01
```

其余公共参数从现有 base.yaml 继承：d=128、heads=2、dropout=0.4、contrastive_weight=0.001、temperature=0.3、学习率 0.002、weight_decay=1e-5、grad_clip=15、seed=42。V2 配置不带旧 `question_embedding_dim`，不同时保留一张未使用的旧题目表。

`--smoke` 时 d=32、r_A=r_R=2，代码总维度为 4；保持当前公共 smoke 的其他行为。正式配置不受 smoke 尺寸影响。

### 13.10 文件、注册、checkpoint 与结果目录

编码任务应新增：

- `src/model/newq_v2/model.py`：`NewQV2`。
- `src/model/newq_v2/__init__.py`：导出 `NewQV2`。
- `configs/models/newq_v2.yaml`：本节默认配置及中文参数说明。
- `tests/test_newq_v2.py`：下节的功能和回归验收。
- `docs/newq_v2.md`：最终实现、命令、参数与实际核验记录。

必要接入位置：

- `src/model/registry.py` 的 MODEL_REGISTRY 注册 `newq_v2`。
- `src/model/__init__.py` 导出新类。
- `src/utils/config.py` 的 MODEL_CONFIGS 注册新配置。
- `src/training/cli.py` 为 V2 增加 smoke 尺寸处理；当前模型 choices 从 MODEL_CONFIGS 派生，沿用该机制。
- 按项目既有约定更新统一命令与文件说明；仅在必要处接入，不重构公共训练器。

预期命令与路径：

```powershell
python main.py assistment2009 --model newq_v2 --epochs 100 --device cuda
python main.py assistment2017 --model newq_v2 --epochs 100 --device cuda
python main.py assistment2009 --model newq_v2 --smoke --device cuda
python main.py assistment2009 --evaluate experiments/newq_v2/assistment2009/newq_v2_assistment2009_100epochs_1/best.pt --device cuda
```

输出按现有分组规则进入 `experiments/newq_v2/<dataset>/newq_v2_<dataset>_<epochs>epochs_<n>/`，不覆盖 newq 目录。checkpoint 的 model_name 必须是 `newq_v2`，model_kwargs 保存两个代码维度、gamma、tau 及所有继承构造参数。

旧 `newq` checkpoint 继续按旧类严格加载；V2 只按 V2 的参数和名称加载。不将旧共享 16 维表切成两半后静默宣称等价恢复。验证/测试仍校验 metadata，数据划分及标签存储不随此次结构变化重建。

### 13.11 必须通过的验收用例

以下测试验证实现契约，不以“注意力必须不均匀”或“AUC 必须上升”作为通过条件。

1. **手工数值。** 在 d=8、r_A=r_R=2 的小模型中，设 query=(1,0)，两个归一化 key 分别为 (1,0)、(0,1)，gamma=2、tau=1；两概念权重应为 softmax([2,0])。设 C1=e1、C2=e2、V1=2e1、V2=4e2、D=0.5、R=0.2e1，则 Q_sem=2p*e1+3(1-p)*e2，original=Q_sem+0.2e1，其中 p=sigmoid(2)。
2. **集合与数值边界。** 排列不变、标签重复不增权、padding 权重及输出为零；单概念权重为 1；空概念集合及零 query/key 不产生 NaN；非法维度、温度、尺度被拒绝。
3. **训练调用捕获。** mock 或 spy `training_representation`，证明 x0/target 是 Q_sem，concept 是 C_bar，标量为 D；R 不在任何扩散输入中；返回 enhanced 恰为 semantic_star+R，残差只加一次。
4. **评估调用捕获。** 同样验证 `sample` 只收到 C_bar、D、valid、generator，返回后才加 R，diffusion_loss=None。
5. **残差绕行的精确性质。** 保持注意力代码、共享参数与采样 generator 固定，只改变 E_R 或 W_R。语义扩散输入及生成结果必须不变，最终 Q_star 的变化应等于 R 的变化，允许 float32 舍入误差。该性质同时覆盖训练路径（固定随机时间步/噪声）和评估路径。
6. **损失梯度路由。** 多概念非退化批次中分别反传 KT、CL、Diff，验证本节梯度表。单独反传 Diff 时 E_R、W_R 不得获得更新；单概念打分梯度为零；padding 嵌入行梯度为零。KT/CL 对残差的非零检查使用确有有效预测和负样本的批次。
7. **对比分支对象。** 捕获原始状态分支输入，证明是 Q_sem+R；主分支输入为 Q_sem_star+R。禁止用删去原始分支残差的方式“简化”测试。
8. **完整基线退化。** 令 W_K=0 获得均匀权重、W_R=0 获得零残差，复制同名共享参数到同配置 DiffuQKT，固定训练随机种子及评估 generator，训练前向、采样前向、损失及预测应与基线在合理 float32 容差内一致。该测试不要求 V2 与 V1 共享表结构相同。
9. **因果与有效位置。** 当前/未来答案不得改变当前及更早预测，未来题目不得改变更早预测；填充区数值不得影响有效位置。
10. **参数预算及运行接入。** 两个真实词表 d=128、8+8 的参数量与本节表格相符；超 4 倍拒绝，3-4 倍警告；模型注册、CLI 配置、smoke 配置、V2 checkpoint 自动恢复、旧模型 checkpoint 恢复均有效。
11. **真实数据流程检查。** 使用已有 processed 小范围完成训练、验证选模、保存、恢复和测试；记录实际范围与设备。小范围流程检查不得写成正式性能结论。

### 13.12 诊断与结果记录要求

至少提供可在保存权重上执行的只读诊断方式，复用 question_components 字段即可，不要求为此改写公共训练器。记录：

- 多概念事件/题目的 alpha 归一化熵，以及相对均匀权重的最大偏差；排除 K<=1，ID 与权重必须按返回顺序对齐。
- R 与 Q_sem 范数之比及 p50/p95/p99；明确按题目还是按交互计权。
- 完整反向生成的语义表示范数 p95/p99/max、最终 Q_star 范数、最大绝对 logit。
- 使用 logits 计算的稳定 BCE，同时保留现有概率 BCE；极端概率裁剪可能掩盖大 logit 损失，两者不可混称。
- 不同代码参数的梯度范数及完整模型参数量；单批梯度探测须明确范围。

V2 的 Diff 重构目标不再包含 R，其数值与 V1 的 Diff 目标不同，不能仅比较 MSE 下降就声称生成更好。仍通过统一验证集 AUC 选择权重，最终测试使用同一数据划分；正式性能结论需另行开展受控重训及多种子实验。

本次不通过减小最大采样步数、给表示做未声明的 clip 或在测试时切换直接 Q 来掩盖异常。若仍出现生成放大，应按实际结果记录，并将采样日程或稳定性改进作为新的独立实验因素。

### 13.13 建议的编码顺序与交付判定

1. 新增独立 V2 类与两个代码表，实现归一化权重及清楚的 components 字段。
2. 实现专属 enhance_questions，首先通过训练/评估路由和“残差只加一次”测试。
3. 验证损失梯度依赖关系及基线退化，修正新测试的语义，保留旧 V1 测试。
4. 接入专属配置、注册、smoke 和 checkpoint，核算实际总参数。
5. 执行现有相关回归检查与小范围真实数据流程检查，补齐 docs/newq_v2.md 的实际结果。

编码完成的含义是：两项目标按上述公式及接口实现，验收通过，版本与旧实验隔离，并如实记录验证范围。当前文档给出了可执行的起始设计，不承诺 V2 在完整训练中一定超过基线；性能验证与代码正确性必须分别报告。

## 14. NewQ V2 编码与验收完成记录（2026-10-18）

已新增独立 NewQV2、专属配置、验收测试和保存权重的只读诊断工具，并完成注册、smoke 及 checkpoint 接入。实现采用第 13 节的 8+8 独立代码和固定尺度余弦匹配，残差不进入语义扩散，仅在训练/采样结束后加入一次，完整原始状态分支保留残差。

87 项回归检查通过，两个真实 processed 数据集各完成 5 轮小范围 CUDA 流程检查及独立恢复评估；另验证真实 smoke 命令。默认参数总量与第 13.9 节一致。完整采样和单批梯度诊断如实记录于 [newq_v2.md](newq_v2.md)，这些结果仅验证代码及局部流程，不作为完整性能提升结论。

## 15. NewQ V2 残差扩散插件：两个配置开关与下一步条件实验（待编码规格，2026-10-18）

### 15.1 本次修改摘要、范围与优先级

用户下一步优先验证方案 B：让题目残差 R 作为独立的额外条件指导语义生成，生成结束后仍直接加入 R。此次在现有 `newq_v2` / `NewQV2` 内实现可配置插件，**不创建 newq_v3、新模型标识或另一套模型注册**。

本节是下一轮 Codex 编码的权威增量规格。第 13～14 节保留为原 V2 的历史规格与实现记录；其中“R 不进入扩散”“Diff 不依赖残差参数”“残差始终在生成后加入”只对本节的关闭模式继续整体成立。启用模式按本节执行，不能用旧断言禁止新增路径。

新增两个模型配置字段：

1. `residual_diffusion_enabled`：bool，决定 R 是否进入扩散模块。
2. `residual_diffusion_role`：字符串枚举 `condition` / `target` / `both`，决定启用后的作用。

这两个字段只控制 R 在扩散中的路径。关闭插件仍保留当前 V2 的残差表、R、原始对比分支及输出端残差；**不是关闭整个残差，也不是关闭扩散**。如需无残差或无扩散对照，应以后另行设计，不复用本次两个字段。

注意力、8+8 题目代码、C/V/D、现有 KT 主干与损失系数均保持。默认不引入额外残差正则、熵损失、难度校准、噪声日程修改或新的预训练流程。本轮性能比较必须在相同 V2 结构内进行，不将旧 V1 与新插件的差值全部归因于残差路径。

### 15.2 两个开关的确定语义与配置校验

| enabled | role | 有效模式 | R 是否作为独立条件 | 加噪与重建目标 | 生成后如何得到 Q_star |
| --- | --- | --- | --- | --- | --- |
| false | condition、target 或 both | `off`，当前 V2 | 否 | Q_sem | 生成结果 + R |
| true | condition | `condition`，下一步方案 B | 是 | Q_sem | 生成结果 + R |
| true | target | `target`，仅进入完整生成目标 | 否 | Q_orig = Q_sem + R | 直接使用生成结果，不再加 R |
| true | both | `both`，同时进入条件和完整目标 | 是 | Q_orig = Q_sem + R | 直接使用生成结果，不再加 R |

**按用户明确选择，target 严格表示 R 仅属于加噪/重建目标，不作为扩散条件；both 是另外新增的枚举选项，才表示同时作为条件和目标。** 不得将 target 自动解释或改写为 both。target/both 都生成完整 Q_orig，因此生成后均不能再加一次 R。

target 的预测过程只以 C_bar、D 为条件，从纯噪声生成完整题目表示；自由题目残差的信息需要由这些条件和已学参数间接恢复。固定其他参数、C_bar、D 与采样 generator 后，仅改变 E_R/W_R 对应的当前 R 值，不应改变 target 的评估预测。这是该实验刻意检验的信息路径，不是实现遗漏。both 向每步去噪器显式提供 R，可与 target 对比条件供给的作用。

`residual_diffusion_enabled` 只接受 Python/YAML bool，拒绝整数 0/1、字符串 `"true"` / `"false"`、None。`residual_diffusion_role` 只接受精确小写 `condition`、`target` 或 `both`；即使 enabled=false 也校验 role，避免关闭时隐藏拼写错误。禁止根据字符串 truthiness 判定启用状态。

有效模式属性建议命名为 `effective_residual_diffusion_mode`，取 `off` / `condition` / `target` / `both`。它由两个配置字段推导，不是第三个用户配置。enabled=false 时，role 保留在配置中但不影响模型结构、参数数量或推断结果。不增加第三个布尔开关，两个内部派生条件必须为：

```python
use_residual_condition = enabled and role in {"condition", "both"}
use_residual_target = enabled and role in {"target", "both"}
```

### 15.3 统一数学定义与四种训练/预测路径

继续使用现有 V2 定义：

$$
\bar C=\sum_k\alpha_{qk}C_k,\qquad
\bar V=\sum_k\alpha_{qk}V_k,\qquad
Q_{sem}=\bar C+D_q\bar V,\qquad
R=W_RE_R[q],\qquad Q_{orig}=Q_{sem}+R.
$$

定义本次扩散目标 X_0 和额外条件 r_cond：

$$
X_0=\begin{cases}
Q_{orig},& enabled\land role\in\{target,both\},\\
Q_{sem},& \text{其余情况},
\end{cases}
\qquad
r_{cond}=\begin{cases}
R,& enabled\land role\in\{condition,both\},\\
\varnothing,& \text{其余情况}.
\end{cases}
$$

训练时仍随机抽取时间步 t、标准高斯噪声 epsilon：

$$
X_t=\sqrt{\bar\alpha_t}X_0+\sqrt{1-\bar\alpha_t}\varepsilon,
\qquad
\widehat X_0=f_\theta(X_t,t,\bar C,D;r_{cond}),
$$

$$
L_{Diff}=\operatorname{MSE}_{valid,d}(\widehat X_0,X_0).
$$

不额外 detach X_0、R 或条件，保留当前目标梯度策略。MSE 仍只在有效位置与 d 个输出维度上归一化，不根据条件输入宽度改变分母。

预测时从标准高斯噪声开始，按照同一有效模式执行完整反向采样：

$$
X_0^*=\operatorname{Sample}_\theta(\bar C,D;r_{cond},\varepsilon_{seed}).
$$

训练和预测统一使用以下输出规则，最终再应用 valid_mask：

$$
Q^*=\begin{cases}
X_0^*,& enabled\land role\in\{target,both\},\\
X_0^*+R,& \text{其余情况}.
\end{cases}
$$

其中训练阶段的 X_0_star 指单次去噪预测，预测阶段指完整反向采样输出。两阶段必须共用同一个目标/输出判定，禁止训练采用 condition、评估采用 target。

condition 中 R 是未加噪的条件；本轮不对它另加噪、不先投影回 8 维、不改成 C_bar+R。target/both 被加噪的都是整体 Q_orig，区别是 target 无 R 条件，both 另外提供未加噪的 R 条件。初始采样状态仍为标准高斯噪声，不将 R 加到初始噪声上，也不在每次迭代后人为累加 R。

“额外约束条件”指生成器的 conditioning 输入，**不表示新增一个损失项**。condition 不保证 R 的输出值被生成器逐值复制，输出端的直接相加负责保留这一直接路径。

### 15.4 条件插件的结构与参数初始化

仅在 use_residual_condition=True，即 condition/both 中，R 才以独立 d 维输入提供给去噪器，不与 C_bar 相加。off/target 不增加 R 条件槽位，保留原网络。本轮采用明确拼接方案，输入顺序固定为：

```python
base_input = torch.cat([concept, noisy, question_scale, time_rep], dim=-1)
# off/target: input_width = 3*d + 1，完整复用当前原 V2 扩散模块。
# condition/both: input_width = 4*d + 1，最后追加 R，保留原输入列顺序。
denoiser_input = torch.cat([base_input, residual_condition], dim=-1)
```

维度仍为：concept/noisy/time_rep/R 各 `[B,T,d]`，question_scale 为 `[B,T,1]`。不改变 D 的形状或把 R 塞进 D。后续 MLP 仍为 `Linear(input_width,2*d) -> GELU -> Linear(2*d,d)`，time_mlp 及噪声日程保持。

建议在 `src/model/newq_v2/diffusion.py` 中新增 V2 专用的条件扩散实现，继承/复用 `ConditionalQuestionDiffusion` 的日程和采样计算；避免复制一套会产生公式漂移的采样器。off/target 保留现有扩散对象；condition/both 只保留一个扩展条件输入的扩散对象。不得把原扩散与新扩散同时注册为两个子模块而留下未使用参数。

为了控制初始化差异，先按当前顺序构造原 V2 的全部参数，再在 use_residual_condition=True 时构造扩展条件模块：time_mlp、原 denoiser 的共有权重/偏置逐值复制；第一层前 `3*d+1` 列复制原权重，新增最后 d 列置零。第二层保持原值。这样同一初始化种子下，四模式的所有共有参数一致；扩展网络在相同 noisy/t/C_bar/D 且 R=0 时与原网络的 predict_x0 输出一致。注意这是去噪器数值等价，不代表 target 与 off 的最终 Q_star 等价，后者是否加 R 不同。训练前仍按项目策略固定数据顺序与噪声随机数，不能仅凭同一个 seed 就默认初始化后随机数状态也一致。

新增条件列初始化为零意味着第一次前向时条件通路尚未起作用；不能要求此刻 Diff 对残差的梯度必须非零。条件权重正常训练后可获得作用。梯度路由测试须人为设置非退化条件权重或先执行受控更新，再判断条件通路是否连通。

开启后的专用接口建议显式支持：

```python
training_representation(original, concept, question_scale, valid,
                        residual_condition=None)
sample(concept, question_scale, valid, generator=None,
       residual_condition=None)
predict_x0(noisy, time, concept, question_scale,
           residual_condition=None)
```

保留 generator 原位置，新参数使用关键字传入。condition/both 的扩展采样循环每一步都必须把同一份 R 传给 predict_x0，不能只在第一步或训练入口传入。off/target 不传入 R 条件。条件 R 应检查最后维度、B/T 形状、device/dtype，padding 区先乘 valid 掩码，沿用第 13.5 节对 padding 非零填充值的保护。

### 15.5 NewQV2 路由与 components 契约

`question_components` 的已有字段在四模式中含义完全相同：`concept=C_bar`、`semantic_original=Q_sem`、`residual=R`、`original=Q_sem+R`、`conditioning=D`。不要为 target/both 把 `semantic_original` 改成完整 Q，也不要为 condition/both 把 `concept` 改成 C_bar+R。

建议集中实现一个内部路由辅助方法，返回 `diffusion_target`、`residual_condition`、`add_residual_after_generation`，让训练、评估和诊断共用。伪代码：

```python
is_target = self.residual_diffusion_enabled and self.residual_diffusion_role in {"target", "both"}
is_condition = self.residual_diffusion_enabled and self.residual_diffusion_role in {"condition", "both"}
target = parts["original"] if is_target else parts["semantic_original"]
residual_condition = parts["residual"] if is_condition else None

if self.training:
    # off/target 保留旧调用签名；condition/both 显式传 residual_condition。
    generated, diffusion_loss = diffusion_train(target, parts["concept"],
        parts["conditioning"], valid, residual_condition=residual_condition)
else:
    generated = diffusion_sample(parts["concept"], parts["conditioning"], valid,
        generator=generator, residual_condition=residual_condition)
    diffusion_loss = None

enhanced = generated if is_target else generated + parts["residual"]
enhanced = enhanced * valid.unsqueeze(-1)
```

上面的 diffusion_train/sample 表示按有效模式调用对应模块的内部包装，不要求修改原基线方法签名。off/target 分支继续按原参数调用旧扩散模块，condition/both 分支才使用新关键字。

`QuestionKTBase.forward` 和状态对比机制保持。原始分支始终使用 Q_orig；增强分支使用本节定义的 Q_star；target/both 也不能把原始分支换成 Q_sem。答案信息只进入既有 KT 路径，插件不读取当前答案、学生状态或未来事件作为生成条件。

### 15.6 损失、梯度与诊断要求

总损失仍为 `KT + 0.001*CL + 0.01*Diff`，四模式都不新增 L_R。评估辅助损失仍为 None。

| 有效模式 | KT/CL 对残差表与投影 | Diff 对残差表与投影 |
| --- | --- | --- |
| off | 保留原有依赖 | 无依赖，None 或零 |
| condition | 有直接输出路径，也可有条件路径 | 可沿条件路径传播；目标本身不含 R |
| target | 训练时沿含 R 的加噪输入、以及原始对比分支传播；无输出端直通 | 沿加噪输入和重建目标传播，不含 R 条件路径 |
| both | 沿含 R 的加噪输入、条件与原始对比分支传播；无输出端直通 | 沿条件、加噪输入和重建目标传播 |

本轮不额外 detach R、条件或训练目标。target/both 的预测主分支不再具有额外的输出端残差直通，不能同时要求“生成完整 Q”且“再加一次 R”。target 在评估时对当前 R 的直接依赖应为零，R 只通过训练已改变的其他参数间接影响预测；训练单步加噪时则有 R 输入路径。若希望比较停止条件梯度，应作为后续独立因素，本轮不增加第三个配置标志。

`diagnostics.py` 现在直接调用 `diffusion.sample(C_bar,D,valid,generator)` 并固定在结果上加 R，此路径必须改为复用模型路由，否则启用后会报错或诊断到错误结构。输出至少保存：两个请求字段、有效模式、重建目标名称、是否存在 R 条件、生成后是否加 R、实际参数量。

生成向量及 MSE 必须按实际目标命名：off/condition 的生成结果是语义表示，target/both 的生成结果是完整题目表示，不能在 target/both 上仍报告为 semantic_generated，或与 Q_sem 计算“重建误差”。梯度验收随有效模式变化。继续报告完整生成范数、max|logit|、稳定 logit BCE 与概率 BCE。

### 15.7 参数预算

off/target 参数量与当前 V2 完全一致。仅 condition/both 扩展 denoiser 第一层输入，增加 `d*(2*d)=2*d^2` 个权重，不增加偏置、另一张题目表或第二个去噪器：

$$
P_{condition}=P_{both}=P_{V2}+2d^2,\qquad P_{off}=P_{target}=P_{V2}.
$$

| 数据集，d=128，8+8 | 原 DiffuQKT | off / target | condition / both | 扩展条件后相对原基线 |
| --- | ---: | ---: | ---: | ---: |
| ASSIST09 | 546,165 | 830,549 | 863,317 | 1.581 倍 |
| ASSIST17 | 525,098 | 562,522 | 595,290 | 1.134 倍 |

独立条件扩展增加 32,768 个参数，仍在 3 倍目标内。保留超过 3 倍警告、超过 4 倍报错；必须将条件模块增量计入预算，不能只校验当前题目代码增量。condition 与 both 参数数量相同，off 与 target 参数数量相同；跨组比较时如实报告条件模块增加的容量，不声称所有组总参数完全一致。

### 15.8 YAML、构造默认值和 checkpoint

完成编码后，在现有 `configs/models/newq_v2.yaml` 的 model 节增加以下两个**生效参数**并逐项写中文注释；不新建另一模型 YAML，不写到所有模型共享的 base.yaml：

```yaml
model:
  # 其他已有 model 参数保留，含 8+8 维、gamma=2、100 步及原 beta 日程。
  residual_diffusion_enabled: true  # 下一步运行 B：R 进入扩散；false 完整恢复原 V2。
  residual_diffusion_role: condition  # condition：仅作条件；target：仅作目标；both：同时作条件和目标。
```

四组切换只改变这两个值：

```yaml
# 当前 V2 控制组
residual_diffusion_enabled: false
residual_diffusion_role: condition

# 下一步优先实验 B
residual_diffusion_enabled: true
residual_diffusion_role: condition

# 后续仅生成目标对照，无 R 条件
residual_diffusion_enabled: true
residual_diffusion_role: target

# 后续同时作为条件与目标的对照
residual_diffusion_enabled: true
residual_diffusion_role: both
```

上述四段是互斥示例，不能把多组同名键重复放入同一个 YAML。配置头部、loss.diffusion_weight 注释及模型文档应同步说明模式，不能再无条件声称 Diff 不依赖残差或残差始终在生成后加入。

**类构造默认值与新训练 YAML 的默认值有意不同：**

```python
NewQV2(..., residual_diffusion_enabled=False,
       residual_diffusion_role="condition")
```

构造函数默认关闭，用于保证缺少新字段的历史 checkpoint 原样恢复。新训练通过专属 YAML 显式启用 condition，以执行下一步 B 实验。

目前公共 `trainer.model_kwargs` 会将整个 config["model"] 传入构造函数；**不能在构造函数尚未支持前只把生效新键写进实际 YAML**，否则常规训练会报 unexpected keyword argument。本文先提供完整配置规范；后续编码必须把构造、路由、实际 YAML 与回归检查作为同一个完整变更落实。

新 checkpoint、summary 和运行目录的 config.yaml 必须保存两个字段，并在日志/summary 中另存推导的有效模式。model_name 仍为 `newq_v2`，沿用现有结果目录自动编号、不覆盖旧运行。评估必须按 checkpoint.model_kwargs 恢复结构，不能由当前默认 YAML 强行改变旧模型模式。

历史 V2 缺少新字段时按 off 建模，保持原层形状、参数键、初始化顺序和严格 state_dict 加载；历史预测应逐值复现。新 off/target checkpoint 与历史层形状相同，condition/both 使用扩展输入层。**off 与 target、condition 与 both 各自虽然权重形状相同，目标/输出含义仍不同；不能仅凭层宽或 strict=True 加载成功推断模式。** 保存的开关必须决定完整路由。禁止用 strict=False、裁剪/填充错误层宽或静默迁移来掩盖模式不匹配。需要从 off 权重初始化 condition 实验时，应作为显式新实验初始化操作，并记录共有参数复制与新增条件列置零，不能称为原 checkpoint 的等价恢复。

### 15.9 文件改动范围与验收清单

后续实现预期修改：

- `src/model/newq_v2/model.py`：两个构造参数、模式校验、V2 专用条件扩散接入、统一目标/输出路由及实际预算。
- `src/model/newq_v2/diffusion.py`：condition/both 可选条件插件，复用原扩散计算，R 每一步作为独立条件。
- `configs/models/newq_v2.yaml`：两个生效字段，下一步默认 true/condition。
- `src/model/newq_v2/diagnostics.py`：复用模式路由，修正目标、输出名称和梯度验收。
- `tests/test_newq_v2.py` 及必要的 V2 专属测试：保留 off 历史验收，覆盖四个有效模式。
- `docs/newq_v2.md`：实现状态、配置命令、参数量和实际核验范围。

不新增模型注册，不修改 NewQ V1、LinearQ、原 DiffuQKT 的运行行为，不重建数据。公共代码仅在确有必要且默认行为有回归保护时调整，不把重构当成本次目标。

必须验证：

1. **配置校验与落地。** 两字段类型、非法 role、关闭时仍校验；真实 load_config -> model_kwargs -> build_model 链路可运行，默认新训练为 condition。
2. **关闭等价。** enabled=false 的三个合法 role 输出相同，参数/层形状与原 V2 一致；历史 best/last 严格加载，固定 generator 后旧预测逐值一致。
3. **确定性手算路由。** 用返回固定向量的去噪/采样 spy，off/condition 输出 generated+R，target/both 输出 generated；验证目标选择和训练/评估一致性，避免 R 加两次或漏加。
4. **独立条件与目标专属路径。** 固定 C_bar、D、noisy、t，使用非退化条件权重改变 R，condition/both 的去噪输出可改变，off/target 的去噪器不接受 R 条件。进一步固定全模型其他参数和 generator，仅改变 E_R/W_R：target 的完整评估预测必须相同；训练目标和加噪输入应改变。不能只检查函数收到参数而不检查数值依赖。
5. **逐步采样。** condition/both 的 spy 验证全部 steps 次调用收到同一 R；off/target 每步不接收 R 条件。保留原 posterior/fixed_large 公式、generator 与标准高斯初始化，不改变采样步数。
6. **梯度。** off 的 Diff 不依赖 E_R/W_R；condition 的 Diff 在非退化条件映射下可向两者传播；target 沿加噪与目标路径传播；both 沿条件、加噪与目标路径传播。新增条件列为零的初始化状态与“断开梯度”应明确区分。
7. **对比与因果性。** original 始终 Q_sem+R；保持已有严格历史检索、不读取当前答案作为生成条件；单/多概念、重复标签、零向量、padding 非零填充值安全。
8. **参数与初始化。** condition/both 增量精确为 2*d^2，off/target 不增加条件参数；共有参数复制、新增条件列为零；相同去噪输入且 R=0 时扩展/原网络输出相同；旧的 3/4 倍预算保护覆盖完整网络。
9. **恢复与诊断。** 四模式 checkpoint 往返一致，评估采用保存模式；诊断输出的目标、生成向量名称、是否后加 R 与真实路由一致，checkpoint 哈希不变。对形状相同但语义不同的模式，专门检查不会被当前 YAML 覆盖。
10. **真实数据流程。** 两个 processed 数据集分别做小范围训练、完整步数验证、保存恢复及诊断，确认没有只在 toy 输入下生效。只读评估与流程核验不能写成完整性能改进结论。

本节当前状态是**待实现的设计摘要与验收契约**，不表示插件已编码或 B 实验已运行。完成实现后另追加实际验收记录，保留本节的四种明确语义。性能对照优先比较 off 与 condition；target/both 后续按同一 V2 结构验证。所有组通过验证集选模，不根据测试集干预结果调节开关。

## 16. 第 15 节残差扩散插件实现与验收记录（2026-10-18）

已在现有 NewQV2 中实现两个配置字段、四种明确路由与独立条件插件，新训练 YAML 默认 true/condition，构造函数默认 false/condition 保证历史恢复。原始组件语义不变，扩散的目标/条件/输出判定由训练、评估、诊断共用；未增加模型名或第三个用户开关。

98 项回归检查通过，包括关闭模式对旧固定输入/随机数的逐值复现、条件逐步传入与数值依赖、target 的预测不直接依赖 R、四模式梯度和保存模式恢复。两个真实 processed 数据集的四模式均完成小范围 CUDA 训练、完整 100 步验证、最佳权重恢复和只读诊断；off/condition 各 5 轮，target/both 各 1 轮，仅作流程检查。参数计数与第 15.7 节一致，历史 best/last 均严格加载。

完整范围、实际数值、初始化和 RNG 处理、模式元数据及回归边界见 [newq_v2.md](newq_v2.md)。这些检查不等同于完成全量 B 性能对照实验，也不承诺性能改善。

## 17. 条件输入 detach 对照（2026-10-19）

按后续用户要求，在已有 NewQV2 中新增独立布尔字段 `residual_condition_detach`，默认 false。该要求补充第 15 节当时“不增加第三个开关、不 detach 条件”的限制；四模式路由、网络结构、参数量与原有三项损失保持一致。

本次最小对照使用：

```yaml
model:
  residual_diffusion_enabled: true
  residual_diffusion_role: condition
  residual_condition_detach: true
```

其表示为：

$$
Q^*=f_\theta(x_t,t,\bar C,D,\operatorname{sg}(R))+R.
$$

仅传给扩散器的独立 R 条件视图使用 `R.detach()`。原始组件、输出端 R 以及原始对比分支均保持正常梯度。生成器仍读取相同 R 数值，条件映射参数继续学习；KT/CL 仍能经正常残差路径更新 E_R/W_R。condition 的 Diff 不再通过条件输入更新残差参数。both 模式也只断开条件路径，其完整目标/加噪输入中的 R 仍可传梯度；off/target 没有独立 R 条件，因此标志不改变计算。

当前实际 YAML 的 enabled=false/role=condition 为用户已有设置，予以保留；新增标志为 false，不自动启动对照。运行对照时显式采用上面三个值。训练保存请求值 `residual_condition_detach` 及推导值 `effective_residual_condition_detach`，评估按 checkpoint 构造参数恢复，不由当前 YAML 改写。历史缺字段权重按 false 严格恢复。

104 项回归检查通过；开/关标志的四模式初始参数、固定随机数下训练/采样前向值逐值一致，默认 false 与变更前的完整梯度逐值一致。ASSIST09 和 ASSIST17 各以 condition 的 false/true 运行 2 轮 CUDA 检查，使用 d=128、8+8 代码、完整 100 步扩散，每划分最多 32 窗口、长度 32、batch=8、每轮最多两个 batch；均完成选模、保存、恢复测试与只读诊断。19 份 best 和 19 份 last 历史权重严格加载，65 个受保护文件哈希不变。临时产物已清理，详细梯度与范围见 [newq_v2.md](newq_v2.md)；本次未开展完整训练性能比较。

## 18. 残差融合与拼接维度控制（2026-10-19）

按后续要求增加 `residual_fusion: add/concat`，默认 add；另增加 `residual_concat_keep_2d`，true/1 表示保留拼接后的 2d 维，false/0 表示用共享可学习无偏置 Linear(2d,d) 映射回 d，默认 false。只接受 bool 或整数 0/1，保存时归一化为 bool；add 时第二项无实际计算作用。

将 Q_sem 与 R 的统一融合写为 F(X,R)：add 为 X+R；concat/true 为 [X;R]；concat/false 为 W_f[X;R]。拼接顺序固定，W_f 初始为 [I,I] 且参与训练。原始分支 Q_orig=F(Q_sem,R)，off/condition 的增强分支为 F(generated,R)，两条分支共享规则及 W_f；target/both 生成完整融合目标后直接使用，不再次融合。

保留 2d 时，语义表示、残差、概念和独立残差条件仍为 d，答案嵌入、检索器和预测头改用 2d。off/condition 的扩散仍生成 d，target/both 的完整扩散目标/输出改为 2d。后者保留 d 维时间 MLP、2d 隐藏层及原训练/采样循环，仅适配去噪输入输出和噪声形状；损失系数、方差公式、采样步数与条件 detach 语义保留。

配置/模型构造参数/checkpoint/summary/诊断均保存两字段与实际维度，旧权重缺字段时按 add/false 严格恢复。默认相加不新增参数，也不改变初始化、训练/评估结果或梯度。全部变更留在 NewQ V2 包和专属配置/测试/说明中，公共主干、数据处理与其他模型保持原样。

114 项检查通过；两个真实数据集、两种拼接维度、四路由共 16 次一轮小范围 CUDA 训练、保存恢复及只读诊断通过，保留完整 100 步扩散。参数量及实际范围见 [newq_v2.md](newq_v2.md)，本次只核验功能与梯度，不据此比较完整训练性能。

## 19. D 作为可选的去噪条件（2026-10-19）

NewQ V2 新增 `model.diffusion_difficulty_condition: true`，默认保留 D 条件。关闭后以 D_cond=0 替代去噪器的显式 D 输入，训练重建与全部逆扩散步骤采用相同设置，概念条件及独立 R 条件按原路由使用。四种残差路由和三种融合形式均兼容。

此开关不移除 `Q_sem=C_bar+D*V_bar` 中的 D，不改变目标、原始分支、损失或参数形状；D 仍可经表示/目标路径学习。它也不是 detach：关闭后生成器不读取显式 D 的数值，但训练加噪目标仍可含难度信息。仅在 NewQ V2 代码和专属配置中接入；基线及共享扩散实现不变。保存和恢复包含开关，历史缺字段按 true 恢复。具体语义和核验见 [newq_v2.md](newq_v2.md)。
