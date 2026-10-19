# NewQ：题目条件的概念融合与修正残差

实现日期：2026-10-18。依据 [q_structure_v1.md](q_structure_v1.md) 的第一版结构与第 11 节精简参数化，实现独立模型 `newq`。

## 本次采用的公式

对题目 q、本次记录的非重复原子概念集合 K：

```text
e_q = question_embedding[q]          # r 维，默认 r=16
D_q = sigmoid(difficulty[q])         # 沿用基线标量难度
s_qk = e_q^T key_projection(C_k) / (sqrt(r) * attention_temperature)
alpha_qk = softmax_K(s_qk)           # 只对当前记录的有效、不重复概念归一化
C_bar = sum_k alpha_qk * C_k
V_bar = sum_k alpha_qk * V_k          # 与 C 使用同一组权重
R_q = residual_projection(e_q)       # d 维，默认 d=128
Q = C_bar + D_q * V_bar + R_q
```

两个投影均不含偏置。题目向量以标准差 0.02 的正态分布初始化，padding 行为 0；投影采用 PyTorch 线性层的默认初始化。小幅题目向量使初始权重接近均匀、残差较小，多概念事件可从第一步向各新增模块传播任务梯度。

alpha 每次前向由网络计算，既不离线缓存，也不 detach。它不使用答案或学生状态。概念顺序与重复标签不会改变融合结果，padding 的权重为 0。单概念事件的权重恒为 1，打分网络在该事件上的梯度为 0，残差及原有嵌入仍可学习。同题在不同记录中的标签集合分别处理，不合并成标签并集；R_q 在同题的不同标签视图间共享。

权重均匀且 R_q=0 时退化为原始 Q。权重是模型内的相对贡献，不自动构成教育学重要性或因果解释。仅用 ID 学习的题目向量不能给未见新题提供可靠的题目特异性估计；当前继续使用学生互斥的常规划分，不新增冷启动协议。

## 如何复用原始扩散

NewQ 继承 `DiffuQKT`，只实现构造函数及 `question_components`。直接继承以下计算：

- `DiffuQKT.enhance_questions`：训练时加噪与 x0 重建，评估时完整逆扩散。
- `QuestionKTBase.forward`：知识检索、主预测、原始/增强状态对比及联合损失。
- 原有注意力、答案嵌入、预测头、掩码和采样规则。

为让残差信息在评估生成时仍有入口，使用讨论稿第 7 节的同维条件方案：

```text
扩散的概念条件 = C_bar + R_q          # d 维
扩散的标量条件 = D_q                  # 1 维
扩散的训练重建目标 = Q                # d 维
扩散输出 = Q*                         # d 维
```

原先名为 `concept` 的接口字段在 NewQ 中存放 `C_bar+R_q`，纯加权概念另存为 `fused_concept`。这样不必修改去噪网络；它的输入宽度仍为 `3*d+1`。R_q 已在训练目标和生成条件中体现，扩散输出后不再额外叠加 R_q。该接法使采样能利用题目残差条件，不保证采样结果逐值等于原始 Q 或逐值保留 R_q。

返回的组件还包括 `concept_ids`、`concept_weights`、`residual` 与 `question_embedding`，供检查使用；知识追踪下游仍读取原来的接口字段。

## 损失与参数预算

本次保持与基线相同的三项目标：

```text
loss = KT + contrastive_weight * LC + diffusion_weight * Diff
默认 = KT + 0.001 * LC + 0.01 * Diff
```

Diff 仍是真实扩散重建 MSE，使用基线的有效交互及嵌入维度归一化。讨论稿中的独立按题残差惩罚 L_R 本次未启用，因此无需额外初始化训练题目集合。验证和测试只计算预测损失，对比/扩散损失均为 null。统一 CSV 保持 `(KT, LC, Diff)` 三元组。

与同配置、同词表的完整 DiffuQKT 相比，新增参数恰为：

```text
(num_problems + 1 + 2 * embedding_dim) * question_embedding_dim
```

预算统计包含完整扩散、知识追踪及预测模块。构造模型时校验总参数不能超过同规模基线的 4 倍，超限直接报错并提示减小 r；默认 r=16 用于满足 3 倍以内的目标。网络输出维度始终为 embedding_dim，不会改成 r。

## 文件与配置

| 文件 | 作用 |
| --- | --- |
| `src/model/newq/model.py` | NewQ 专属参数、概念集合融合、残差、扩散条件适配及参数预算检查。 |
| `src/model/newq/__init__.py` | 导出 NewQ 类。 |
| `configs/models/newq.yaml` | r=16、注意力温度 1、与基线一致的扩散参数及损失权重；所有参数附有中文注释。 |
| `tests/test_newq.py` | 手工数值、题目条件权重、去重/顺序不变性、padding、三项任务梯度、单概念性质、基线退化、评估残差通路、因果性与参数上限检查。 |

注册及加载接入 `src/model/registry.py`、`src/model/__init__.py`、`src/utils/config.py` 与 `src/training/cli.py`。训练器、结果汇总、数据格式、DiffuQKT 与 LinearQ 的实现及配置均沿用现有版本。

## 训练与恢复

在项目根目录、已激活的 giakt-main 环境中运行：

```powershell
# 默认数据集 ASSIST09，最多 50 轮
python main.py --model newq --epochs 50 --device cuda

# 显式指定数据集
python main.py assistment2009 --model newq --epochs 50 --device cuda
python main.py assistment2017 --model newq --epochs 50 --device cuda

# 小规模流程检查；显式的 --epochs 覆盖 smoke 的轮数
python main.py assistment2009 --model newq --smoke --epochs 5 --device cuda

# 自动从 checkpoint 恢复 NewQ，无需再写 --model
python main.py assistment2017 --evaluate experiments/newq/assistment2017/newq_assistment2017_50epochs_1/best.pt --device cuda
```

正式训练输出自动位于 `experiments/newq/<数据集>/newq_<数据集>_<轮数>epochs_<编号>/`，结果按现有规则汇总到 `experiments/results.csv`。早停继续生效。模型名称与构造参数写入 checkpoint，包含题目向量维度和注意力温度；评估从保存的参数重建模型，并校验数据 metadata。

`--smoke` 沿用公共的小主干 d=32，同时把 NewQ 的 r 改为 4，以免缩小主干后题目表占比过大而超过参数上限；正式训练不使用 --smoke 时仍默认 d=128、r=16。smoke 不改变扩散步数和损失权重。

更多命令与默认值见 [统一命令说明](../src/training/commands.md)。

## 本次实际验证

75 项回归检查全部通过。KT、对比、扩散三个损失分别反向传播时，新题目向量、概念打分投影与残差投影在多概念批次上均有有限非零梯度；单概念打分梯度为零。均匀权重且残差为零时，训练和采样结果均退化到基线，比较允许 float32 加权求和与 sum/count 运算产生的微小舍入差异。

两个当前真实 processed 数据集各完成 5 轮 CUDA 流程检查，采用基线规模 d=128、100 步扩散、r=16，每个划分最多前 32 个窗口、长度最多 32、batch=8，每轮最多 2 个 batch。复用已有划分，不执行静态预处理。环境为 giakt-main 的 PyTorch 2.4.1+cu124，实际设备 NVIDIA GeForce RTX 3050 Ti Laptop GPU。

| 数据集 | 基线参数 | NewQ 参数 | 总参数倍率 | 最佳轮次 | 该轮训练 KT | 该轮训练 LC | 该轮训练 Diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSIST09 | 546165 | 832597 | 1.524 | 4 | 0.620542 | 2.457633 | 0.966840 |
| ASSIST17 | 525098 | 564570 | 1.075 | 2 | 0.711498 | 3.015646 | 0.828072 |

这些是小规模流程检查，不代表完整训练或预测性能结论。两次均完成训练、验证选模、权重保存、最终测试，再通过独立 main.py 命令自动识别 NewQ 并评估；AUC、ACC、BCE 与保存的最终测试结果一致，验证/测试辅助损失为 null，临时汇总 CSV 共六行。

另通过真实 `--smoke` 命令完成 ASSIST09 一轮 CUDA 检查（d=32、r=4、每个划分最多 8 个窗口、每轮最多 1 个 batch），确认默认小规模模式可训练、保存并恢复最佳权重。该模式在 ASSIST09/17 上的参数倍率分别为 2.229、1.223，均未超过上限。

当前 13 份已有 best.pt 仍全部严格加载成功，其中 DiffuQKT 5 份、LinearQ 8 份。DiffuQKT/LinearQ 实现、主入口、训练器与结果汇总、已有有效配置、数据 metadata 和正式 summary/history/results 共 44 个受保护文件的 SHA256 保持一致。临时检查脚本、配置、日志与权重已清理，正式实验目录未加入这些流程检查结果。
