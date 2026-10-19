# 模型目录与复用约定

更新时间：2026-10-19。

DiffuQKT 是完整基线，基础嵌入、题目表示、知识检索、预测和对比学习代码均由它维护。LinearQ、NewQ、NewQ V2、DHA-KT 和后续实验模型在各自目录中实现差异，通过继承或组合复用基线层。DHA-KT 有独立两遍 forward，不沿用旧的增强→InfoNCE 闭环。

## 当前目录

```text
src/model/
├── __init__.py              # 对外导出已有模型类
├── registry.py              # 按模型名称创建网络、识别 checkpoint
├── diffuqkt/
│   ├── __init__.py          # 导出 DiffuQKT 和兼容的对比损失接口
│   ├── model.py             # 完整 DiffuQKT 的组装与扩散增强接口
│   ├── base.py              # 基础嵌入、Q=C+DV、下游知识追踪与损失组合
│   ├── attention.py         # AKT 遗忘注意力与知识检索器
│   └── diffusion.py         # 扩散日程、条件去噪、训练重建与评估采样
├── linearq/
│   ├── __init__.py          # 导出 LinearQ
│   └── model.py             # 线性增强、按训练题目等权残差约束及题目集合初始化
├── newq/
│   ├── __init__.py          # 导出 NewQ V1
│   └── model.py             # 初始 Q 的概念加权、题目残差与同维条件适配
├── newq_v2/
│   ├── __init__.py          # 导出独立 NewQV2
│   ├── model.py             # 独立代码、余弦融合、残差相加/拼接与宽度、四模式路由
│   ├── diffusion.py         # 独立残差条件及 2d 完整目标适配，复用基线扩散公式
│   └── diagnostics.py       # 按保存模式执行的只读表示/梯度诊断
└── dha_kt/
    ├── __init__.py          # 导出 DHAKT
    ├── model.py             # 独立两遍同化、配置/损失、阶段/恢复接口
    ├── item_encoder.py      # 不构造旧扩散的题目组件/融合适配
    ├── diffusion.py         # 轮廓 x0/DDIM 和 legacy 可微原循环适配
    ├── candidates.py        # 按键噪声、唯一候选、两后端和评估缓存
    ├── evidence.py          # 稳定混合、责任、矩、共享候选 dropout 预测头
    ├── retriever.py         # 组合旧层，分离题目 query 与历史 values
    ├── reference.py         # 固定 artifact 的 buffer、manifest 和 split/fold 查询
    └── diagnostics.py       # 只读概率/责任/校准/分组/梯度诊断
```

`diffuqkt/base.py` 中的 `QuestionKTBase` 属于基线实现，但同时提供给实验模型复用。DiffuQKT 继承它并接入条件扩散；LinearQ 直接继承它，以线性增强替换扩散；NewQ 继承完整 DiffuQKT，只重写初始题目组件。每个模型实例各自创建参数，代码复用不会让不同实验共享参数对象。

NewQV2 同样直接继承 DiffuQKT，重写 question_components 和 enhance_questions；两个路由字段控制 off/condition/target/both，另有条件 detach、相加/拼接及拼接维度标志。目标、独立条件与输出融合集中管理。它不继承 V1，不保留 V1 的共享题目表，也不新增模型标识。

迁移对应关系如下；旧的平铺源码不再保留第二份。

| 原路径 | 当前路径 |
| --- | --- |
| `src/model/base.py` | `src/model/diffuqkt/base.py` |
| `src/model/attention.py` | `src/model/diffuqkt/attention.py` |
| `src/model/diffusion.py` | `src/model/diffuqkt/diffusion.py` |
| `src/model/diffuqkt.py` | `src/model/diffuqkt/model.py` |
| `src/model/LinearQ.py` | `src/model/linearq/model.py` |

## 实验模型可以修改的边界

`QuestionKTBase.forward` 依次调用：

1. `question_components(batch)`：构造原始 Q 与增强需要的组成部分。
2. `enhance_questions(components, batch, generator)`：返回增强 Q* 与模型专属的表示损失。
3. 同一个知识检索器和预测头：Q* 进入预测分支，Q 进入训练时的状态对比分支，按原规则组合损失。

只研究 Q 构造的 NewQ 已放入 `src/model/newq/`，继承 `DiffuQKT` 并重写 `question_components`，从而沿用扩散与后续任务。保留扩散时，返回字典需要提供以下接口：

| 字段 | 形状 | 用途 |
| --- | --- | --- |
| `concept` | `[batch, time, d]` | 扩散的概念条件 |
| `conditioning` | `[batch, time, 1]` | 扩散的标量条件，基线为难度 D |
| `original` | `[batch, time, d]`，V2 可选 2d | 原始 Q，供下游对比分支；基线与 V1 也以它为扩散目标 |
| `semantic_original` | `[batch, time, d]` | V2 的纯语义 Q，off/condition 将其作为扩散目标；target/both 使用 original |
| `variation`、`scale` | 基线分别为 `[batch, time, d]`、`[batch, time, 1]` | 供基线公式与派生表示复用 |

输出需要保持维度与有效位置掩码约定。若新公式改变扩散条件的维度或语义，所需的适配也应放入 NewQ 自己的目录。若替换整个增强方式，可像 LinearQ 一样继承 `QuestionKTBase`，实现自身的增强与表示损失。新模型专属的数学函数、损失或编码器也放在该模型目录；多个模型明确复用的基线实现继续放在 `diffuqkt`。

NewQ 已按讨论稿的精简方案实现 `Q=C_bar+D*V_bar+R`，以 `C_bar+R` 作为原维度概念条件；没有重写扩散增强或主 forward，损失沿用基线三项。具体公式、参数预算与运行方式见 [newq.md](newq.md)。

NewQ V2 使用 8+8 独立代码，各模式 concept 均严格为纯 d 维 C_bar、semantic_original 为 Q_sem，original 按 residual_fusion 相加或拼接融合 R。concat 的 residual_concat_keep_2d=true/1 保留 2d，false/0 用共享线性层映射回 d；原始与增强分支共用规则。off/condition 生成语义后融合 R，target/both 生成完整 original 后直接使用。保留 2d 时，本模型重建同算法的 2d 答案/检索/预测层，并为完整扩散目标适配层宽；基线源代码不变。residual_condition_detach 仅 detach 独立条件，主 forward 与诊断路由继续复用。接口见 [newq_v2.md](newq_v2.md)。

## 训练前的模型专属准备

共享训练器在构造模型、读取运行时训练集后，检查模型是否实现 `prepare_training(dataset)`。有此方法时调用一次，再将返回的构造参数补充进 checkpoint 的 `model_kwargs`；没有此方法时直接进入训练。

LinearQ 在自己的 `model.py` 中实现该方法：从运行时训练集提取去重题目集合，初始化按题等权的残差约束，并返回 `train_item_ids`，使 checkpoint 恢复同一集合。训练器不再按 `model_name == 'linearq'` 分支初始化模型。DiffuQKT 不需要此步骤。训练器继续负责通用的优化器、训练循环、验证选模与结果保存。

模型还可提供 experiment_metadata，训练器将其输出写入启动日志、checkpoint 和 summary。V2 通过该接口保存请求开关、有效模式、目标/条件/输出路径和实际参数量；没有接口的模型保持既有字段。checkpoint 构造始终使用保存的 model_kwargs，不使用当前 YAML 推断旧模式。

V2 的 prepare_training 返回归一化路由、条件 detach、残差融合及拼接维度字段，使旧配置启动的新训练也显式保存默认值；有模型元数据接口时同步到实际运行配置。该步骤不读取或重建数据，也不改变旧 checkpoint 的加载默认值。

V2 另有 `diffusion_difficulty_condition` 布尔开关（默认 true），只控制训练去噪/逆扩散的显式 D 条件。generate_questions 关闭该输入时传入零值，保留 Q_sem 中的 D 及现有参数形状；prepare_training 和元数据同步保存开关。旧权重缺字段时按 true 恢复，其他模型及共享扩散实现不受影响。

DHA 的 prepare_training 只验证并载入预先构建的 reference artifact，核对运行范围，将 bank_schema 和归一化 profile 写入恢复参数。静态交叉拟合在 scripts/dha_kt/build_reference.py，运行时 reference.py 不训练教师、不重建统计。必要参考 buffer 与 manifest 自包含于 checkpoint，因此恢复不重新依赖 artifact 路径。

共享 trainer 增加可选 set_data_context、on_train_epoch_start、on_optimizer_step、selection_group、extra_metric_names 和 supports_resume 协议。DHA 用显式 train/valid/test 查询参考库，使用独立 dha_kt_stages.py 控制阶段；Adam 包含全部未来解冻参数，保存阶段与随机数后可恢复。没有这些 hook 的旧模型仍走原路径。DHA 的 pilot_loss 单独聚合，不借用 contrastive_loss；旧 CSV 三元组及其他模型日志含义保留。

题目查询与历史交互的隔离是 DHA 独立 forward 的原因：第二遍不能把答案条件均值当作 question 参数传回旧 retriever。它继承已有注意力层，在自己目录新增 forward_with_interactions；两遍共用一个检索器和预测头，query/key 始终来自 Q0，只有严格历史 values 使用同化后的交互。详细文件职责和实际命令见 [dha_kt_usage.md](dha_kt_usage.md)。

## `src` 其他目录如何管理

| 目录或模块 | 当前管理方式 | 后续需要按模型区分的条件 |
| --- | --- | --- |
| `src/data/` | 保留共同的运行时数据读取、有效掩码、预测掩码与批次拼接。五个现有模型使用同一批次格式；DHA 在模型内查询参考 buffer。 | 模型确实需要额外运行时输入时，再增加专属适配文件；静态数据转换仍放 `scripts/`，共用脚本在一级目录、数据集适配在对应子目录。 |
| `src/training/trainer.py` | 共同维护训练、评估、早停、checkpoint、计时和日志汇总，模型返回损失。 | 出现预训练阶段、交替优化或不同优化器组等训练策略时，再为该策略增加独立控制模块；仅改变 Q 或损失公式不需要复制训练器。 |
| `src/training/cli.py` | 保留一个入口，用 `--model` 选择模型。 | 新增模型时扩展可选名称；无需为每个模型复制 `main.py`。 |
| `src/training/results.py` | 保留统一指标及实验目录汇总。 | 新的损失名称需要显示时增加对应说明或列定义；不按模型复制 CSV 汇总逻辑。 |
| `src/utils/` | 共用配置解析、设备选择、随机种子、指标和文件校验。 | 模型专属表示、损失、数学函数应放回相应模型目录，避免混入通用工具。 |
| `src/visualization/` | 当前为空包。 | 有实际的模型专属绘图时再增加对应模块或子目录；共用训练曲线和指标绘图保持共享。 |

`configs/models/` 已有按模型区分的配置，位于 `src` 外，继续沿用。数据集配置保持独立，以便同一数据集比较多个模型。

## 注册与运行

新增模型时，需要增加其模型包与专属配置，在 `src/model/registry.py` 注册构造器，在 `src/utils/config.py` 的 `MODEL_CONFIGS` 中登记默认配置路径。CLI 的可选名称直接读取 MODEL_CONFIGS，无需再维护第二份名称列表。需要从 `src.model` 直接导入时，再补充根包的导出。NewQ 和 NewQ V2 均已完成接入。

现有命令保持不变：

```powershell
python main.py assistment2017 --model diffuqkt --epochs 50
python main.py assistment2017 --model linearq --epochs 50
python main.py assistment2017 --model newq --epochs 50
python main.py assistment2017 --model newq_v2 --epochs 100
```

`from src.model import DiffuQKT, LinearQ, NewQ, NewQV2` 以及 `from src.model.diffuqkt import DiffuQKT` 有效。内部直接导入旧平铺文件的代码需要使用上表的新路径。项目 checkpoint 保存参数字典及构造参数，不依赖平铺文件的位置；目录整理保持已有模型的参数名称、初始化顺序和计算公式。

完整运行命令见 [统一命令说明](../src/training/commands.md)，逐文件说明见 [project_files.md](project_files.md)。

## 本次整理验证

2026-10-18 完成以下检查：

- 现有 60 项回归检查全部通过。
- 用重排前保存的固定输入与随机种子，对两个模型的初始化参数、state_dict 键名、训练/评估输出和全部参数梯度逐值比较，均完全一致。
- 现有 14 份 best.pt 严格加载成功，其中 DiffuQKT 6 份、LinearQ 8 份。
- 两个模型分别通过真实 main.py 在临时小数据上完成一轮 CPU 训练、最佳权重恢复及独立评估；再次评估的 AUC、ACC、BCE 与原测试结果一致，统一 CSV 共生成六行。该检查用于验证入口及文件结构，不用于评价模型性能。
- 主入口、命令行解析、配置工具和五份有效配置共 8 个文件的 SHA256 保持一致；已有两套数据 metadata 与实验 summary/history/results 共 31 个文件的 SHA256 保持一致。临时验证脚本、数据与权重检查后清理。
