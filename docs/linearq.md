# LinearQ：线性增强题目表示

LinearQ 与原始 DiffuQKT 并列，通过 `--model linearq` 选择。两种模型共用基础嵌入、知识检索、状态对比学习和预测实现，区别是增强题目表示 Q* 的生成方式及对应的表示约束损失。

实现位于 `src/model/linearq/model.py`，直接继承 `src/model/diffuqkt/base.py` 的 `QuestionKTBase`。复用的嵌入、注意力与下游代码由完整 DiffuQKT 基线目录维护，LinearQ 目录只维护自己的表示与约束。目录与扩展规则见 [model_architecture.md](model_architecture.md)。

## 表示与学习方式

对题目 q 及其知识点 k：

```text
C = concept[k]
V = variation[k]
D = sigmoid(difficulty[q])
Q = C + D * V

E_q = question_embedding[q]
A = linear_a(E_q) = W_A * E_q
B = linear_b(E_q) = W_B * E_q
Q* = (C + A) + D * (V + B)
Q* - Q = A + D * B
```

C、V、A、B、Q、Q* 均为 d 维向量，D 是原始模型使用的一个可学习难度标量。E_q 是新增的可学习题目 ID 向量，维度由 question_embedding_dim 指定，null 时跟随 d。两个线性层参数互相独立，不包含偏置、激活、噪声采样或扩散时间步。即使两道题具有相同知识点和相同 D，也可以通过各自的 E_q 产生不同 A、B。

E_q 使用标准差 0.02 的正态分布初始化，让初始残差较小且两组线性权重和 E_q 都能从第一步开始获得梯度。这里的“拟合”随 KT、对比学习和残差约束联合进行，没有预先生成的扩散教师标签。正则项直接使用逐题残差 r_i=A_i+D_i*B_i，与 Q*-Q 等价，并对完整训练题目集合等权计算。

ASSIST09 的多个原子知识点继续分别对 C、V 取均值，然后加入该题的 A、B。一个作答仍只对应一个预测目标。padding ID 为 0，E_q 的 padding 行与梯度保持为 0，最终表示再次应用有效位置掩码。

## 下游计算与损失

Q* 进入原有知识检索器，得到主分支状态 h*；Q 通过同一个知识检索器得到原始分支状态 h。状态对比学习继续使用相同的正负样本规则、温度和预测掩码。预测头继续拼接 h* 与 Q* 预测答题正确性。

```text
DiffuQKT: loss = KT + 0.001 * LC + 0.01 * Diff
LinearQ:  loss = KT + contrastive_weight * LC + diffusion_weight * L_item
```

LinearQ 不实例化扩散网络，也不保存扩散日程和扩散参数。正则项为 `L_item = mean_i,dim((A_i + D_i*B_i)^2)`，等于所有训练题目残差平方和除以训练题目数及嵌入维度。训练开始前从运行时 train.npz 的有效位置提取去重题目 ID，排除 padding，包含仅出现在首个上下文位置的训练题目；验证/测试独有题目不加入集合。每次参数更新都在这组固定题目上重新计算正则，每题贡献一次，与当前 batch、交互频率、重叠窗口次数无关。

原始完整运行使用完整训练划分的题目集合；max_windows / max_seq_len 会缩小运行时训练数据范围，集合随之缩小。max_batches 只限制每轮迭代数，不改变已加载训练池的题目集合。题目 ID 集合缓存在模型 buffer 中，实际损失和梯度每一步重新计算，不能缓存为旧参数下的标量。

当前专属配置的权重为 0.03，本次按题等权修改保留该值。为兼容原训练与结果格式，复用 diffusion_loss / diffusion_weight 字段；对 LinearQ，它们表示 L_item 及其权重，CSV train 行为 `(KT, LC, L_item)`。该正则日志按训练步骤取均值；KT、LC 以及原始 DiffuQKT 的损失统计沿用原规则。验证/测试继续只计算预测损失，训练辅助损失为 null。

公共下游结构会在每个独立模型内正常训练；共用实现不会让两次实验共享同一组可变参数。LinearQ 在训练和评估时采用相同的 Q* 公式；评估关闭 Dropout 后，表示计算不依赖扩散随机种子。

## 运行、配置与恢复

在项目根目录运行：

```powershell
python main.py assistment2017 --model linearq --epochs 50 --device cuda
python main.py assistment2009 --model linearq --epochs 50 --device cuda

# 原始 DiffuQKT；未指定 --model 时也默认运行它
python main.py assistment2017 --model diffuqkt --epochs 50 --device cuda

# 自动从 checkpoint 识别 LinearQ；数据集需要与训练时一致
python main.py assistment2017 --evaluate experiments/linearq/assistment2017/linearq_assistment2017_50epochs_1/best.pt --device cuda
```

配置覆盖顺序为：命令行 > 数据集或自定义 YAML > 所选模型专属默认配置 > 公共默认配置。公共配置不再包含扩散专属参数；选择 LinearQ 时只合并 LinearQ 默认值。`question_embedding_dim: null` 表示随 embedding_dim 变化，也可以指定正整数。已有完整运行配置中的模型名称可以直接用于重启同一模型。

实验统一按 `experiments/<模型目录>/<数据集>/<实验名>/` 分组。原始 DiffuQKT 放在 diffuqk 下，单次实验沿用 `<数据集>_<轮数>epochs_<编号>`；LinearQ 放在 linearq 下，单次实验使用 `linearq_<数据集>_<轮数>epochs_<编号>`。两者分别编号，由根目录 experiments/results.csv 递归统一汇总。

新的 checkpoint、summary.json 和实际配置保存 model_name。评估按 checkpoint 中的模型名称和构造参数创建网络；未记录模型名称的旧 checkpoint 按 DiffuQKT 兼容处理。显式指定与 checkpoint 不符的 --model 会报错。所有评估继续校验 processed/metadata.json 的 SHA256。 新 LinearQ checkpoint 的 model_kwargs 还保存 train_item_ids，加载后可恢复相同正则集合。该集合使用非持久 buffer，不改变历史 state_dict 键名；旧 checkpoint 仍可评估。直接构造裸 LinearQ 做训练时，必须传入 train_item_ids 或调用 set_training_items；主训练入口自动处理。

当前题目集合初始化封装在 LinearQ 的 `prepare_training(dataset)` 中，由共享训练器在每次训练开始前统一调用。该方法返回需要写入 checkpoint 的 `train_item_ids`；模型名称专属的判断不再留在训练器中。

耗时从训练数据加载开始，到最佳权重最终测试结束，排除静态预处理，CSV 继续使用分钟和秒的 ambs 格式。全部命令维护在 [统一命令说明](../src/training/commands.md)。

## 首次接入验证（加入残差约束之前，2026-10-17）

- 52 项单元检查通过，覆盖公式、两项损失独立梯度、参数更新、padding、多知识点、因果性、下游一致性、模型选择和新旧 checkpoint 评估。
- 使用重构前保存的固定输入与随机种子逐值对比：原始 DiffuQKT 的初始化参数、state_dict 名称、训练前向与评估前向完全一致。现有 4 份原版 best.pt 也全部严格加载成功。
- 在 giakt-main 的 PyTorch 2.4.1+cu124、NVIDIA GeForce RTX 3050 Ti Laptop GPU 上，ASSIST09、ASSIST17 各完成 5 轮真实 CUDA 训练。采用 d=128，每个 split 最多前 32 个窗口、长度最多 32，batch=8，每轮最多 2 个 batch；复用当前 processed，不执行静态预处理。这是流程检查，不是收敛实验。

| 数据集 | 完成轮数 | 最佳轮次 | 该轮训练 KT | 该轮训练 LC | Diff |
| ------ | -------- | -------- | ----------- | ----------- | ---- |
| ASSIST09 | 5 | 5 | 0.491242 | 1.844687 | null |
| ASSIST17 | 5 | 4 | 0.675444 | 1.728902 | null |

两次训练均保存完整 checkpoint、summary 和三行 CSV，并在不指定 --model 的情况下用独立命令自动恢复 LinearQ 再次评估；AUC、ACC、loss 与最终测试一致。已有数据 metadata、正式 summary/history/results.csv 的哈希保持不变。临时权重、配置及验证脚本已清理。

## 按交互残差约束接入验证（上一版本）

新增残差 MSE 后，55 项检查全部通过，包含有效位置/嵌入维度归一化、padding 排除、两项线性头与题目向量的独立梯度、仅优化残差项使其下降以及权重作用。另通过真实 main.py 完成两轮临时小数据 CPU 训练，确认总损失等于 KT + 0.001*LC + 0.01*L_delta；该次记录的残差约为 0.00023929，CSV 第三项为该值，验证/测试辅助损失仍为 null，保存权重可重新评估。临时数据和权重已清理。

本次功能代码只修改 LinearQ.py 及其专属配置；DiffuQKT、公共 base、扩散/注意力实现、训练入口、训练器、注册表、公共配置和 DiffuQKT 配置共 10 个文件的 SHA256 均与修改前一致。相关测试与说明同步更新。题目向量维度保留当前配置的 8，未恢复成默认跟随 d 的设置。

## 按训练题目等权的残差约束验证（当前版本）

60 项检查通过。新增检查使用残差分别为 1、3 的两道题验证损失为 (1+9)/2=5，并确认训练题 ID 的重复和 padding 不增加权重；复制交互或改变 batch 中题目频次时，正则值和梯度逐值相同。另验证不同大小 batch 的正则日志按步骤平均、验证/测试独有题目不参与梯度，以及 checkpoint 恢复后训练题目集合一致。

使用当前两个真实 processed 数据集分别进行一轮 CPU 流程检查，每轮最多两个 batch，batch=8、d=128，不限制窗口数或序列长度，正则题目集合覆盖完整训练数据。ASSIST09 提取 16837 道训练题，ASSIST17 提取 2190 道训练题；两次均确认正则为有限正数、权重保留为 0.03、checkpoint 完整恢复题目集合，临时实验 CSV 正常生成。原数据 metadata 保持不变，临时数据与权重已清理。这是流程检查，不是收敛实验。

本次只改变 LinearQ 的正则及其训练数据接入和日志汇总；训练器新增的题目集合初始化仅在 model_name=linearq 时执行。DiffuQKT 模型、公共 base、扩散、注意力、main.py、公共配置与 DiffuQKT 配置共 7 个文件的 SHA256 与修改前一致；DiffuQKT 继续使用原有交互加权损失及统计规则。
