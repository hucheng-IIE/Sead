# Semantics-Preserving Temporal Adversarial Graph Contrastive Learning for Event Prediction

本仓库为论文 **Semantics-Preserving Temporal Adversarial Graph Contrastive Learning for Event Prediction** 的代码实现。该论文已被 **ECML-PKDD 2026 Research Track** 录用。

## 方法概述

Sead 面向动态知识图谱上的多事件预测任务。给定历史时间窗口内的事件图序列，模型预测未来时间步可能共同发生的事件类型分布。每个事件被表示为包含头实体、关系类型、尾实体、时间戳和上下文文本的时序事件。

论文指出，直接将对抗图对比学习用于事件预测会破坏事件之间的语义关联。为此，Sead 通过语义保持的多视图对比学习，在提升噪声鲁棒性的同时保留事件语义关系。

整体框架包含三个部分：

1. **视图生成**：在原始干净视图之外构造三类辅助视图。
2. **语义感知事件编码器**：对不同视图进行编码，学习事件类型的完整语义表示。
3. **事件预测**：基于多视图对比学习得到的鲁棒表示预测未来事件类型分布。

## 核心思想

### 1. 多视图构造

- **Clean View**：保留原始时序事件图的结构和语义特征。
- **Semantics-preserving View**：利用事件上下文语义相似性构造语义保持视图。论文中使用冻结的 LLaMA 编码事件上下文，并基于 k 近邻相似度保留语义相近事件之间的关联。
- **Adversarial View**：对事件图结构和特征施加最坏情形扰动，用于提升模型面对噪声数据时的鲁棒性。
- **Temporal Perturbation View**：扰动事件在历史时间窗口中的时序特征，用于模拟真实数据中的时间噪声。

### 2. 语义感知事件编码器

语义感知事件编码器由图聚合和时间编码两部分组成。

- **图聚合**：根据语义分布自适应选择传播路径，减少无关实体带来的噪声；同时采用增量采样策略，避免关键语义实体在多跳传播中被丢弃。随后使用 CompGCN 风格的消息传递更新节点和事件边表示。
- **时间编码**：将同一事件类型在历史窗口中的语义表示加入位置编码后输入 Transformer，建模事件类型随时间演化的依赖关系，并得到最终事件类型表示。

### 3. 事件预测与训练目标

Sead 将干净视图中的事件类型表示与三类辅助视图中的表示进行对比学习，使同一事件类型在不同视图下保持一致，同时区分不同事件类型。最终表示输入分类器，预测未来时间步的事件类型分布。

训练目标由两部分组成：

- **图对比学习损失**：约束 clean view 与语义保持视图、对抗视图、时间扰动视图之间的表示一致性。
- **事件预测损失**：由于同一天可能发生多个事件类型，论文将标签定义为事件类型频率分布，并采用分布式预测目标进行优化。

## 仓库结构

```text
Sead/
|-- README.md
|-- __init__.py
`-- src/
    |-- train.py                 # 训练、验证与测试入口
    |-- build_baseline.py        # 参数扩展与模型构建
    |-- models.py                # Sead 主模型
    |-- aggregators.py           # 事件图聚合与表示学习模块
    |-- propagations.py          # 图传播与 CompGCN 风格消息传递
    |-- Generator.py             # 对抗学习相关模块
    |-- Discriminator.py         # 对抗学习相关模块
    |-- data.py                  # 数据集封装
    |-- event_data_processing.py # 事件数据预处理
    |-- utils.py                 # 数据读取、评价指标与辅助函数
    |-- modules_f.py             # 注意力等基础模块
    `-- modules/                 # memory/cache/TGN 相关组件
```

## 运行入口

主要实验入口为：

```bash
python src/train.py --model Sead --dataset <DATASET_NAME> --dp <DATA_ROOT>/
```

其中 `<DATA_ROOT>/<DATASET_NAME>/` 下需要包含：

```text
stat.txt
train.txt
valid.txt
test.txt
dg_dict.txt
```

`train.txt`、`valid.txt` 和 `test.txt` 采用四元组格式：

```text
head relation tail time
```

`stat.txt` 记录实体数与关系数，`dg_dict.txt` 存储按时间组织的 DGL 图字典。

## 实验设置

论文在 GDELT 真实事件数据上进行实验，选取 Egypt、Iran 和 Israel 三个国家数据集，时间粒度为天，时间范围为 2015 年 2 月至 2022 年 3 月。实验按 8:1:1 划分训练集、验证集和测试集，历史窗口设为 7 天，预测提前量设为 1 天，并使用 Recall、F1 和 F2 等指标评估多事件预测效果。

## 输出结果

训练过程中，模型检查点默认保存至：

```text
models/<DATASET_NAME>/
```

实验汇总结果会写入 `src/train.py` 中配置的 CSV 路径。复现实验时，可根据本地环境调整数据路径、模型保存路径和结果输出路径。

## 依赖说明

当前仓库主要包含源代码，未包含数据集、预处理后的图字典和依赖清单。代码中使用的主要库包括 PyTorch、DGL、NumPy、SciPy、scikit-learn、pandas、tqdm、torch-scatter 和 matplotlib。
