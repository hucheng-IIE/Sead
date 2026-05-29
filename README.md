# Sead

## Semantics-Preserving Temporal Adversarial Graph Contrastive Learning for Event Prediction

本仓库为论文 **Semantics-Preserving Temporal Adversarial Graph Contrastive Learning for Event Prediction** 的代码实现，主要用于动态知识图谱上的事件预测任务。代码围绕时间序列事件图建模，将历史四元组构造成按时间组织的图快照，并预测后续时间步的关系级事件分布。

## 仓库结构

```text
Sead/
|-- README.md
|-- __init__.py
`-- src/
    |-- train.py                 # 训练、验证与测试入口
    |-- build_baseline.py        # 参数扩展与模型构建
    |-- models.py                # Sead/APEP 主模型
    |-- aggregators.py           # 时序图聚合与相关基线模块
    |-- propagations.py          # 图传播与 CompGCN 风格消息传递
    |-- Generator.py             # 对抗生成器
    |-- Discriminator.py         # 对抗判别器
    |-- data.py                  # PyTorch 数据集封装
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

## 主要模块

- `src/models.py`：定义 Sead 使用的 `APEP` 模型，包含时序关系表示学习、对抗扰动、判别器反馈与图对比学习目标。
- `src/aggregators.py`：实现事件图聚合器，其中 `aggregator_event_AEPE` 是 Sead 的核心时序图表示模块。
- `src/propagations.py`：实现面向有向动态图的 CompGCN 风格传播层。
- `src/Generator.py` 与 `src/Discriminator.py`：实现对抗训练组件。
- `src/data.py` 与 `src/event_data_processing.py`：负责事件四元组读取、时间步组织和训练样本构造。
- `src/utils.py`：提供数据加载、经验分布构造、图增强、损失函数和评价指标。

## 输出结果

训练过程中，模型检查点默认保存至：

```text
models/<DATASET_NAME>/
```

实验汇总结果会写入 `src/train.py` 中配置的 CSV 路径。复现实验时，可根据本地环境调整数据路径、模型保存路径和结果输出路径。

## 依赖说明

当前仓库主要包含源代码，未包含数据集、预处理后的图字典和依赖清单。代码中使用的主要库包括 PyTorch、DGL、NumPy、SciPy、scikit-learn、pandas、tqdm、torch-scatter 和 matplotlib。
