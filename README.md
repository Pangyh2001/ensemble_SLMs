# 环境
```bash
conda activate /data2/pyh/env/ensembleLLM
```
# TODO
- 数据集我已经处理好了，你可以直接复制过去。在/data2/pyh/ensembleLLM/Claude_code/dataset
- 现在的dataset里有多个SLM的训练和测试集，其中GSM8k的训练集有问题，有的模型不是7472条训练数据，这个我已经github提了issue，你检查一下代码是怎么训练的吧，看看有没有问题。
- MMLU的没问题。上述确定没问题了之后，整理一下不同encoder的所有结果吧，参考我发群里的格式。
- 检查出问题可以先发我看看。


# 小模型集成框架

这是一个基于门控网络的小模型集成框架，通过训练多个门控网络来学习每个小模型适合回答的问题类型，从而实现媲美大模型的性能。

## 框架结构

```
.
├── config.py              # 配置文件
├── data_loader.py         # 数据加载器
├── data_split.py          # 数据集划分
├── gate_model.py          # 门控网络模型定义
├── trainer.py             # 训练器
├── evaluator.py           # 评估器
├── main.py                # 主运行脚本
├── requirements.txt       # 依赖包
└── README.md             # 使用说明
```

## 核心思想

1. **门控网络**: 为每个小模型训练一个基于 Transformer Encoder + MLP 的门控网络
2. **积极性分数**: 门控网络输出 0-1 的分数，表示该模型对某问题的适合程度
3. **偏离程度**: 通过模型预测与真实标签的偏离程度来训练门控网络
4. **集成策略**: 
   - 阈值方法: 选择分数超过阈值的模型进行集成
   - 加权方法: 使用归一化的分数作为权重进行加权集成

## 数据格式

### MMLU 数据集
```
dataset/mmlu_hf/
├── gemma-2b/
│   ├── topic1.csv
│   ├── topic2.csv
│   └── ...
├── gemma-7b/
└── ...
```

每个 CSV 文件格式:
```csv
idx,question,prediction,label
0,"Question text","[-1.56, -1.38, -1.23, -1.40]",B
```

### GSM8K 数据集
```
dataset/gsm8k/
├── train.json
├── test.json
├── gemma-2b/
│   ├── train/
│   │   ├── run_6980_predictions.npy
│   │   └── run_6980_outputs.pkl
│   └── test/
└── ...
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 0. 评估baseline准确率

```bash
# 同时评估 MMLU 任务和 GSM8K 任务
python check_slm_accuracy_simple.py --gsm8k_gold ./dataset/gsm8k/test.json --gsm8k_pred /gsm8k/预测文件/npy格式 --mmlu_dir /mmlu/预测文件夹

# 比如：
# python check_slm_accuracy_simple.py --gsm8k_gold ./dataset/gsm8k/test.json --gsm8k_pred ./dataset/gsm8k/Llama-2-70b-chat-hf/test/run_1318_predictions.npy --mmlu_dir ./dataset/mmlu_hf/Llama-2-70b-hf
```

### 1. 完整流程
#### 1.1 标准完整流程（数据划分 + 训练 + 单阈值评估）

```bash
# MMLU 任务
python main.py --mode all --task mmlu

# GSM8K 任务
python main.py --mode all --task gsm8k

# 同时运行两个任务
python main.py --mode all --task both
```

#### 1.2 增强完整流程（数据划分 + 训练 + 多阈值评估）

```bash
# 运行完整流程并进行多阈值(0.1-0.9)评估
python main.py --mode full --task mmlu

# GSM8K任务完整流程
python main.py --mode full --task gsm8k

# 两个任务都运行
python main.py --mode full --task both
```

### 2. 分步执行
#### 2.1 仅划分数据集

```bash
python main.py --mode split --task mmlu
python main.py --mode split --task gsm8k

# 或者直接运行：
python data_split.py --task mmlu
python data_split.py --task gsm8k
```

#### 2.2 仅训练门控网络

```bash
python main.py --mode train --task mmlu
python main.py --mode train --task gsm8k

# 或者直接运行：
python trainer.py --task mmlu
python trainer.py --task gsm8k
```

#### 2.3 仅评估

```bash
# 单阈值评估
python main.py --mode eval --task mmlu
python main.py --mode eval --task gsm8k

# 多阈值评估（快速测试不同阈值效果）
python main.py --mode threshold-only --task mmlu

# 或者直接运行：
python evaluator.py --task mmlu --all-thresholds
python evaluator.py --task gsm8k --multi-threshold
```

### 3. 选择embedding

支持通过 --embedding 参数选择语义提取模型：

bert: bert-base-uncased (默认)

e5-base: intfloat/e5-base-v2

e5-large: intfloat/e5-large-v2

gte-large: Alibaba-NLP/gte-large-en-v1.5

minilm: sentence-transformers/all-MiniLM-L6-v2

```Bash

# 运行完整流程 (MMLU 任务, 使用 E5-Base)
python main.py --mode all --task mmlu --embedding e5-base

# 仅训练 GSM8K 门控
python main.py --mode train --task gsm8k --embedding bert

# 仅评估
python main.py --mode eval --task mmlu
```

### 4. 自定义阈值参数

```bash
# 自定义阈值范围和步长
python main.py --mode full --task mmlu --threshold-range 0.2,0.8 --threshold-step 0.05
```

### 5. 结果解读
评估完成后会在终端输出以下指标，并保存详细日志至 results/{task}/detailed_logs.json：

threshold: 门控阈值筛选后的准确率

threshold_random_baseline: 随机选择相同数量模型的准确率

sampling: 门控概率采样后的准确率

sampling_random_baseline: 随机采样相同数量模型的准确率

weighted: 门控加权集成后的准确率

weighted_random_baseline: 简单平均集成（Simple Averaging）的准确率

## 配置说明

在 `config.py` 中可以修改以下配置:

### 训练配置
```python
TRAIN_CONFIG = {
    "batch_size": 32,           # 批次大小
    "learning_rate": 1e-4,      # 学习率
    "num_epochs": 50,           # 训练轮数
    "early_stopping_patience": 10,  # 早停耐心值
    "hidden_dim": 256,          # 隐藏层维度
    "num_heads": 8,             # 注意力头数
    "num_layers": 4,            # Transformer 层数
    "dropout": 0.1,             # Dropout 率
    "max_length": 512           # 最大序列长度
}
```

### 评估配置
```python
EVAL_CONFIG = {
    "threshold_method": True,   # 是否使用阈值方法
    "score_threshold": 0.5,     # 分数阈值
    "use_both_methods": True    # 是否同时使用两种方法评估
}
```

## 模型说明

### 参与集成的模型 (训练使用)
- **MMLU**: gemma-2b, gemma-7b, Llama-2-7b-hf, Llama-2-13b-hf, phi-2
- **GSM8K**: gemma-2b, gemma-7b, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, phi-2

### 对比基准模型 (不参与训练，仅用于性能对比)
- Llama-2-70b-hf
- Mistral-7B-Instruct-v0.2
- Mixtral-8x7B-v0.1

## 损失函数

### MMLU
通过模型输出的置信度与真实标签计算偏离程度：
```
偏离程度 = 1 - P(正确答案)
积极性分数 ≈ 1 - 偏离程度
损失 = MSE(积极性分数, 1 - 偏离程度)
```

### GSM8K
使用 10 次采样的错误率作为偏离程度：
```
偏离程度 = 错误次数 / 总采样次数
积极性分数 ≈ 1 - 偏离程度
损失 = MSE(积极性分数, 1 - 偏离程度)
```

## 输出文件

- **门控网络模型**: `GATE/{model_name}_{task_type}.pt`
- **数据划分**: `dataset/splits/{task}_train.pkl`, `dataset/splits/{task}_test.pkl`
- **评估结果**: `results/{task}/evaluation_results.json`

## 评估结果格式

```json
{
    "ensemble_threshold": 0.85,
    "ensemble_weighted": 0.87,
    "single_gemma-2b": 0.65,
    "single_gemma-7b": 0.72,
    "single_Llama-2-7b-hf": 0.70,
    "single_Llama-2-13b-hf": 0.78,
    "single_phi-2": 0.68
}
```

## 注意事项

1. 首次运行需要下载预训练的 BERT 模型，可能需要一些时间
2. 训练过程中会自动保存最佳模型，支持断点续训
3. 建议使用 GPU 进行训练以加快速度
4. 数据集路径需要按照指定格式组织

## 扩展性

框架设计具有良好的扩展性:
- 可以轻松添加新的小模型
- 可以调整门控网络架构
- 可以实现自定义的集成策略
- 支持更多数据集和任务类型

## 更新特性

- **灵活的 Embedding 支持**: 支持 BERT, E5, GTE 等多种 Embedding 模型。
- **简化的架构**: 移除了冗余的 Transformer Encoder，采用直接的 Projection 结构。
- **增强的评估策略**: 
  1. **阈值筛选 (Threshold)**: 仅使用置信度高于阈值的模型。
  2. **概率采样 (Sampling)**: 将置信度作为概率进行采样（固定随机种子，可复现）。
  3. **加权集成 (Weighted)**: 基于 Softmax 归一化分数的加权投票。
- **基线对比**: 每种策略均包含对应的“随机选择”基线，以验证门控有效性。
- **GSM8K 完整支持**: 修复了 GSM8K 数据加载与解空间对齐的逻辑。