import os

# 目录配置
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "dataset"
RESULT_DIR = "results"
GATE_DIR = "GATE"
CHECKPOINT_DIR = "checkpoints"

# 创建必要的目录
os.makedirs(GATE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Embedding 模型配置
EMBEDDING_MODELS = {
    "bert": "bert-base-uncased",
    "e5-base": "intfloat/e5-base-v2",
    "e5-large": "intfloat/e5-large-v2",
    "gte-large": "Alibaba-NLP/gte-large-en-v1.5"
}

# 当前选择的 Embedding（全局共享，不参与训练）
SELECTED_EMBEDDING = "bert" 

# Gate网络架构类型
# 可选: "mlp", "attention", "resnet"
GATE_TYPE = "mlp"

# =============================================
# MMLU 模型配置
# =============================================
# 参与训练和集成的小模型（这些模型会训练Gate并参与集成）
MMLU_TRAIN_MODELS = [
    "gemma-2b",
    "gemma-7b",
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "phi-2",
    "deepseek-llm-7b-chat",
    "Llama-3.1-8B-Instruct",
    "Qwen2-7B-Instruct",
    "StableLM-Zephyr-3B",
    "Falcon3-10B-Instruct"
]

# 用于性能对比的大模型baseline（不参与训练，仅用于对比）
MMLU_BASELINE_MODELS = [
    "Llama-2-70b-hf",
    "Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-v0.1"
]

# 所有模型列表（用于数据加载）
MMLU_MODELS = MMLU_TRAIN_MODELS  # 训练时只使用训练模型

# =============================================
# GSM8K 模型配置
# =============================================
# 参与训练和集成的小模型
GSM8K_TRAIN_MODELS = [
    "gemma-2b",
    "gemma-7b",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "phi-2",
    "deepseek-llm-7b-chat",
    "Llama-3.1-8B-Instruct",
    "Qwen2-7B-Instruct",
    "StableLM-Zephyr-3B",
    "Falcon3-10B-Instruct"
]

# 用于性能对比的大模型baseline
GSM8K_BASELINE_MODELS = [
    "Llama-2-70b-chat-hf",
    "Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-v0.1"
]

# 所有模型列表
GSM8K_MODELS = GSM8K_TRAIN_MODELS

# 训练配置
TRAIN_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "early_stopping_patience": 10,
    "hidden_dim": 256,
    "dropout": 0.1,
    "max_length": 512,
    "embedding_name": SELECTED_EMBEDDING,
    "gate_type": GATE_TYPE,  # Gate网络架构
    # Attention Gate 特定参数
    "num_heads": 8,
    # ResNet Gate 特定参数
    "num_blocks": 4
}

# GSM8K 特定配置
GSM8K_CONFIG = {
    "num_runs": 10,  # 每个问题采样10次
    "max_answer_value": 1e6  # 答案值的合理上限
}

# 评估配置
EVAL_CONFIG = {
    "seed": 42,              # 固定随机种子
    "threshold": 0.5,        # 阈值方法的阈值
    "random_baseline": True  # 是否运行随机Baseline对比
}