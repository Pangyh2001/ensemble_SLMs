# experiment_config.py
"""
GSM8K实验配置文件
"""

# 实验参数
EXPERIMENT_CONFIG = {
    "seeds": [42, 123, 0],           # 随机种子
    "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 阈值
    "num_runs": 10,                  # GSM8K每个问题的采样次数
    "embedding": "bert",             # 使用的embedding模型
    "gate_type": "mlp",              # 门控网络类型
    "batch_size": 64,                # 训练batch大小
    "learning_rate": 1e-4,           # 学习率
    "num_epochs": 50,                # 训练轮数
    "early_stopping_patience": 10,   # 早停耐心值
}

# 模型列表
GSM8K_MODELS = [
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