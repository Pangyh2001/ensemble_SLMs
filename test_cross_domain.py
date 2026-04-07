"""
跨领域泛化能力测试脚本

实验目的：测试门控架构在跨领域情况下的泛化能力。
- 用GSM8K训练的门控 在MMLU测试集上测试
- 用MMLU训练的门控 在GSM8K测试集上测试

实验配置：
- 种子: 0, 42, 123（结果取平均值±标准差）
- 阈值: 0.1, 0.2, ..., 0.9
- 门控模型保存在 gate_kua/ 目录
- 支持断点续训（已有模型不重复训练）
- 结果保存为 results_kua.csv
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import argparse
import random
import gc
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_loader import (load_mmlu_data, load_gsm8k_data,
                         load_gsm8k_raw_predictions, compute_gsm8k_accuracy,
                         normalize_question)
from embedding_manager import get_embedding_manager, EmbeddingManager
from gate_model import GateNetwork
from trainer import (
    MMLUEmbeddingDataset, GSM8KEmbeddingDataset,
    collate_fn_mmlu, collate_fn_gsm8k,
    compute_deviation_mmlu_relative, compute_deviation_gsm8k_relative
)

# =============================================
# 实验配置
# =============================================
CROSS_CONFIG = {
    "seeds": [0, 42, 123],
    "thresholds": [round(0.1 * i, 1) for i in range(1, 10)],  # 0.1 ~ 0.9
    "embedding_key": "bert",
    "gate_type": "mlp",
    "gate_dir": "gate_kua",           # 门控模型存放目录
    "split_dir": "splits_kua",        # 数据划分存放目录
    "results_file": "results_kua.csv",
    "batch_size": 64,
}


def set_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================
# STEP 1: 数据准备
# =============================================
def prepare_mmlu_data_custom(seed):
    """
    MMLU数据准备：将所有子领域8:2分，
    "8"部分整合打乱作为训练集，"2"部分整合打乱作为测试集。
    每个seed对应一套不同的划分。
    """
    split_dir = os.path.join(CUR_DIR, "dataset", CROSS_CONFIG["split_dir"])
    os.makedirs(split_dir, exist_ok=True)

    train_path = os.path.join(split_dir, f"mmlu_train_seed{seed}.pkl")
    test_path = os.path.join(split_dir, f"mmlu_test_seed{seed}.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  MMLU split for seed={seed} already exists, skipping.")
        return

    print(f"  Preparing MMLU split for seed={seed}...")
    set_seed(seed)

    # 加载所有MMLU数据
    data_dict, questions, labels, topics = load_mmlu_data(MMLU_TRAIN_MODELS)

    # 获取所有子领域
    unique_topics = sorted(set(topics))
    print(f"    Total topics: {len(unique_topics)}")
    print(f"    Total samples: {len(questions)}")

    # 对每个子领域进行8:2划分
    train_indices = []
    test_indices = []

    for topic in unique_topics:
        topic_indices = [i for i, t in enumerate(topics) if t == topic]
        np.random.shuffle(topic_indices)

        split_point = int(len(topic_indices) * 0.8)
        train_indices.extend(topic_indices[:split_point])
        test_indices.extend(topic_indices[split_point:])

    # 打乱顺序
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # 构造数据
    train_data = {
        'data': {m: data_dict[m][train_indices] for m in MMLU_TRAIN_MODELS},
        'questions': [questions[i] for i in train_indices],
        'labels': [labels[i] for i in train_indices],
        'topics': [topics[i] for i in train_indices],
    }
    test_data = {
        'data': {m: data_dict[m][test_indices] for m in MMLU_TRAIN_MODELS},
        'questions': [questions[i] for i in test_indices],
        'labels': [labels[i] for i in test_indices],
        'topics': [topics[i] for i in test_indices],
    }

    with open(train_path, "wb") as f:
        pkl.dump(train_data, f)
    with open(test_path, "wb") as f:
        pkl.dump(test_data, f)

    print(f"    Train: {len(train_indices)}, Test: {len(test_indices)}")
    print(f"    Saved to {split_dir}")


def prepare_gsm8k_data_custom():
    """
    GSM8K数据准备：使用官方划分（train/test），所有seed共用。
    """
    split_dir = os.path.join(CUR_DIR, "dataset", CROSS_CONFIG["split_dir"])
    os.makedirs(split_dir, exist_ok=True)

    train_path = os.path.join(split_dir, "gsm8k_train.pkl")
    test_path = os.path.join(split_dir, "gsm8k_test.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("  GSM8K data already prepared, skipping.")
        return

    print("  Preparing GSM8K data (official train/test split)...")
    input_dir = os.path.join(CUR_DIR, DATA_DIR, "gsm8k")

    # 训练集
    train_predictions, train_questions, train_labels = load_gsm8k_raw_predictions(
        input_dir, GSM8K_TRAIN_MODELS,
        dataset_name="train",
        num_runs=GSM8K_CONFIG["num_runs"],
        num_samples=None
    )
    train_data = {
        'raw_predictions': train_predictions,
        'questions': train_questions,
        'labels': train_labels
    }

    # 测试集
    test_predictions, test_questions, test_labels = load_gsm8k_raw_predictions(
        input_dir, GSM8K_TRAIN_MODELS,
        dataset_name="test",
        num_runs=GSM8K_CONFIG["num_runs"],
        num_samples=None
    )
    test_data = {
        'raw_predictions': test_predictions,
        'questions': test_questions,
        'labels': test_labels
    }

    with open(train_path, "wb") as f:
        pkl.dump(train_data, f)
    with open(test_path, "wb") as f:
        pkl.dump(test_data, f)

    print(f"    Train: {len(train_questions)}, Test: {len(test_questions)}")


def prepare_all_data():
    """准备所有数据"""
    print("=" * 80)
    print("STEP 1: Data Preparation")
    print("=" * 80)

    # GSM8K: 所有seed共享同一份数据
    prepare_gsm8k_data_custom()

    # MMLU: 每个seed一份不同的划分
    for seed in CROSS_CONFIG["seeds"]:
        prepare_mmlu_data_custom(seed)

    print("Data preparation complete!\n")


# =============================================
# STEP 2: Embedding预计算
# =============================================
def precompute_embeddings_custom(embedding_manager, task_type, seed=None):
    """
    预计算embedding并缓存。
    MMLU: 每个seed有不同的划分，所以缓存区分seed。
    GSM8K: 所有seed共享同一份数据。
    """
    cache_dir = os.path.join(CUR_DIR, "dataset", "embedding_cache_kua")
    os.makedirs(cache_dir, exist_ok=True)

    if task_type == "mmlu":
        cache_file = os.path.join(cache_dir, f"mmlu_bert_seed{seed}.pkl")
    else:
        cache_file = os.path.join(cache_dir, f"gsm8k_bert.pkl")

    if os.path.exists(cache_file):
        print(f"  Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pkl.load(f)
        return cached['train_embeddings'], cached['test_embeddings']

    split_dir = os.path.join(CUR_DIR, "dataset", CROSS_CONFIG["split_dir"])

    if task_type == "mmlu":
        train_pkl = os.path.join(split_dir, f"mmlu_train_seed{seed}.pkl")
        test_pkl = os.path.join(split_dir, f"mmlu_test_seed{seed}.pkl")
    else:
        train_pkl = os.path.join(split_dir, "gsm8k_train.pkl")
        test_pkl = os.path.join(split_dir, "gsm8k_test.pkl")

    with open(train_pkl, "rb") as f:
        train_data = pkl.load(f)
    with open(test_pkl, "rb") as f:
        test_data = pkl.load(f)

    print(f"  Computing train embeddings for {task_type} (seed={seed})...")
    train_embeddings = embedding_manager.encode_batch(train_data['questions'], batch_size=32)

    print(f"  Computing test embeddings for {task_type} (seed={seed})...")
    test_embeddings = embedding_manager.encode_batch(test_data['questions'], batch_size=32)

    with open(cache_file, 'wb') as f:
        pkl.dump({
            'train_embeddings': train_embeddings,
            'test_embeddings': test_embeddings,
        }, f)

    print(f"  Saved embeddings: train={train_embeddings.shape}, test={test_embeddings.shape}")
    return train_embeddings, test_embeddings


# =============================================
# STEP 3: 训练门控网络
# =============================================
def get_gate_model_path(train_task, model_name, seed):
    """获取门控模型保存路径"""
    gate_dir = os.path.join(CUR_DIR, CROSS_CONFIG["gate_dir"])
    os.makedirs(gate_dir, exist_ok=True)
    return os.path.join(
        gate_dir,
        f"{train_task}_{model_name}_{CROSS_CONFIG['gate_type']}_seed{seed}.pt"
    )


def is_gate_trained(train_task, model_name, seed):
    """检查门控模型是否已训练"""
    path = get_gate_model_path(train_task, model_name, seed)
    if os.path.exists(path) and os.path.getsize(path) > 50 * 1024:
        return True
    return False


def train_single_gate(train_task, model_name, model_idx, embedding_manager, seed):
    """训练单个门控网络"""
    if is_gate_trained(train_task, model_name, seed):
        print(f"  [SKIP] {model_name} (seed={seed}) already trained.")
        return

    print(f"  [TRAIN] {model_name} (seed={seed})...")
    set_seed(seed)

    split_dir = os.path.join(CUR_DIR, "dataset", CROSS_CONFIG["split_dir"])

    if train_task == "mmlu":
        train_pkl = os.path.join(split_dir, f"mmlu_train_seed{seed}.pkl")
        model_list = MMLU_TRAIN_MODELS
    else:
        train_pkl = os.path.join(split_dir, "gsm8k_train.pkl")
        model_list = GSM8K_TRAIN_MODELS

    with open(train_pkl, "rb") as f:
        train_data = pkl.load(f)

    # 获取embedding
    if train_task == "mmlu":
        train_embeddings, _ = precompute_embeddings_custom(
            embedding_manager, "mmlu", seed=seed)
    else:
        train_embeddings, _ = precompute_embeddings_custom(
            embedding_manager, "gsm8k")

    embedding_dim = embedding_manager.get_encoder_dim()

    # 创建数据集
    if train_task == "mmlu":
        dataset = MMLUEmbeddingDataset(
            train_embeddings, train_data['labels'],
            model_name, train_data['data'], model_list
        )
        collate_fn = collate_fn_mmlu
    else:
        dataset = GSM8KEmbeddingDataset(
            train_embeddings, train_data['raw_predictions'],
            train_data['labels'], model_idx
        )
        collate_fn = collate_fn_gsm8k

    # 划分训练/验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=CROSS_CONFIG["batch_size"],
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=CROSS_CONFIG["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gate_model = GateNetwork(
        input_dim=embedding_dim,
        gate_type=CROSS_CONFIG["gate_type"],
        hidden_dim=TRAIN_CONFIG["hidden_dim"],
        num_heads=TRAIN_CONFIG.get("num_heads", 8),
        num_blocks=TRAIN_CONFIG.get("num_blocks", 4),
        dropout=TRAIN_CONFIG["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(gate_model.parameters(), lr=TRAIN_CONFIG["learning_rate"])

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        # 训练
        gate_model.train()
        train_loss = 0
        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            labels_batch = batch['labels']
            all_predictions = batch['all_predictions'].to(device)
            midx = batch['model_idx']

            scores = gate_model(embeddings)

            if train_task == "mmlu":
                deviation = compute_deviation_mmlu_relative(all_predictions, labels_batch, midx)
            else:
                deviation = compute_deviation_gsm8k_relative(all_predictions, labels_batch, midx)

            target = (1 - deviation).unsqueeze(1)
            loss = torch.nn.functional.mse_loss(scores, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        gate_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                labels_batch = batch['labels']
                all_predictions = batch['all_predictions'].to(device)
                midx = batch['model_idx']

                scores = gate_model(embeddings)
                if train_task == "mmlu":
                    deviation = compute_deviation_mmlu_relative(all_predictions, labels_batch, midx)
                else:
                    deviation = compute_deviation_gsm8k_relative(all_predictions, labels_batch, midx)
                target = (1 - deviation).unsqueeze(1)
                loss = torch.nn.functional.mse_loss(scores, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': gate_model.state_dict(),
                'gate_type': CROSS_CONFIG["gate_type"],
                'seed': seed,
                'best_val_loss': best_val_loss,
                'train_task': train_task,
            }, get_gate_model_path(train_task, model_name, seed))
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    print(f"    Done. best_val_loss={best_val_loss:.4f}")

    del gate_model, optimizer, train_loader, val_loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_all_gates():
    """训练所有门控网络（支持断点续训）"""
    print("=" * 80)
    print("STEP 2: Training Gate Networks")
    print("=" * 80)

    embedding_manager = get_embedding_manager(
        CROSS_CONFIG["embedding_key"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for train_task in ["gsm8k", "mmlu"]:
        model_list = GSM8K_TRAIN_MODELS if train_task == "gsm8k" else MMLU_TRAIN_MODELS

        print(f"\n{'='*60}")
        print(f"Training gates on {train_task.upper()} ({len(model_list)} models)")
        print(f"{'='*60}")

        for seed in CROSS_CONFIG["seeds"]:
            print(f"\n--- Seed={seed} ---")

            # 统计跳过和需要训练的
            skip_count = sum(1 for m in model_list if is_gate_trained(train_task, m, seed))
            need_count = len(model_list) - skip_count
            print(f"  Already trained: {skip_count}, Need training: {need_count}")

            if need_count == 0:
                print("  All gates for this seed already trained, skipping.")
                continue

            for i, model_name in enumerate(model_list):
                train_single_gate(train_task, model_name, i, embedding_manager, seed)

    print("\nAll gate training complete!\n")


# =============================================
# STEP 4: 加载门控模型
# =============================================
def load_gate_models(train_task, model_list, seed):
    """加载一组训练好的门控模型"""
    embedding_manager = get_embedding_manager(
        CROSS_CONFIG["embedding_key"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    embedding_dim = embedding_manager.get_encoder_dim()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gate_models = {}
    for model_name in model_list:
        path = get_gate_model_path(train_task, model_name, seed)
        if not os.path.exists(path):
            print(f"  WARNING: Gate model not found: {path}")
            continue

        gate_model = GateNetwork(
            input_dim=embedding_dim,
            gate_type=CROSS_CONFIG["gate_type"],
            hidden_dim=TRAIN_CONFIG["hidden_dim"],
            num_heads=TRAIN_CONFIG.get("num_heads", 8),
            num_blocks=TRAIN_CONFIG.get("num_blocks", 4),
            dropout=TRAIN_CONFIG["dropout"]
        ).to(device)

        checkpoint = torch.load(path, map_location=device)
        gate_model.load_state_dict(checkpoint['model_state_dict'])
        gate_model.eval()
        gate_models[model_name] = gate_model

    return gate_models


# =============================================
# STEP 5: 跨领域评估
# =============================================
def evaluate_cross_domain(train_task, eval_task, seed, threshold):
    """
    跨领域评估：用 train_task 训练的门控，在 eval_task 测试集上评估。

    返回:
        accuracy: DeGater准确率
        avg_activated: 平均激活模型数
        random_accuracy: random baseline准确率
    """
    # 注意：门控是在 train_task 的模型上训练的，模型列表一致
    # 但跨领域时，MMLU模型列表和GSM8K模型列表的模型名可能略有不同
    # (比如 Llama-2-7b-hf vs Llama-2-7b-chat-hf)
    # 为了跨领域测试，我们需要用同一套模型名
    # 这里我们使用 train_task 的模型列表（因为门控是按这个列表训练的）
    # 但评估时需要在 eval_task 的数据上找到对应的模型数据

    if train_task == "gsm8k":
        gate_model_list = GSM8K_TRAIN_MODELS  # 门控对应的模型列表
    else:
        gate_model_list = MMLU_TRAIN_MODELS

    if eval_task == "mmlu":
        eval_model_list = MMLU_TRAIN_MODELS
    else:
        eval_model_list = GSM8K_TRAIN_MODELS

    # 加载门控模型
    gate_models = load_gate_models(train_task, gate_model_list, seed)
    if len(gate_models) == 0:
        print(f"  ERROR: No gate models loaded for {train_task}, seed={seed}")
        return 0.0, 0.0, 0.0

    # 加载评估数据
    split_dir = os.path.join(CUR_DIR, "dataset", CROSS_CONFIG["split_dir"])
    if eval_task == "mmlu":
        test_pkl = os.path.join(split_dir, f"mmlu_test_seed{seed}.pkl")
    else:
        test_pkl = os.path.join(split_dir, "gsm8k_test.pkl")

    with open(test_pkl, "rb") as f:
        test_data = pkl.load(f)

    # 获取测试集embedding
    embedding_manager = get_embedding_manager(
        CROSS_CONFIG["embedding_key"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if eval_task == "mmlu":
        _, test_embeddings = precompute_embeddings_custom(
            embedding_manager, "mmlu", seed=seed)
    else:
        _, test_embeddings = precompute_embeddings_custom(
            embedding_manager, "gsm8k")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 建立门控模型名 -> 评估模型名的映射
    # 两个列表的模型应该一一对应（顺序一致，名字可能略有不同）
    # 这里我们按索引对应
    num_models = min(len(gate_model_list), len(eval_model_list))

    # 构建映射：gate模型索引 -> eval模型名
    gate_to_eval = {}
    for idx in range(num_models):
        gate_name = gate_model_list[idx]
        eval_name = eval_model_list[idx]
        if gate_name in gate_models:
            gate_to_eval[gate_name] = eval_name

    if len(gate_to_eval) == 0:
        print("  ERROR: No valid gate-to-eval mapping found.")
        return 0.0, 0.0, 0.0

    # 评估
    num_questions = len(test_data['questions'])
    num_embeddings = test_embeddings.shape[0]
    total = min(num_questions, num_embeddings)

    correct_degater = 0
    correct_random = 0
    total_activated = 0

    np.random.seed(seed)  # 固定random baseline的种子

    for i in range(total):
        embedding = test_embeddings[i].unsqueeze(0).to(device)

        # 计算所有门控分数
        scores = {}
        with torch.no_grad():
            for gate_name, gate_model in gate_models.items():
                if gate_name in gate_to_eval:
                    s = gate_model(embedding).item()
                    scores[gate_name] = s

        # 阈值选择
        selected_gate_names = [name for name, s in scores.items() if s > threshold]
        if len(selected_gate_names) == 0:
            # 没有超过阈值的，选分数最高的
            best_gate_name = max(scores, key=scores.get)
            selected_gate_names = [best_gate_name]

        num_activated = len(selected_gate_names)
        total_activated += num_activated

        # 转换为评估模型名
        selected_eval_names = [gate_to_eval[gn] for gn in selected_gate_names]

        # DeGater预测
        pred_degater = _predict_ensemble(eval_task, i, selected_eval_names,
                                         eval_model_list, test_data)
        true_label = test_data['labels'][i]
        if _check_correct(eval_task, pred_degater, true_label):
            correct_degater += 1

        # Random baseline: 随机选同样数量的模型
        random_eval_names = list(np.random.choice(
            eval_model_list, size=num_activated, replace=False))
        pred_random = _predict_ensemble(eval_task, i, random_eval_names,
                                        eval_model_list, test_data)
        if _check_correct(eval_task, pred_random, true_label):
            correct_random += 1

    accuracy = correct_degater / total if total > 0 else 0.0
    avg_activated = total_activated / total if total > 0 else 0.0
    random_accuracy = correct_random / total if total > 0 else 0.0

    # 清理
    for m in gate_models.values():
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return accuracy, avg_activated, random_accuracy


def _predict_ensemble(task_type, idx, selected_models, model_list, test_data):
    """集成预测（硬投票）"""
    if task_type == "mmlu":
        # MMLU: 加和 softmax概率，再取argmax
        total_logits = np.zeros(4)
        model_predictions = test_data['data']
        for m in selected_models:
            if m in model_predictions:
                total_logits += np.exp(model_predictions[m][idx])
        return ['A', 'B', 'C', 'D'][np.argmax(total_logits)]
    else:
        # GSM8K: 多数投票
        raw_preds = test_data['raw_predictions']
        answers = []
        for m in selected_models:
            m_idx = model_list.index(m)
            model_runs = raw_preds[idx, m_idx, :]
            valid_answers = [a for a in model_runs if not np.isnan(a)]
            if valid_answers:
                rounded = [round(float(a), 4) for a in valid_answers]
                most_common = Counter(rounded).most_common(1)[0][0]
                answers.append(most_common)

        if not answers:
            return np.nan
        rounded_all = [round(float(a), 4) for a in answers]
        return Counter(rounded_all).most_common(1)[0][0]


def _check_correct(task_type, pred, label):
    """检查预测是否正确"""
    if task_type == "mmlu":
        return pred == label
    else:
        try:
            return abs(float(pred) - float(label)) < 1e-4
        except:
            return False


# =============================================
# STEP 6: 运行完整实验并生成结果
# =============================================
def run_full_experiment():
    """运行完整的跨领域实验"""
    print("=" * 80)
    print("STEP 3: Cross-Domain Evaluation")
    print("=" * 80)

    seeds = CROSS_CONFIG["seeds"]
    thresholds = CROSS_CONFIG["thresholds"]

    # 跨领域实验配置:
    # 1. GSM8K训练 -> MMLU测试
    # 2. MMLU训练 -> GSM8K测试
    cross_pairs = [
        ("gsm8k", "mmlu"),  # 用GSM8K训练的门控测MMLU
        ("mmlu", "gsm8k"),  # 用MMLU训练的门控测GSM8K
    ]

    all_results = []

    for train_task, eval_task in cross_pairs:
        print(f"\n{'='*70}")
        print(f"Cross-Domain: Train on {train_task.upper()} -> Eval on {eval_task.upper()}")
        print(f"{'='*70}")

        for threshold in thresholds:
            print(f"\n  Threshold={threshold:.1f}")

            accs = []
            act_counts = []
            rand_accs = []

            for seed in seeds:
                print(f"    Seed={seed}...", end=" ")
                acc, avg_act, rand_acc = evaluate_cross_domain(
                    train_task, eval_task, seed, threshold)
                accs.append(acc)
                act_counts.append(avg_act)
                rand_accs.append(rand_acc)
                print(f"DeGater={acc:.4f}, AvgAct={avg_act:.2f}, Random={rand_acc:.4f}")

            # 计算平均值和标准差
            mean_acc = np.mean(accs)
            std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
            mean_act = np.mean(act_counts)
            std_act = np.std(act_counts, ddof=1) if len(act_counts) > 1 else 0.0
            mean_rand = np.mean(rand_accs)
            std_rand = np.std(rand_accs, ddof=1) if len(rand_accs) > 1 else 0.0

            all_results.append({
                "train_task": train_task.upper(),
                "eval_task": eval_task.upper(),
                "threshold": threshold,
                "degater_accuracy": f"{mean_acc:.4f}+-{std_acc:.4f}",
                "avg_activated_models": f"{mean_act:.2f}+-{std_act:.2f}",
                "random_baseline_accuracy": f"{mean_rand:.4f}+-{std_rand:.4f}",
                # 也保存原始数值方便后续分析
                "degater_acc_mean": mean_acc,
                "degater_acc_std": std_acc,
                "avg_act_mean": mean_act,
                "avg_act_std": std_act,
                "random_acc_mean": mean_rand,
                "random_acc_std": std_rand,
            })

            print(f"    => DeGater: {mean_acc:.4f}+-{std_acc:.4f}, "
                  f"AvgAct: {mean_act:.2f}+-{std_act:.2f}, "
                  f"Random: {mean_rand:.4f}+-{std_rand:.4f}")

    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(CUR_DIR, CROSS_CONFIG["results_file"])
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # 打印最终汇总表
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    for train_task, eval_task in cross_pairs:
        print(f"\n--- Train: {train_task.upper()} -> Eval: {eval_task.upper()} ---")
        print(f"{'Threshold':<12} {'DeGater Acc':<22} {'Avg Activated':<20} {'Random Baseline':<22}")
        print("-" * 76)

        subset = [r for r in all_results
                  if r['train_task'] == train_task.upper()
                  and r['eval_task'] == eval_task.upper()]
        for r in subset:
            print(f"{r['threshold']:<12.1f} "
                  f"{r['degater_accuracy']:<22} "
                  f"{r['avg_activated_models']:<20} "
                  f"{r['random_baseline_accuracy']:<22}")

    return all_results


# =============================================
# 主函数
# =============================================
def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Generalization Test")
    parser.add_argument("--mode", default="all", choices=["prepare", "train", "eval", "all"],
                        help="prepare=数据准备, train=训练门控, eval=跨领域评估, all=全部")
    parser.add_argument("--force_train", action="store_true",
                        help="强制重新训练所有门控模型")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CROSS-DOMAIN GENERALIZATION TEST")
    print("=" * 80)
    print(f"Seeds: {CROSS_CONFIG['seeds']}")
    print(f"Thresholds: {CROSS_CONFIG['thresholds']}")
    print(f"Embedding: {CROSS_CONFIG['embedding_key']}")
    print(f"Gate type: {CROSS_CONFIG['gate_type']}")
    print(f"Gate dir: {CROSS_CONFIG['gate_dir']}")
    print(f"Results file: {CROSS_CONFIG['results_file']}")
    print("=" * 80 + "\n")

    if args.force_train:
        import shutil
        gate_dir = os.path.join(CUR_DIR, CROSS_CONFIG["gate_dir"])
        if os.path.exists(gate_dir):
            shutil.rmtree(gate_dir)
            print(f"Removed existing gate dir: {gate_dir}")

    if args.mode in ["prepare", "all"]:
        prepare_all_data()

    if args.mode in ["train", "all"]:
        train_all_gates()

    if args.mode in ["eval", "all"]:
        run_full_experiment()

    print("\n" + "=" * 80)
    print("Cross-Domain Test Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    gc.enable()
    main()
