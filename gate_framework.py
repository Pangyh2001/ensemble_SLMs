# experiment_script_fixed.py

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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from config import *
from data_split import split_mmlu_data, prepare_gsm8k_data
from embedding_manager import get_embedding_manager
from gate_model import GateNetwork
from data_loader import compute_gsm8k_accuracy
from trainer import (
    MMLUEmbeddingDataset, GSM8KEmbeddingDataset,
    collate_fn_mmlu, collate_fn_gsm8k,
    compute_deviation_mmlu_relative, compute_deviation_gsm8k_relative
)

# 设置实验配置
EXPERIMENT_CONFIG = {
    "gate_models": ["mlp", "attention", "resnet"],  # 三种门控网络结构
    "seeds": [42, 123, 0],
    "top_k": 4,  # top-k聚合策略
    "embedding_key": "bert",  # 使用bert-base编码器（固定）
    "results_dir": "experiment_results",
    "gate_model_dir": "gate_model",  # 专门存放门控模型的目录
    "split_dir": "splits_gate",  # 划分数据存放目录
    "train_batch_size": 32,  # 使用较小的batch size
    "eval_batch_size": 64,
}

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data():
    """准备数据（只划分一次）"""
    print("=" * 80)
    print("STEP 1: Preparing Data")
    print("=" * 80)
    
    # 备份原始DATA_DIR
    global DATA_DIR, CUR_DIR
    original_data_dir = DATA_DIR
    
    # 临时修改DATA_DIR以保存到splits_gate文件夹
    DATA_DIR = os.path.join("dataset", EXPERIMENT_CONFIG["split_dir"])
    
    # 创建目录
    os.makedirs(os.path.join(CUR_DIR, DATA_DIR), exist_ok=True)
    
    # 检查是否已存在划分好的数据
    split_dir = os.path.join(CUR_DIR, DATA_DIR)
    
    # MMLU数据
    if not os.path.exists(os.path.join(split_dir, "mmlu_train.pkl")):
        print("Preparing MMLU data...")
        split_mmlu_data()
    else:
        print("MMLU data already prepared.")
    
    # GSM8K数据
    if not os.path.exists(os.path.join(split_dir, "gsm8k_train.pkl")):
        print("Preparing GSM8K data...")
        prepare_gsm8k_data()
    else:
        print("GSM8K data already prepared.")
    
    # 恢复原始DATA_DIR
    DATA_DIR = original_data_dir
    
    print("✓ Data preparation complete!")

def check_existing_models(task_type, gate_type, seed):
    """
    检查已存在的模型，返回需要训练的模型列表
    """
    gate_model_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["gate_model_dir"])
    
    if not os.path.exists(gate_model_dir):
        os.makedirs(gate_model_dir, exist_ok=True)
        return [], []
    
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
    else:
        model_list = GSM8K_TRAIN_MODELS
    
    existing_models = []
    missing_models = []
    
    for model_name in model_list:
        model_path = os.path.join(
            gate_model_dir,
            f"{task_type}_{model_name}_{gate_type}_seed{seed}.pt"
        )
        
        if os.path.exists(model_path):
            # 检查文件大小是否合理（至少100KB）
            if os.path.getsize(model_path) > 100 * 1024:
                existing_models.append(model_name)
            else:
                print(f"  Warning: Model file {model_path} is too small, will retrain")
                missing_models.append(model_name)
        else:
            missing_models.append(model_name)
    
    return existing_models, missing_models

def train_gate_for_model(task_type, model_name, embedding_manager, gate_type, seed):
    """为单个模型训练门控网络"""
    print(f"Training gate for {model_name} with {gate_type} gate (seed={seed})...")
    
    set_seed(seed)
    
    # 1. 准备数据
    split_dir = os.path.join(CUR_DIR, "dataset", EXPERIMENT_CONFIG["split_dir"])
    with open(os.path.join(split_dir, f"{task_type}_train.pkl"), "rb") as f:
        train_data = pkl.load(f)
    
    # 预计算embedding（使用项目原有的embedding manager）
    train_embeddings, _ = embedding_manager.precompute_embeddings(task_type)
    embedding_dim = embedding_manager.get_encoder_dim()
    
    # 2. 创建数据集（使用项目原有的Dataset类）
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
        
        dataset = MMLUEmbeddingDataset(
            train_embeddings,
            train_data['labels'], 
            model_name,
            train_data['data'],
            model_list
        )
        collate_fn = collate_fn_mmlu
        
    else:  # gsm8k
        model_list = GSM8K_TRAIN_MODELS
        model_idx = model_list.index(model_name)
        
        dataset = GSM8KEmbeddingDataset(
            train_embeddings,
            train_data['raw_predictions'],
            train_data['labels'],
            model_idx
        )
        collate_fn = collate_fn_gsm8k
    
    # 3. 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=EXPERIMENT_CONFIG["train_batch_size"], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=EXPERIMENT_CONFIG["train_batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 4. 初始化门控网络
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gate_model = GateNetwork(
        input_dim=embedding_dim,
        gate_type=gate_type,
        hidden_dim=TRAIN_CONFIG["hidden_dim"],
        num_heads=TRAIN_CONFIG.get("num_heads", 8),
        num_blocks=TRAIN_CONFIG.get("num_blocks", 4),
        dropout=TRAIN_CONFIG["dropout"]
    ).to(device)
    
    optimizer = torch.optim.Adam(
        gate_model.parameters(), 
        lr=TRAIN_CONFIG["learning_rate"]
    )
    
    # 5. 训练
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        # 训练
        gate_model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels']
            all_predictions = batch['all_predictions'].to(device)
            model_idx = batch['model_idx']
            
            # 获取Gate分数
            scores = gate_model(embeddings)
            
            # 计算相对偏离程度（使用项目原有的函数）
            if task_type == "mmlu":
                deviation = compute_deviation_mmlu_relative(all_predictions, labels, model_idx)
            else:  # gsm8k
                deviation = compute_deviation_gsm8k_relative(all_predictions, labels, model_idx)
            
            # 目标：积极性 = 1 - 偏离程度
            target = (1 - deviation).unsqueeze(1)
            
            # MSE损失
            loss = torch.nn.functional.mse_loss(scores, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 每20个batch清理一次内存
            if batch_idx % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        
        # 验证
        gate_model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                labels = batch['labels']
                all_predictions = batch['all_predictions'].to(device)
                model_idx = batch['model_idx']
                
                scores = gate_model(embeddings)
                
                if task_type == "mmlu":
                    deviation = compute_deviation_mmlu_relative(all_predictions, labels, model_idx)
                else:  # gsm8k
                    deviation = compute_deviation_gsm8k_relative(all_predictions, labels, model_idx)
                
                target = (1 - deviation).unsqueeze(1)
                loss = torch.nn.functional.mse_loss(scores, target)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存模型
            gate_model_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["gate_model_dir"])
            os.makedirs(gate_model_dir, exist_ok=True)
            
            model_path = os.path.join(
                gate_model_dir,
                f"{task_type}_{model_name}_{gate_type}_seed{seed}.pt"
            )
            
            torch.save({
                'model_state_dict': gate_model.state_dict(),
                'gate_type': gate_type,
                'seed': seed,
                'best_val_loss': best_val_loss,
                'embedding_key': EXPERIMENT_CONFIG["embedding_key"]
            }, model_path)
            
        else:
            patience_counter += 1
        
        if patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"✓ Gate for {model_name} trained (gate_type={gate_type}, seed={seed}, val_loss={best_val_loss:.4f})")
    
    # 清理内存
    del gate_model, optimizer, train_loader, val_loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_val_loss

def train_all_gates():
    """训练所有门控网络，跳过已存在的模型"""
    print("\n" + "=" * 80)
    print("STEP 2: Training All Gate Networks")
    print("=" * 80)
    print(f"Embedding: {EXPERIMENT_CONFIG['embedding_key']} (shared, frozen)")
    print(f"Gate Models: {EXPERIMENT_CONFIG['gate_models']}")
    print(f"Seeds: {EXPERIMENT_CONFIG['seeds']}")
    print("=" * 80)
    
    tasks = ["mmlu", "gsm8k"]
    
    # 获取共享的embedding manager（使用项目原有的单例模式）
    embedding_manager = get_embedding_manager(
        EXPERIMENT_CONFIG["embedding_key"], 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 统计信息
    total_models_to_train = 0
    total_existing_models = 0
    
    for task in tasks:
        print(f"\n{'='*40}")
        print(f"Task: {task.upper()}")
        print('='*40)
        
        if task == "mmlu":
            model_list = MMLU_TRAIN_MODELS
        else:
            model_list = GSM8K_TRAIN_MODELS
        
        # 预计算当前任务的embedding
        print("Precomputing embeddings...")
        embedding_manager.precompute_embeddings(task, force_recompute=False)
        
        for gate_type in EXPERIMENT_CONFIG["gate_models"]:
            print(f"\nGate Type: {gate_type}")
            print("-" * 30)
            
            for seed in EXPERIMENT_CONFIG["seeds"]:
                print(f"\nSeed: {seed}")
                
                # 检查已存在的模型
                existing_models, missing_models = check_existing_models(task, gate_type, seed)
                
                print(f"  Existing models ({len(existing_models)}): {existing_models}")
                print(f"  Missing models ({len(missing_models)}): {missing_models}")
                
                total_existing_models += len(existing_models)
                total_models_to_train += len(missing_models)
                
                if len(missing_models) == 0:
                    print("  All models already trained, skipping...")
                    continue
                
                # 为每个缺失的模型训练门控网络
                for model_name in missing_models:
                    try:
                        train_gate_for_model(task, model_name, embedding_manager, gate_type, seed)
                    except Exception as e:
                        print(f"  Error training {model_name}: {e}")
                        import traceback
                        traceback.print_exc()
    
    # 打印总结
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total existing models: {total_existing_models}")
    print(f"Total models to train: {total_models_to_train}")
    
    if total_models_to_train == 0:
        print("\nAll models are already trained! Proceeding to evaluation...")
    else:
        print(f"\nTraining {total_models_to_train} models...")
    
    print("\n✓ Gate network training phase complete!")

def load_gate_model(task_type, model_name, gate_type, seed):
    """加载训练好的门控模型"""
    gate_model_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["gate_model_dir"])
    
    model_path = os.path.join(
        gate_model_dir,
        f"{task_type}_{model_name}_{gate_type}_seed{seed}.pt"
    )
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found: {model_path}")
        return None
    
    # 检查文件大小
    if os.path.getsize(model_path) < 100 * 1024:
        print(f"Warning: Model file {model_path} is too small, may be corrupted")
        return None
    
    # 获取embedding manager以获取维度
    embedding_manager = get_embedding_manager(
        EXPERIMENT_CONFIG["embedding_key"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    embedding_dim = embedding_manager.get_encoder_dim()
    
    # 初始化门控网络
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gate_model = GateNetwork(
        input_dim=embedding_dim,
        gate_type=gate_type,
        hidden_dim=TRAIN_CONFIG["hidden_dim"],
        num_heads=TRAIN_CONFIG.get("num_heads", 8),
        num_blocks=TRAIN_CONFIG.get("num_blocks", 4),
        dropout=TRAIN_CONFIG["dropout"]
    ).to(device)
    
    try:
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        gate_model.load_state_dict(checkpoint['model_state_dict'])
        gate_model.eval()
        
        # 验证模型完整性
        with torch.no_grad():
            test_input = torch.randn(1, embedding_dim).to(device)
            test_output = gate_model(test_input)
            if test_output is None or torch.isnan(test_output).any():
                print(f"Warning: Model {model_path} produces NaN output")
                return None
        
        return gate_model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def check_evaluation_models(task_type, gate_type, seed):
    """检查评估所需的所有模型是否都存在"""
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
    else:
        model_list = GSM8K_TRAIN_MODELS
    
    missing_models = []
    
    for model_name in model_list:
        gate_model = load_gate_model(task_type, model_name, gate_type, seed)
        if gate_model is None:
            missing_models.append(model_name)
        else:
            # 清理内存
            del gate_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return missing_models

def evaluate_topk_ensemble(task_type, gate_type, seed):
    """使用top-k策略评估集成模型"""
    print(f"Evaluating {task_type.upper()} with {gate_type} gate (seed={seed})...")
    
    # 1. 准备数据
    split_dir = os.path.join(CUR_DIR, "dataset", EXPERIMENT_CONFIG["split_dir"])
    test_path = os.path.join(split_dir, f"{task_type}_test.pkl")
    
    if not os.path.exists(test_path):
        print(f"Error: Test file not found: {test_path}")
        return 0.0
    
    with open(test_path, "rb") as f:
        test_data = pkl.load(f)
    
    # 2. 获取embedding manager并加载测试集embedding
    embedding_manager = get_embedding_manager(
        EXPERIMENT_CONFIG["embedding_key"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    _, test_embeddings = embedding_manager.precompute_embeddings(task_type, force_recompute=False)
    
    # 3. 验证数据一致性 - 关键修复部分
    # 获取测试数据的样本数量
    if task_type == "mmlu":
        num_questions = len(test_data['questions'])
    else:  # gsm8k
        num_questions = len(test_data['raw_predictions'])
    
    # 获取嵌入的数量
    if hasattr(test_embeddings, 'shape'):
        num_embeddings = test_embeddings.shape[0]
    else:
        num_embeddings = len(test_embeddings)
    
    print(f"  Data validation:")
    print(f"    - Questions in test_data: {num_questions}")
    print(f"    - Embeddings generated: {num_embeddings}")
    
    # 确定评估的样本总数
    if num_questions != num_embeddings:
        print(f"  WARNING: Data mismatch detected! Questions={num_questions}, Embeddings={num_embeddings}")
        print(f"  Will evaluate on {min(num_questions, num_embeddings)} samples")
        total = min(num_questions, num_embeddings)
    else:
        total = num_questions
    
    # 4. 检查所需模型是否都存在
    missing_models = check_evaluation_models(task_type, gate_type, seed)
    if missing_models:
        print(f"Error: Missing models for evaluation: {missing_models}")
        print("Please train the missing models first.")
        return 0.0
    
    # 5. 加载所有门控模型
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
    else:
        model_list = GSM8K_TRAIN_MODELS
    
    gate_models = {}
    for model_name in model_list:
        gate_model = load_gate_model(task_type, model_name, gate_type, seed)
        if gate_model is not None:
            gate_models[model_name] = gate_model
        else:
            print(f"Error: Failed to load model {model_name}")
            return 0.0
    
    if len(gate_models) != len(model_list):
        print(f"Warning: Only loaded {len(gate_models)}/{len(model_list)} gate models")
        return 0.0
    
    # 6. 评估
    correct = 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i in tqdm(range(total), desc=f"Evaluating {gate_type}"):
        try:
            # 获取问题的embedding - 安全访问
            embedding = test_embeddings[i].unsqueeze(0).to(device)
            
            # 计算每个模型的gate分数
            scores = {}
            for model_name, gate_model in gate_models.items():
                with torch.no_grad():
                    score = gate_model(embedding).item()
                    scores[model_name] = score
            
            # 选择top-k模型
            topk = EXPERIMENT_CONFIG["top_k"]
            selected_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
            selected_models = [m[0] for m in selected_models]
            
            # 集成预测
            if task_type == "mmlu":
                # MMLU: 多数投票
                votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                
                for model_name in selected_models:
                    model_idx = model_list.index(model_name)
                    pred_probs = np.exp(test_data['data'][model_name][i])
                    pred_class = ['A', 'B', 'C', 'D'][np.argmax(pred_probs)]
                    votes[pred_class] += 1
                
                pred_label = max(votes, key=votes.get)
                true_label = test_data['labels'][i]
                
                if pred_label == true_label:
                    correct += 1
                    
            else:  # gsm8k
                # GSM8K: 多数投票（基于10次采样中最常见的答案）
                all_answers = []
                
                for model_name in selected_models:
                    model_idx = model_list.index(model_name)
                    pred_runs = test_data['raw_predictions'][i, model_idx, :]
                    
                    # 找到10次采样中最常见的答案
                    valid_answers = [float(a) for a in pred_runs if not np.isnan(a)]
                    if valid_answers:
                        # 四舍五入到4位小数以避免浮点误差
                        rounded_answers = [round(a, 4) for a in valid_answers]
                        most_common = Counter(rounded_answers).most_common(1)
                        if most_common:
                            all_answers.append(most_common[0][0])
                
                if all_answers:
                    # 在选出的模型中再进行多数投票
                    rounded_all = [round(a, 4) for a in all_answers]
                    pred_answer = Counter(rounded_all).most_common(1)[0][0]
                    true_label = test_data['labels'][i]
                    
                    try:
                        if abs(float(pred_answer) - float(true_label)) < 1e-4:
                            correct += 1
                    except:
                        pass
        
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue  # 跳过有问题的样本
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy on {total} samples: {accuracy:.4f}")
    
    # 清理内存
    for model in gate_models.values():
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return accuracy

def validate_data_consistency():
    """验证所有任务的数据一致性"""
    print("\n" + "=" * 80)
    print("Validating Data Consistency")
    print("=" * 80)
    
    tasks = ["mmlu", "gsm8k"]
    all_valid = True
    
    for task in tasks:
        print(f"\nValidating {task.upper()}...")
        
        # 检查测试数据文件
        split_dir = os.path.join(CUR_DIR, "dataset", EXPERIMENT_CONFIG["split_dir"])
        test_path = os.path.join(split_dir, f"{task}_test.pkl")
        
        if not os.path.exists(test_path):
            print(f"  ERROR: Test file not found: {test_path}")
            all_valid = False
            continue
        
        # 加载测试数据
        with open(test_path, "rb") as f:
            test_data = pkl.load(f)
        
        # 获取嵌入
        embedding_manager = get_embedding_manager(EXPERIMENT_CONFIG["embedding_key"])
        _, embeddings = embedding_manager.precompute_embeddings(task, force_recompute=False)
        
        # 检查数量
        if task == "mmlu":
            num_questions = len(test_data['questions'])
        else:
            num_questions = len(test_data['raw_predictions'])
        
        if hasattr(embeddings, 'shape'):
            num_embeddings = embeddings.shape[0]
        else:
            num_embeddings = len(embeddings)
        
        print(f"  Questions in data: {num_questions}")
        print(f"  Embeddings generated: {num_embeddings}")
        
        if num_questions == num_embeddings:
            print(f"  ✓ Data consistency: PASSED")
        else:
            print(f"  ✗ Data consistency: FAILED (difference: {abs(num_questions - num_embeddings)})")
            all_valid = False
    
    return all_valid

def run_evaluation():
    """运行所有评估"""
    print("\n" + "=" * 80)
    print("STEP 3: Evaluation with Top-K Strategy")
    print("=" * 80)
    print(f"Top-K: {EXPERIMENT_CONFIG['top_k']}")
    print("=" * 80)
    
    # 首先验证数据一致性
    if not validate_data_consistency():
        print("\nWARNING: Data consistency issues detected!")
        print("Evaluation will continue but results may be incomplete.")
    
    tasks = ["mmlu", "gsm8k"]
    results = {}
    
    for task in tasks:
        print(f"\n{'='*40}")
        print(f"Task: {task.upper()}")
        print('='*40)
        
        task_results = {}
        
        for gate_type in EXPERIMENT_CONFIG["gate_models"]:
            print(f"\nGate Type: {gate_type}")
            
            accuracies = []
            for seed in EXPERIMENT_CONFIG["seeds"]:
                accuracy = evaluate_topk_ensemble(task, gate_type, seed)
                accuracies.append(accuracy)
            
            # 计算平均数和标准差
            if len(accuracies) > 0:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies, ddof=1)  # 样本标准差
            else:
                mean_acc = 0.0
                std_acc = 0.0
            
            task_results[gate_type] = {
                'mean': mean_acc,
                'std': std_acc,
                'accuracies': accuracies
            }
            
            print(f"  Mean: {mean_acc:.4f}, Std: {std_acc:.4f}")
        
        results[task] = task_results
    
    # 保存结果
    results_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存为CSV
    csv_data = []
    
    for task in tasks:
        for gate_type in EXPERIMENT_CONFIG["gate_models"]:
            if gate_type in results[task]:
                mean_val = results[task][gate_type]['mean']
                std_val = results[task][gate_type]['std']
                
                csv_data.append({
                    'Task': task.upper(),
                    'Gate_Model': gate_type,
                    'Mean': f"{mean_val:.4f}",
                    'Std': f"{std_val:.4f}",
                    'Result': f"{mean_val:.4f} ± {std_val:.4f}"
                })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(results_dir, "model_results.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Results saved to: {csv_path}")
    else:
        print("\n✗ No results to save!")
    
    # 打印结果
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{task.upper()}:")
        print("-" * 40)
        
        for gate_type in EXPERIMENT_CONFIG["gate_models"]:
            if gate_type in results[task]:
                mean_val = results[task][gate_type]['mean']
                std_val = results[task][gate_type]['std']
                accuracies = results[task][gate_type]['accuracies']
                
                print(f"{gate_type}:")
                print(f"  Accuracies: {[f'{a:.4f}' for a in accuracies]}")
                print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
                print()
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行完整的实验")
    parser.add_argument("--mode", default="all", choices=["prepare", "train", "eval", "all"],
                       help="运行模式: prepare(准备数据), train(训练), eval(评估), all(全部)")
    parser.add_argument("--force_train", action="store_true",
                       help="强制重新训练所有模型，即使已存在")
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("SMALL LANGUAGE MODEL ENSEMBLE EXPERIMENT")
    print("=" * 80)
    print(f"Embedding: {EXPERIMENT_CONFIG['embedding_key']}")
    print(f"Gate Models: {EXPERIMENT_CONFIG['gate_models']}")
    print(f"Seeds: {EXPERIMENT_CONFIG['seeds']}")
    print(f"Top-K: {EXPERIMENT_CONFIG['top_k']}")
    if args.force_train:
        print("Mode: FORCE TRAIN (所有模型将重新训练)")
    else:
        print("Mode: SMART TRAIN (跳过已训练的模型)")
    print("=" * 80 + "\n")
    
    # 如果是强制训练模式，删除已存在的模型
    if args.force_train:
        print("Force training mode: Removing existing models...")
        gate_model_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["gate_model_dir"])
        if os.path.exists(gate_model_dir):
            import shutil
            shutil.rmtree(gate_model_dir)
            print(f"Removed existing model directory: {gate_model_dir}")
    
    if args.mode in ["prepare", "all"]:
        prepare_data()
    
    if args.mode in ["train", "all"]:
        train_all_gates()
    
    if args.mode in ["eval", "all"]:
        results = run_evaluation()
        
        # 打印最终的CSV格式结果
        print("\n" + "=" * 80)
        print("CSV FORMAT RESULTS")
        print("=" * 80)
        
        csv_path = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["results_dir"], "model_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("\nModel Results:")
            print(df.to_string(index=False))
        else:
            print("Results file not found. Please run evaluation first.")
    
    print("\n" + "=" * 80)
    print("Experiment Completed!")
    print("=" * 80)

if __name__ == "__main__":
    # 启用垃圾回收和内存清理
    gc.enable()
    
    # 如果遇到CUDA内存问题，可以启用以下选项
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    main()