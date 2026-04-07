import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import argparse
import random
from collections import Counter
import gc

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from config import *
from data_split import split_mmlu_data, prepare_gsm8k_data
from embedding_manager import EmbeddingManager
from gate_model import GateNetwork
from data_loader import (
    normalize_question, 
    compute_gsm8k_accuracy,
    load_mmlu_single_model,
    load_gsm8k_single_model
)

# 设置实验配置
EXPERIMENT_CONFIG = {
    "encoders": ["bert-base-uncased", "sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-large-v2"],
    "seeds": [42, 123, 0],
    "top_k": 4,  # top-k聚合策略
    "gate_type": "mlp",  # 使用MLP门控网络
    "results_dir": "experiment_results",
    "gate_encoder_dir": "gate_encoder",  # 专门存放门控encoder模型的目录
    "train_batch_size": 32,  # 训练batch size
    "cache_dir": "embedding_cache_experiment"  # embedding缓存目录
}

# 更新embedding模型配置
EMBEDDING_MODELS.update({
    "bert": "bert-base-uncased",
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-large": "intfloat/e5-large-v2"
})

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
    
    # 检查是否已存在划分好的数据
    split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
    
    # MMLU数据
    mmlu_train_path = os.path.join(split_dir, "mmlu_train.pkl")
    mmlu_test_path = os.path.join(split_dir, "mmlu_test.pkl")
    
    if not os.path.exists(mmlu_train_path) or not os.path.exists(mmlu_test_path):
        print("Preparing MMLU data...")
        split_mmlu_data()
    else:
        print("MMLU data already prepared.")
    
    # GSM8K数据
    gsm8k_train_path = os.path.join(split_dir, "gsm8k_train.pkl")
    gsm8k_test_path = os.path.join(split_dir, "gsm8k_test.pkl")
    
    if not os.path.exists(gsm8k_train_path) or not os.path.exists(gsm8k_test_path):
        print("Preparing GSM8K data...")
        prepare_gsm8k_data()
    else:
        print("GSM8K data already prepared.")
    
    print("✓ Data preparation complete!")

def check_existing_gate_models(task_type, encoder_name, seed):
    """
    检查已存在的门控模型，返回需要训练的模型列表
    
    Args:
        task_type: "mmlu" or "gsm8k"
        encoder_name: 编码器名称
        seed: 随机种子
    
    Returns:
        existing_models: 已存在的模型列表
        missing_models: 缺失的模型列表
    """
    gate_encoder_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["gate_encoder_dir"])
    
    if not os.path.exists(gate_encoder_dir):
        os.makedirs(gate_encoder_dir, exist_ok=True)
        return [], []
    
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
    else:
        model_list = GSM8K_TRAIN_MODELS
    
    encoder_key = encoder_name.replace("/", "_").replace("-", "_")
    existing_models = []
    missing_models = []
    
    for model_name in model_list:
        model_path = os.path.join(
            gate_encoder_dir,
            f"{task_type}_{model_name}_{encoder_key}_seed{seed}.pt"
        )
        
        if os.path.exists(model_path):
            # 检查文件大小是否合理（至少50KB）
            if os.path.getsize(model_path) > 50 * 1024:
                # 尝试加载模型验证完整性
                try:
                    # 创建临时的encoder管理器以获取维度
                    temp_encoder = ExperimentEmbeddingManager(encoder_name, device='cpu')
                    embedding_dim = temp_encoder.encoder_dim
                    
                    # 创建模型结构
                    gate_model = GateNetwork(
                        input_dim=embedding_dim,
                        gate_type=EXPERIMENT_CONFIG["gate_type"],
                        hidden_dim=TRAIN_CONFIG["hidden_dim"],
                        dropout=TRAIN_CONFIG["dropout"]
                    ).to('cpu')
                    
                    # 加载权重
                    checkpoint = torch.load(model_path, map_location='cpu')
                    gate_model.load_state_dict(checkpoint['model_state_dict'])
                    gate_model.eval()
                    
                    # 验证模型能正常前向传播
                    with torch.no_grad():
                        test_input = torch.randn(1, embedding_dim)
                        test_output = gate_model(test_input)
                        if test_output is not None and not torch.isnan(test_output).any():
                            existing_models.append(model_name)
                        else:
                            print(f"  Warning: Model {model_name} produces NaN output, will retrain")
                            missing_models.append(model_name)
                    
                    # 清理内存
                    del gate_model, temp_encoder
                    gc.collect()
                    
                except Exception as e:
                    print(f"  Warning: Model {model_name} may be corrupted ({e}), will retrain")
                    missing_models.append(model_name)
            else:
                print(f"  Warning: Model file {model_name} is too small, will retrain")
                missing_models.append(model_name)
        else:
            missing_models.append(model_name)
    
    return existing_models, missing_models

class ExperimentEmbeddingManager:
    """实验专用的Embedding管理器"""
    
    def __init__(self, encoder_name, device='cuda'):
        self.encoder_name = encoder_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Initializing embedding encoder: {encoder_name}")
        
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name).to(self.device)
        
        # 冻结encoder参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.eval()
        self.encoder_dim = self.encoder.config.hidden_size
        
        # 对于sentence-transformers模型，特殊处理
        if "sentence-transformers" in encoder_name:
            self.use_sentence_transformer = True
        else:
            self.use_sentence_transformer = False
    
    def encode_batch(self, questions, batch_size=32):
        """批量编码问题"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i+batch_size]
                
                encoding = self.tokenizer(
                    batch_questions,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                model_output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 使用mean pooling
                token_embeddings = model_output.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                    input_mask_expanded.sum(1), min=1e-9
                )
                
                all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def precompute_embeddings(self, task_type, force_recompute=False):
        """预计算并缓存embedding"""
        cache_dir = os.path.join(CUR_DIR, DATA_DIR, EXPERIMENT_CONFIG["cache_dir"])
        os.makedirs(cache_dir, exist_ok=True)
        
        # 创建唯一的缓存文件名
        encoder_key = self.encoder_name.replace("/", "_").replace("-", "_")
        cache_file = os.path.join(
            cache_dir, 
            f"{task_type}_{encoder_key}_embeddings.pkl"
        )
        
        # 检查缓存
        if not force_recompute and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pkl.load(f)
            return cached_data['train_embeddings'], cached_data['test_embeddings']
        
        # 加载数据
        print(f"Computing embeddings for {task_type} using {self.encoder_name}...")
        split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
        
        with open(os.path.join(split_dir, f"{task_type}_train.pkl"), "rb") as f:
            train_data = pkl.load(f)
        with open(os.path.join(split_dir, f"{task_type}_test.pkl"), "rb") as f:
            test_data = pkl.load(f)
        
        # 计算embedding
        train_embeddings = self.encode_batch(train_data['questions'])
        test_embeddings = self.encode_batch(test_data['questions'])
        
        # 缓存到磁盘
        print(f"Saving embeddings to {cache_file}")
        with open(cache_file, 'wb') as f:
            pkl.dump({
                'train_embeddings': train_embeddings,
                'test_embeddings': test_embeddings,
                'encoder_name': self.encoder_name,
                'encoder_dim': self.encoder_dim
            }, f)
        
        return train_embeddings, test_embeddings

def train_gate_for_model(task_type, model_name, encoder_manager, seed):
    """为单个模型训练门控网络"""
    print(f"Training gate for {model_name} with seed {seed}...")
    
    set_seed(seed)
    
    # 1. 准备数据
    split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
    with open(os.path.join(split_dir, f"{task_type}_train.pkl"), "rb") as f:
        train_data = pkl.load(f)
    
    train_embeddings, _ = encoder_manager.precompute_embeddings(task_type)
    embedding_dim = encoder_manager.encoder_dim
    
    # 2. 创建数据集
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
        model_idx = model_list.index(model_name)
        
        # 创建数据字典格式
        data_dict = train_data['data']
        
        # 准备数据
        embeddings_tensor = train_embeddings
        labels = train_data['labels']
        
        # 获取所有模型的预测
        all_predictions = []
        for m in model_list:
            preds = data_dict[m]
            all_predictions.append(preds)
        
        all_predictions = np.stack(all_predictions, axis=1)  # (N, M, 4)
        current_preds = all_predictions[:, model_idx, :]  # 当前模型的预测
        
        # 计算目标值：当前模型相对于其他模型的优势
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        label_indices = np.array([label_map[l] for l in labels])
        
        # 计算每个模型在正确答案上的概率
        all_probs = np.exp(all_predictions)
        all_probs = all_probs / np.sum(all_probs, axis=2, keepdims=True)
        
        # 提取正确答案概率
        batch_indices = np.arange(len(labels))
        model_indices = np.arange(len(model_list))
        batch_grid, model_grid = np.meshgrid(batch_indices, model_indices, indexing='ij')
        correct_probs = all_probs[batch_grid, model_grid, label_indices[:, None]]
        
        # 计算当前模型的相对优势
        current_correct = correct_probs[:, model_idx]
        other_mean = np.mean(np.delete(correct_probs, model_idx, axis=1), axis=1)
        relative_advantage = current_correct - other_mean
        
        # 转换为目标值 (0-1范围，0.5表示中性)
        targets = 0.5 + 0.5 * np.tanh(relative_advantage)  # 优势越大，值越接近1
        
        # 创建数据集
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(
            embeddings_tensor,
            torch.FloatTensor(targets).unsqueeze(1)
        )
        
    else:  # gsm8k
        model_list = GSM8K_TRAIN_MODELS
        model_idx = model_list.index(model_name)
        
        raw_predictions = train_data['raw_predictions']
        labels = train_data['labels']
        
        # 计算每个模型的准确率
        batch_size = raw_predictions.shape[0]
        num_models = raw_predictions.shape[1]
        num_runs = raw_predictions.shape[2]
        
        accuracies = np.zeros((batch_size, num_models))
        
        for i in range(batch_size):
            label = labels[i]
            if np.isnan(label):
                continue
                
            for j in range(num_models):
                correct_count = 0
                for k in range(num_runs):
                    pred_val = raw_predictions[i, j, k]
                    try:
                        if abs(float(pred_val) - float(label)) < 1e-4:
                            correct_count += 1
                    except:
                        pass
                accuracies[i, j] = correct_count / num_runs
        
        # 计算相对优势
        current_acc = accuracies[:, model_idx]
        other_mean = np.mean(np.delete(accuracies, model_idx, axis=1), axis=1)
        relative_advantage = current_acc - other_mean
        
        # 转换为目标值
        targets = 0.5 + 0.5 * np.tanh(relative_advantage)
        
        # 创建数据集
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(
            train_embeddings,
            torch.FloatTensor(targets).unsqueeze(1)
        )
    
    # 3. 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=EXPERIMENT_CONFIG["train_batch_size"], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=EXPERIMENT_CONFIG["train_batch_size"], 
        shuffle=False
    )
    
    # 4. 初始化门控网络
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gate_model = GateNetwork(
        input_dim=embedding_dim,
        gate_type=EXPERIMENT_CONFIG["gate_type"],
        hidden_dim=TRAIN_CONFIG["hidden_dim"],
        dropout=TRAIN_CONFIG["dropout"]
    ).to(device)
    
    optimizer = torch.optim.Adam(
        gate_model.parameters(), 
        lr=TRAIN_CONFIG["learning_rate"]
    )
    criterion = torch.nn.MSELoss()
    
    # 5. 训练
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        # 训练
        gate_model.train()
        train_loss = 0
        for emb_batch, target_batch in train_loader:
            emb_batch = emb_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            outputs = gate_model(emb_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        gate_model.eval()
        val_loss = 0
        with torch.no_grad():
            for emb_batch, target_batch in val_loader:
                emb_batch = emb_batch.to(device)
                target_batch = target_batch.to(device)
                
                outputs = gate_model(emb_batch)
                loss = criterion(outputs, target_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 打印训练进度（每5个epoch）
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型
            gate_encoder_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["gate_encoder_dir"])
            os.makedirs(gate_encoder_dir, exist_ok=True)
            
            encoder_key = encoder_manager.encoder_name.replace("/", "_").replace("-", "_")
            model_path = os.path.join(
                gate_encoder_dir,
                f"{task_type}_{model_name}_{encoder_key}_seed{seed}.pt"
            )
            
            torch.save({
                'model_state_dict': gate_model.state_dict(),
                'encoder_name': encoder_manager.encoder_name,
                'seed': seed,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'embedding_dim': embedding_dim,
                'training_complete': True
            }, model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch+1}, best epoch: {best_epoch+1}")
            break
    
    print(f"✓ Gate for {model_name} trained (seed={seed}, val_loss={best_val_loss:.4f}, best_epoch={best_epoch+1})")
    
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
    
    tasks = ["mmlu", "gsm8k"]
    
    # 训练统计
    total_existing = 0
    total_trained = 0
    total_to_train = 0
    
    for task in tasks:
        print(f"\n{'='*40}")
        print(f"Task: {task.upper()}")
        print('='*40)
        
        if task == "mmlu":
            model_list = MMLU_TRAIN_MODELS
        else:
            model_list = GSM8K_TRAIN_MODELS
        
        total_models = len(model_list)
        
        for encoder_name in EXPERIMENT_CONFIG["encoders"]:
            print(f"\nEncoder: {encoder_name}")
            print("-" * 30)
            
            # 初始化encoder管理器
            encoder_manager = ExperimentEmbeddingManager(encoder_name)
            
            # 预计算embedding（所有模型共享）
            encoder_manager.precompute_embeddings(task, force_recompute=False)
            
            for seed in EXPERIMENT_CONFIG["seeds"]:
                print(f"\nSeed: {seed}")
                
                # 检查已存在的模型
                existing_models, missing_models = check_existing_gate_models(task, encoder_name, seed)
                
                existing_count = len(existing_models)
                missing_count = len(missing_models)
                
                total_existing += existing_count
                total_to_train += missing_count
                
                print(f"  ✓ Existing models: {existing_count}/{total_models}")
                print(f"  ✗ Missing models: {missing_count}/{total_models}")
                
                if missing_count == 0:
                    print("  All models already trained, skipping...")
                    continue
                
                # 显示缺失的模型
                if missing_count <= 10:
                    print(f"  Missing models: {missing_models}")
                else:
                    print(f"  Missing models (first 10): {missing_models[:10]}...")
                
                # 为每个缺失的模型训练门控网络
                trained_in_batch = 0
                for i, model_name in enumerate(missing_models):
                    try:
                        print(f"  [{i+1}/{missing_count}] ", end="")
                        train_gate_for_model(task, model_name, encoder_manager, seed)
                        trained_in_batch += 1
                        total_trained += 1
                    except Exception as e:
                        print(f"Error training {model_name}: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"  Trained {trained_in_batch}/{missing_count} models for this configuration")
    
    # 打印总结
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total existing models: {total_existing}")
    print(f"Total models to train: {total_to_train}")
    print(f"Total models trained in this run: {total_trained}")
    
    if total_to_train == 0:
        print("\n✓ All models are already trained and validated! Proceeding to evaluation...")
    else:
        print(f"\n✓ Training completed. Total trained: {total_trained} models")
    
    print("\n✓ Gate network training phase complete!")

def load_gate_model(task_type, model_name, encoder_name, seed):
    """加载训练好的门控模型"""
    gate_encoder_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["gate_encoder_dir"])
    
    encoder_key = encoder_name.replace("/", "_").replace("-", "_")
    model_path = os.path.join(
        gate_encoder_dir,
        f"{task_type}_{model_name}_{encoder_key}_seed{seed}.pt"
    )
    
    if not os.path.exists(model_path):
        return None
    
    # 检查文件大小
    if os.path.getsize(model_path) < 50 * 1024:
        print(f"Warning: Model file {model_name} is too small, may be corrupted")
        return None
    
    # 初始化门控网络
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # 先加载checkpoint获取embedding维度
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'embedding_dim' not in checkpoint:
            # 如果没有保存维度，使用encoder管理器获取
            encoder_manager = ExperimentEmbeddingManager(encoder_name, device='cpu')
            embedding_dim = encoder_manager.encoder_dim
        else:
            embedding_dim = checkpoint['embedding_dim']
        
        # 创建模型
        gate_model = GateNetwork(
            input_dim=embedding_dim,
            gate_type=EXPERIMENT_CONFIG["gate_type"],
            hidden_dim=TRAIN_CONFIG["hidden_dim"],
            dropout=TRAIN_CONFIG["dropout"]
        ).to(device)
        
        # 加载权重
        gate_model.load_state_dict(checkpoint['model_state_dict'])
        gate_model.eval()
        
        # 验证模型完整性
        with torch.no_grad():
            test_input = torch.randn(1, embedding_dim).to(device)
            test_output = gate_model(test_input)
            if test_output is None or torch.isnan(test_output).any():
                print(f"Warning: Model {model_name} produces NaN output")
                return None
        
        return gate_model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def check_evaluation_models(task_type, encoder_name, seed):
    """检查评估所需的所有模型是否都存在"""
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
    else:
        model_list = GSM8K_TRAIN_MODELS
    
    missing_models = []
    
    for model_name in model_list:
        gate_model = load_gate_model(task_type, model_name, encoder_name, seed)
        if gate_model is None:
            missing_models.append(model_name)
        else:
            # 清理内存
            del gate_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return missing_models

def evaluate_topk_ensemble(task_type, encoder_name, seed):
    """使用top-k策略评估集成模型"""
    print(f"Evaluating {task_type.upper()} with {encoder_name} (seed={seed})...")
    
    # 1. 准备数据
    split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
    with open(os.path.join(split_dir, f"{task_type}_test.pkl"), "rb") as f:
        test_data = pkl.load(f)
    
    # 2. 初始化encoder并计算embedding
    encoder_manager = ExperimentEmbeddingManager(encoder_name)
    _, test_embeddings = encoder_manager.precompute_embeddings(task_type)
    
    # 3. 检查所需模型是否都存在
    missing_models = check_evaluation_models(task_type, encoder_name, seed)
    if missing_models:
        print(f"Error: Missing models for evaluation: {missing_models}")
        print("Please train the missing models first.")
        return 0.0
    
    # 4. 加载所有门控模型
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
    else:
        model_list = GSM8K_TRAIN_MODELS
    
    gate_models = {}
    for model_name in model_list:
        gate_model = load_gate_model(task_type, model_name, encoder_name, seed)
        if gate_model is not None:
            gate_models[model_name] = gate_model
        else:
            print(f"Error: Failed to load model {model_name}")
            return 0.0
    
    if len(gate_models) != len(model_list):
        print(f"Warning: Only loaded {len(gate_models)}/{len(model_list)} gate models")
        return 0.0
    
    # 5. 评估
    correct = 0
    total = len(test_data['questions'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i in tqdm(range(total), desc=f"Evaluating {encoder_name}"):
        # 获取问题的embedding
        embedding = test_embeddings[i].unsqueeze(0)
        
        # 计算每个模型的gate分数
        scores = {}
        for model_name, gate_model in gate_models.items():
            with torch.no_grad():
                embedding_device = embedding.to(device)
                score = gate_model(embedding_device).item()
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
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f}")
    
    # 清理内存
    for model in gate_models.values():
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return accuracy

def run_evaluation():
    """运行所有评估"""
    print("\n" + "=" * 80)
    print("STEP 3: Evaluation with Top-K Strategy")
    print("=" * 80)
    
    tasks = ["mmlu", "gsm8k"]
    results = {}
    
    for task in tasks:
        print(f"\n{'='*40}")
        print(f"Task: {task.upper()}")
        print('='*40)
        
        task_results = {}
        
        for encoder_name in EXPERIMENT_CONFIG["encoders"]:
            print(f"\nEncoder: {encoder_name}")
            
            accuracies = []
            for seed in EXPERIMENT_CONFIG["seeds"]:
                accuracy = evaluate_topk_ensemble(task, encoder_name, seed)
                accuracies.append(accuracy)
            
            # 计算平均数和标准差
            if len(accuracies) > 0:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies, ddof=1)  # 样本标准差
            else:
                mean_acc = 0.0
                std_acc = 0.0
            
            task_results[encoder_name] = {
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
        for encoder_name in EXPERIMENT_CONFIG["encoders"]:
            if encoder_name in results[task]:
                mean_val = results[task][encoder_name]['mean']
                std_val = results[task][encoder_name]['std']
                
                csv_data.append({
                    'Task': task.upper(),
                    'Encoder': encoder_name,
                    'Mean': f"{mean_val:.4f}",
                    'Std': f"{std_val:.4f}",
                    'Result': f"{mean_val:.4f} ± {std_val:.4f}"
                })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(results_dir, "encoder_results.csv")
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
        
        for encoder_name in EXPERIMENT_CONFIG["encoders"]:
            if encoder_name in results[task]:
                mean_val = results[task][encoder_name]['mean']
                std_val = results[task][encoder_name]['std']
                accuracies = results[task][encoder_name]['accuracies']
                
                print(f"{encoder_name}:")
                print(f"  Accuracies: {[f'{a:.4f}' for a in accuracies]}")
                print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
                print()
    
    return results

def check_models_status():
    """检查所有模型的训练状态"""
    print("\n" + "=" * 80)
    print("MODEL STATUS CHECK")
    print("=" * 80)
    
    tasks = ["mmlu", "gsm8k"]
    
    status_summary = {}
    
    for task in tasks:
        print(f"\n{task.upper()}:")
        print("-" * 40)
        
        if task == "mmlu":
            model_list = MMLU_TRAIN_MODELS
        else:
            model_list = GSM8K_TRAIN_MODELS
        
        task_status = {}
        
        for encoder_name in EXPERIMENT_CONFIG["encoders"]:
            print(f"\n  Encoder: {encoder_name}")
            
            encoder_status = {}
            
            for seed in EXPERIMENT_CONFIG["seeds"]:
                existing_models, missing_models = check_existing_gate_models(task, encoder_name, seed)
                encoder_status[seed] = {
                    'existing': len(existing_models),
                    'missing': len(missing_models),
                    'total': len(model_list)
                }
                
                print(f"    Seed {seed}: {len(existing_models)}/{len(model_list)} models ready")
                if missing_models and len(missing_models) <= 5:
                    print(f"      Missing: {missing_models}")
            
            task_status[encoder_name] = encoder_status
        
        status_summary[task] = task_status
    
    return status_summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行完整的实验")
    parser.add_argument("--mode", default="all", choices=["prepare", "train", "eval", "status", "all"],
                       help="运行模式: prepare(准备数据), train(训练), eval(评估), status(检查状态), all(全部)")
    parser.add_argument("--force_train", action="store_true",
                       help="强制重新训练所有模型，即使已存在")
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("SMALL LANGUAGE MODEL ENSEMBLE EXPERIMENT")
    print("=" * 80)
    print(f"Encoders: {EXPERIMENT_CONFIG['encoders']}")
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
        gate_encoder_dir = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["gate_encoder_dir"])
        if os.path.exists(gate_encoder_dir):
            import shutil
            shutil.rmtree(gate_encoder_dir)
            print(f"Removed existing model directory: {gate_encoder_dir}")
    
    if args.mode == "status":
        check_models_status()
        return
    
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
        
        csv_path = os.path.join(CUR_DIR, EXPERIMENT_CONFIG["results_dir"], "encoder_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print("\nEncoder Results:")
            print(df.to_string(index=False))
        else:
            print("Results file not found. Please run evaluation first.")
    
    print("\n" + "=" * 80)
    print("Experiment Completed!")
    print("=" * 80)

if __name__ == "__main__":
    # 启用垃圾回收
    gc.enable()
    main()