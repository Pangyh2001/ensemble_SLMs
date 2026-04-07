import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
from config import *
from gate_model import GateNetwork
from embedding_manager import get_embedding_manager
from collections import Counter

class EmbeddingDataset(Dataset):
    """
    使用预计算embedding的数据集基类
    """
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings  # Tensor: (N, embed_dim)
        self.labels = labels  # List
        
    def __len__(self):
        return len(self.labels)


class MMLUEmbeddingDataset(EmbeddingDataset):
    def __init__(self, embeddings, labels, model_name, data_dict, model_list):
        super().__init__(embeddings, labels)
        # 当前模型的预测
        self.predictions = torch.FloatTensor(data_dict[model_name])
        
        # 所有模型的预测 (按model_list顺序)
        self.all_predictions = torch.stack([
            torch.FloatTensor(data_dict[m]) for m in model_list
        ], dim=1)  # (num_samples, num_models, 4)
        
        # 当前模型在model_list中的索引
        self.model_idx = model_list.index(model_name)
        
        print(f"  Dataset: all_predictions shape = {self.all_predictions.shape}")
        
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'prediction': self.predictions[idx],
            'all_predictions': self.all_predictions[idx],  # (num_models, 4)
            'label': self.labels[idx],
            'model_idx': self.model_idx
        }


class GSM8KEmbeddingDataset(EmbeddingDataset):
    def __init__(self, embeddings, raw_predictions, labels, model_idx):
        """
        Args:
            embeddings: (num_samples, embed_dim)
            raw_predictions: (num_samples, num_models, num_runs)
            labels: List[float]
            model_idx: int 当前模型的索引
        """
        super().__init__(embeddings, labels)
        # 当前模型的预测: (num_samples, num_runs)
        self.predictions = torch.FloatTensor(raw_predictions[:, model_idx, :])
        # 所有模型的预测
        self.all_predictions = torch.FloatTensor(raw_predictions)  # (num_samples, num_models, num_runs)
        self.model_idx = model_idx
        
        print(f"  Dataset: all_predictions shape = {self.all_predictions.shape}")
        
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'predictions': self.predictions[idx],  # (num_runs,)
            'all_predictions': self.all_predictions[idx],  # (num_models, num_runs)
            'label': self.labels[idx],
            'model_idx': self.model_idx
        }


def collate_fn_mmlu(batch):
    embeddings = torch.stack([item['embedding'] for item in batch])
    predictions = torch.stack([item['prediction'] for item in batch])
    all_predictions = torch.stack([item['all_predictions'] for item in batch])
    labels = [item['label'] for item in batch]
    model_idx = batch[0]['model_idx']  # 同一个batch的model_idx相同
    
    return {
        'embeddings': embeddings,
        'predictions': predictions,
        'all_predictions': all_predictions,  # (batch, num_models, 4)
        'labels': labels,
        'model_idx': model_idx
    }


def collate_fn_gsm8k(batch):
    embeddings = torch.stack([item['embedding'] for item in batch])
    predictions = torch.stack([item['predictions'] for item in batch])
    all_predictions = torch.stack([item['all_predictions'] for item in batch])
    labels = [item['label'] for item in batch]
    model_idx = batch[0]['model_idx']
    
    return {
        'embeddings': embeddings,
        'predictions': predictions,  # (batch, num_runs)
        'all_predictions': all_predictions,  # (batch, num_models, num_runs)
        'labels': labels,
        'model_idx': model_idx
    }


def compute_deviation_mmlu_relative(all_predictions, labels, model_idx):
    """
    计算相对偏离程度 - 考虑该模型相对于其他模型的表现（推荐使用）
    
    核心思想：Gate应该学习"我的模型什么时候比其他模型更适合回答这道题"
    
    Args:
        all_predictions: (batch, num_models, 4) - 所有模型的logits
        labels: List[str] - 'A', 'B', 'C', 'D'
        model_idx: int - 当前模型的索引
    Returns:
        deviation: (batch,) - 相对偏离程度 [0, 1]
            - 接近0: 当前模型比其他模型强很多（Gate应该给高分）
            - 接近1: 当前模型比其他模型弱很多（Gate应该给低分）
            - 0.5: 当前模型和其他模型差不多
    """
    batch_size = all_predictions.shape[0]
    num_models = all_predictions.shape[1]
    device = all_predictions.device
    
    # 转换标签为索引
    label_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    for i, label in enumerate(labels):
        label_indices[i] = label_map[label]
    
    # 计算所有模型的正确答案概率
    all_probs = torch.softmax(all_predictions, dim=2)  # (batch, num_models, 4)
    
    # 提取每个模型在正确答案上的概率
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_models)
    model_indices = torch.arange(num_models, device=device).unsqueeze(0).expand(batch_size, -1)
    label_indices_expanded = label_indices.unsqueeze(1).expand(-1, num_models)
    
    correct_probs = all_probs[batch_indices, model_indices, label_indices_expanded]  # (batch, num_models)
    
    # 当前模型的置信度
    current_prob = correct_probs[:, model_idx]  # (batch,)
    
    # 其他模型的平均置信度
    mask = torch.ones(num_models, dtype=torch.bool, device=device)
    mask[model_idx] = False
    other_probs_mean = correct_probs[:, mask].mean(dim=1)  # (batch,)
    
    # 相对优势：当前模型 vs 其他模型平均
    # relative_advantage 范围大约在 [-1, 1]
    # 当前模型更好时 > 0，更差时 < 0
    relative_advantage = current_prob - other_probs_mean
    
    # 转换为偏离度 [0, 1]
    # advantage > 0 (当前模型更好) -> deviation 应该小 (接近0)
    # advantage < 0 (当前模型更差) -> deviation 应该大 (接近1)
    # advantage = 0 (差不多) -> deviation = 0.5
    deviation = 0.5 - 0.5 * torch.tanh(relative_advantage)  # 使用tanh压缩到合理范围
    
    return deviation


def compute_deviation_gsm8k_relative(all_predictions, labels, model_idx):
    """
    计算GSM8K的相对偏离程度 - 考虑该模型相对于其他模型的表现（推荐使用）
    
    Args:
        all_predictions: (batch, num_models, num_runs) - 所有模型的原始答案值
        labels: List[float] - 正确答案
        model_idx: int - 当前模型的索引
    Returns:
        deviation: (batch,) - 相对偏离程度 [0, 1]
    """
    batch_size = all_predictions.shape[0]
    num_models = all_predictions.shape[1]
    num_runs = all_predictions.shape[2]
    device = all_predictions.device
    
    # 计算每个模型的准确率（10次采样中正确的比例）
    accuracies = torch.zeros(batch_size, num_models, device=device)
    
    for i in range(batch_size):
        label = labels[i]
        if np.isnan(label):
            # 无效标签，所有模型准确率都是0
            continue
        
        for j in range(num_models):
            correct_count = 0
            for k in range(num_runs):
                pred_val = all_predictions[i, j, k].item()
                try:
                    if abs(float(pred_val) - float(label)) < 1e-4:
                        correct_count += 1
                except:
                    pass
            accuracies[i, j] = correct_count / num_runs
    
    # 当前模型的准确率
    current_acc = accuracies[:, model_idx]  # (batch,)
    
    # 其他模型的平均准确率
    mask = torch.ones(num_models, dtype=torch.bool, device=device)
    mask[model_idx] = False
    other_acc_mean = accuracies[:, mask].mean(dim=1)  # (batch,)
    
    # 相对优势：当前准确率 - 其他平均准确率
    # 范围在 [-1, 1]
    relative_advantage = current_acc - other_acc_mean
    
    # 转换为偏离度
    # advantage > 0 (当前模型更好) -> deviation 小
    # advantage < 0 (当前模型更差) -> deviation 大
    deviation = 0.5 - 0.5 * torch.tanh(relative_advantage)
    
    return deviation


class GateTrainer:
    def __init__(self, task_type, model_name, embedding_dim, gate_type="mlp"):
        self.task_type = task_type
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化Gate模型
        self.gate_model = GateNetwork(
            input_dim=embedding_dim,
            gate_type=gate_type,
            hidden_dim=TRAIN_CONFIG["hidden_dim"],
            num_heads=TRAIN_CONFIG.get("num_heads", 8),
            num_blocks=TRAIN_CONFIG.get("num_blocks", 4),
            dropout=TRAIN_CONFIG["dropout"]
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.gate_model.parameters(), 
            lr=TRAIN_CONFIG["learning_rate"]
        )
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 添加统计信息
        self.deviation_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
        
    def train_epoch(self, dataloader):
        self.gate_model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training {self.model_name}")):
            embeddings = batch['embeddings'].to(self.device)
            labels = batch['labels']
            
            # 获取Gate分数
            scores = self.gate_model(embeddings)  # (batch, 1)
            
            # 计算相对偏离程度
            all_predictions = batch['all_predictions'].to(self.device)
            model_idx = batch['model_idx']
            
            if self.task_type == "mmlu":
                deviation = compute_deviation_mmlu_relative(all_predictions, labels, model_idx)
            else:  # gsm8k
                deviation = compute_deviation_gsm8k_relative(all_predictions, labels, model_idx)
            
            # 记录统计信息（第一个batch）
            if batch_idx == 0:
                self.deviation_stats['mean'].append(deviation.mean().item())
                self.deviation_stats['std'].append(deviation.std().item())
                self.deviation_stats['min'].append(deviation.min().item())
                self.deviation_stats['max'].append(deviation.max().item())
            
            # 目标：积极性 = 1 - 偏离程度
            target = (1 - deviation).unsqueeze(1)
            
            # MSE损失
            loss = torch.nn.functional.mse_loss(scores, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.gate_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['labels']
                
                scores = self.gate_model(embeddings)
                
                # 计算相对偏离程度
                all_predictions = batch['all_predictions'].to(self.device)
                model_idx = batch['model_idx']
                
                if self.task_type == "mmlu":
                    deviation = compute_deviation_mmlu_relative(all_predictions, labels, model_idx)
                else:  # gsm8k
                    deviation = compute_deviation_gsm8k_relative(all_predictions, labels, model_idx)
                
                target = (1 - deviation).unsqueeze(1)
                loss = torch.nn.functional.mse_loss(scores, target)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader):
        print(f"\n  Using RELATIVE deviation (comparing with other models)")
        print(f"  This should give different results for different gate architectures!\n")
        
        for epoch in range(TRAIN_CONFIG["num_epochs"]):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # 打印偏离度统计
            if epoch == 0:
                stats = self.deviation_stats
                print(f"\n  Deviation statistics (epoch 1):")
                print(f"    Mean: {stats['mean'][-1]:.4f}, Std: {stats['std'][-1]:.4f}")
                print(f"    Range: [{stats['min'][-1]:.4f}, {stats['max'][-1]:.4f}]")
                print(f"    (0=model is best, 0.5=same as others, 1=model is worst)\n")
            
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_model()
                print(f"  -> Best model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= TRAIN_CONFIG["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def save_model(self):
        gate_type = TRAIN_CONFIG.get("gate_type", "mlp")
        save_path = os.path.join(GATE_DIR, f"{self.model_name}_{self.task_type}_{gate_type}.pt")
        torch.save({
            'model_state_dict': self.gate_model.state_dict(),
            'best_loss': self.best_loss,
            'gate_type': gate_type,
            'deviation_stats': self.deviation_stats
        }, save_path)


def train_all_gates(task_type="mmlu", embedding_key="bert", gate_type="mlp"):
    """训练所有模型的Gate"""
    print(f"\n{'='*80}")
    print(f"Training Gates for {task_type.upper()}")
    print(f"Embedding: {embedding_key} (shared, frozen)")
    print(f"Gate Type: {gate_type}")
    print(f"Using RELATIVE deviation loss (model vs others)")
    print(f"{'='*80}\n")
    
    # 1. 获取共享的embedding manager并预计算embedding
    emb_manager = get_embedding_manager(embedding_key, device='cuda')
    train_embeddings, _ = emb_manager.precompute_embeddings(task_type, force_recompute=False)
    embedding_dim = emb_manager.get_encoder_dim()
    
    # 2. 加载原始训练数据
    split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
    with open(os.path.join(split_dir, f"{task_type}_train.pkl"), "rb") as f:
        train_data = pkl.load(f)
    
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
        DatasetClass = MMLUEmbeddingDataset
        collate_fn = collate_fn_mmlu
    else:
        model_list = GSM8K_TRAIN_MODELS
        DatasetClass = GSM8KEmbeddingDataset
        collate_fn = collate_fn_gsm8k
    
    print(f"Training gates for {len(model_list)} models: {model_list}\n")
    
    # 3. 为每个模型训练Gate
    for i, model_name in enumerate(model_list):
        print(f"\n{'='*80}")
        print(f"Training Gate for {model_name} ({i+1}/{len(model_list)})")
        print(f"{'='*80}")
        
        # 创建数据集
        if task_type == "mmlu":
            dataset = DatasetClass(
                train_embeddings,
                train_data['labels'], 
                model_name,
                train_data['data'],
                model_list  # 传入完整的模型列表
            )
        else:
            dataset = DatasetClass(
                train_embeddings,
                train_data['raw_predictions'],
                train_data['labels'],
                i  # model_idx
            )
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=TRAIN_CONFIG["batch_size"], 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=TRAIN_CONFIG["batch_size"], 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # 训练
        trainer = GateTrainer(task_type, model_name, embedding_dim, gate_type)
        trainer.train(train_loader, val_loader)
        
        print(f"✓ Gate for {model_name} trained successfully")
        print(f"  Best validation loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mmlu", choices=["mmlu", "gsm8k"])
    parser.add_argument("--embedding", default="bert")
    parser.add_argument("--gate_type", default="mlp", choices=["mlp", "attention", "resnet"])
    args = parser.parse_args()
    train_all_gates(args.task, args.embedding, args.gate_type)