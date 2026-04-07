import torch
import numpy as np
import os
import pickle as pkl
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from config import EMBEDDING_MODELS, TRAIN_CONFIG, CUR_DIR, DATA_DIR

class EmbeddingManager:
    """
    全局Embedding管理器 - 负责预计算和缓存问题的embedding向量
    所有Gate网络共享同一个frozen embedding encoder
    """
    
    def __init__(self, embedding_key="bert", device='cuda'):
        self.embedding_key = embedding_key
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        model_name = EMBEDDING_MODELS.get(embedding_key, "bert-base-uncased")
        print(f"Initializing shared embedding encoder: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 冻结encoder参数 - 不参与训练
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.encoder.eval()  # 设置为评估模式
        
        self.encoder_dim = self.encoder.config.hidden_size
        self.cache = {}  # 缓存已计算的embedding
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 考虑attention mask的平均池化"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def encode_batch(self, questions, batch_size=32):
        """
        批量编码问题
        Args:
            questions: List[str]
            batch_size: int
        Returns:
            embeddings: (len(questions), encoder_dim)
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i+batch_size]
                
                encoding = self.tokenizer(
                    batch_questions,
                    padding=True,
                    truncation=True,
                    max_length=TRAIN_CONFIG["max_length"],
                    return_tensors="pt"
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                model_output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Mean pooling
                batch_embeddings = self.mean_pooling(model_output, attention_mask)
                all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def precompute_embeddings(self, task_type, force_recompute=False):
        """
        预计算训练集和测试集的所有embedding并缓存到磁盘
        Args:
            task_type: "mmlu" or "gsm8k"
            force_recompute: 是否强制重新计算
        Returns:
            train_embeddings, test_embeddings: Tensor
        """
        cache_dir = os.path.join(CUR_DIR, DATA_DIR, "embedding_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(
            cache_dir, 
            f"{task_type}_{self.embedding_key}_embeddings.pkl"
        )
        
        # 检查缓存
        if not force_recompute and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pkl.load(f)
            return cached_data['train_embeddings'], cached_data['test_embeddings']
        
        # 加载数据
        print(f"Computing embeddings for {task_type}...")
        split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
        
        with open(os.path.join(split_dir, f"{task_type}_train.pkl"), "rb") as f:
            train_data = pkl.load(f)
        with open(os.path.join(split_dir, f"{task_type}_test.pkl"), "rb") as f:
            test_data = pkl.load(f)
        
        # 计算训练集embedding
        print("  Computing train embeddings...")
        train_questions = train_data['questions']
        train_embeddings = self.encode_batch(
            train_questions, 
            batch_size=32
        )
        
        # 计算测试集embedding
        print("  Computing test embeddings...")
        test_questions = test_data['questions']
        test_embeddings = self.encode_batch(
            test_questions,
            batch_size=32
        )
        
        # 缓存到磁盘
        print(f"  Saving embeddings to {cache_file}")
        with open(cache_file, 'wb') as f:
            pkl.dump({
                'train_embeddings': train_embeddings,
                'test_embeddings': test_embeddings,
                'embedding_key': self.embedding_key,
                'encoder_dim': self.encoder_dim
            }, f)
        
        print(f"  ✓ Train: {train_embeddings.shape}, Test: {test_embeddings.shape}")
        
        return train_embeddings, test_embeddings
    
    def encode_single(self, question):
        """
        编码单个问题（用于推理时）
        Args:
            question: str
        Returns:
            embedding: (encoder_dim,)
        """
        # 检查缓存
        if question in self.cache:
            return self.cache[question]
        
        with torch.no_grad():
            encoding = self.tokenizer(
                [question],
                padding=True,
                truncation=True,
                max_length=TRAIN_CONFIG["max_length"],
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            model_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            embedding = self.mean_pooling(model_output, attention_mask)[0].cpu()
            
            # 缓存结果
            self.cache[question] = embedding
            
            return embedding
    
    def get_encoder_dim(self):
        """返回encoder的输出维度"""
        return self.encoder_dim


# 全局单例 - 确保整个程序只有一个embedding encoder
_global_embedding_manager = None

def get_embedding_manager(embedding_key="bert", device='cuda'):
    """获取全局embedding manager单例"""
    global _global_embedding_manager
    
    if _global_embedding_manager is None:
        _global_embedding_manager = EmbeddingManager(embedding_key, device)
    elif _global_embedding_manager.embedding_key != embedding_key:
        # 如果需要切换embedding类型，重新创建
        print(f"Switching embedding from {_global_embedding_manager.embedding_key} to {embedding_key}")
        _global_embedding_manager = EmbeddingManager(embedding_key, device)
    
    return _global_embedding_manager