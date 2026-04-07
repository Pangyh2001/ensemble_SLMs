import os
import sys
import numpy as np
import torch
import json
import pickle as pkl
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import random
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_loader import load_mmlu_data, normalize_question
from embedding_manager import get_embedding_manager
from gate_model import GateNetwork
from trainer import GateTrainer, MMLUEmbeddingDataset, collate_fn_mmlu

class MMLUCombinedExperiment:
    """MMLU完整实验（合并所有功能）"""
    
    def __init__(self):
        self.seeds = [42, 123, 0]  # 固定的3个seed
        self.thresholds = [i/10 for i in range(1, 10)]  # 0.1-0.9
        self.gate_mmlu_dir = "gate_mmlu_main"  # 专门存放MMLU的门控模型
        self.split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits_main")  # 主实验数据划分目录
        self.results_dir = "results_mmlu_main"  # 结果目录
        
        # 创建必要的目录
        os.makedirs(self.gate_mmlu_dir, exist_ok=True)
        os.makedirs(self.split_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 全局embedding manager
        self.emb_manager = get_embedding_manager(SELECTED_EMBEDDING, self.device)
        
        # 实验数据
        self.train_data = None
        self.test_data = None
        self.train_embeddings = None
        self.test_embeddings = None
        self.embedding_dim = None
        
    def split_data_by_topic_once(self, test_size=0.2):
        """
        按领域划分数据（只做一次）
        1. 对每个csv文件（每个领域）独立进行8:2划分
        2. 所有领域的"8"混合作为训练集，"2"混合作为测试集
        3. 混合后随机打乱
        """
        print("="*80)
        print("STEP 1: Splitting MMLU data by topic (8:2 per topic)")
        print("="*80)
        
        # 检查是否已经划分过
        train_path = os.path.join(self.split_dir, "mmlu_main_train.pkl")
        test_path = os.path.join(self.split_dir, "mmlu_main_test.pkl")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("Data already split, loading from cache...")
            with open(train_path, 'rb') as f:
                self.train_data = pkl.load(f)
            with open(test_path, 'rb') as f:
                self.test_data = pkl.load(f)
            return
        
        print("Performing one-time data split...")
        
        # 加载原始数据
        print("Loading MMLU data...")
        data_dict, questions, labels, topics = load_mmlu_data(MMLU_TRAIN_MODELS)
        
        # 获取所有唯一的领域
        unique_topics = sorted(set(topics))
        print(f"Found {len(unique_topics)} topics:")
        topic_counts = Counter(topics)
        for i, (topic, count) in enumerate(topic_counts.most_common()):
            print(f"  {i+1:2d}. {topic}: {count} samples")
        
        # 为每个模型准备训练和测试索引
        train_indices_all = []
        test_indices_all = []
        
        # 对每个领域进行划分
        print("\nSplitting each topic (8:2 train:test)...")
        for topic in unique_topics:
            # 获取该领域的所有样本索引
            topic_indices = [i for i, t in enumerate(topics) if t == topic]
            
            if len(topic_indices) < 5:  # 如果样本太少，全部放入训练集
                print(f"  {topic}: {len(topic_indices)} samples (too few, all to train)")
                train_indices_all.extend(topic_indices)
                continue
            
            # 对该领域的样本进行分层抽样（按标签）
            topic_labels = [labels[i] for i in topic_indices]
            
            # 划分训练和测试（8:2）
            try:
                train_idx, test_idx = train_test_split(
                    topic_indices,
                    test_size=test_size,
                    random_state=42,  # 固定随机种子确保可重现
                    stratify=topic_labels
                )
            except:
                # 如果分层失败，使用简单划分
                print(f"  {topic}: Stratified split failed, using random split")
                train_idx, test_idx = train_test_split(
                    topic_indices,
                    test_size=test_size,
                    random_state=42
                )
            
            train_indices_all.extend(train_idx)
            test_indices_all.extend(test_idx)
            
            print(f"  {topic}: {len(train_idx)} train, {len(test_idx)} test samples")
        
        # 随机打乱（确保不同领域的样本混合）
        print("\nShuffling mixed data...")
        np.random.seed(42)
        np.random.shuffle(train_indices_all)
        np.random.shuffle(test_indices_all)
        
        # 统计信息
        print(f"\nTotal statistics:")
        print(f"  Train samples: {len(train_indices_all)}")
        print(f"  Test samples: {len(test_indices_all)}")
        print(f"  Total samples: {len(train_indices_all) + len(test_indices_all)}")
        
        # 统计标签分布
        train_labels = [labels[i] for i in train_indices_all]
        test_labels = [labels[i] for i in test_indices_all]
        
        print(f"\nLabel distribution in train: {Counter(train_labels)}")
        print(f"Label distribution in test: {Counter(test_labels)}")
        
        # 准备训练数据
        self.train_data = {
            'data': {m: data_dict[m][train_indices_all] for m in MMLU_TRAIN_MODELS},
            'questions': [questions[i] for i in train_indices_all],
            'labels': [labels[i] for i in train_indices_all],
            'topics': [topics[i] for i in train_indices_all],
            'indices': train_indices_all
        }
        
        # 准备测试数据
        self.test_data = {
            'data': {m: data_dict[m][test_indices_all] for m in MMLU_TRAIN_MODELS},
            'questions': [questions[i] for i in test_indices_all],
            'labels': [labels[i] for i in test_indices_all],
            'topics': [topics[i] for i in test_indices_all],
            'indices': test_indices_all
        }
        
        # 保存划分
        with open(train_path, 'wb') as f:
            pkl.dump(self.train_data, f)
        with open(test_path, 'wb') as f:
            pkl.dump(self.test_data, f)
        
        print(f"\n✓ Data saved to {self.split_dir}")
        
        # 保存划分信息为CSV便于查看
        self.save_split_info(questions, labels, topics, train_indices_all, test_indices_all)
    
    def save_split_info(self, questions, labels, topics, train_indices, test_indices):
        """保存划分信息为CSV"""
        info_data = []
        
        for idx in train_indices:
            info_data.append({
                'index': idx,
                'question': questions[idx][:100] + '...' if len(questions[idx]) > 100 else questions[idx],
                'label': labels[idx],
                'topic': topics[idx],
                'split': 'train'
            })
        
        for idx in test_indices:
            info_data.append({
                'index': idx,
                'question': questions[idx][:100] + '...' if len(questions[idx]) > 100 else questions[idx],
                'label': labels[idx],
                'topic': topics[idx],
                'split': 'test'
            })
        
        df_info = pd.DataFrame(info_data)
        info_path = os.path.join(self.split_dir, "split_info.csv")
        df_info.to_csv(info_path, index=False)
        print(f"  Split info saved to: {info_path}")
    
    def precompute_embeddings_once(self):
        """预计算embedding（只做一次）"""
        print("\n" + "="*80)
        print("STEP 2: Precomputing embeddings")
        print("="*80)
        
        cache_dir = os.path.join(CUR_DIR, DATA_DIR, "embedding_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(
            cache_dir, 
            f"mmlu_main_{SELECTED_EMBEDDING}_embeddings.pkl"
        )
        
        # 检查缓存
        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pkl.load(f)
            self.train_embeddings = cached_data['train_embeddings']
            self.test_embeddings = cached_data['test_embeddings']
            self.embedding_dim = cached_data['encoder_dim']
        else:
            print("Computing embeddings...")
            
            # 计算训练集embedding
            print("  Computing train embeddings...")
            train_questions = self.train_data['questions']
            self.train_embeddings = self.emb_manager.encode_batch(train_questions, batch_size=32)
            
            # 计算测试集embedding
            print("  Computing test embeddings...")
            test_questions = self.test_data['questions']
            self.test_embeddings = self.emb_manager.encode_batch(test_questions, batch_size=32)
            
            self.embedding_dim = self.emb_manager.get_encoder_dim()
            
            # 缓存到磁盘
            print(f"  Saving embeddings to {cache_file}")
            with open(cache_file, 'wb') as f:
                pkl.dump({
                    'train_embeddings': self.train_embeddings,
                    'test_embeddings': self.test_embeddings,
                    'embedding_key': SELECTED_EMBEDDING,
                    'encoder_dim': self.embedding_dim
                }, f)
        
        print(f"  ✓ Train embeddings: {self.train_embeddings.shape}")
        print(f"  ✓ Test embeddings: {self.test_embeddings.shape}")
        print(f"  ✓ Embedding dimension: {self.embedding_dim}")
    
    def prepare_all_data(self):
        """准备所有数据（划分 + embedding）"""
        print("\n" + "="*80)
        print("DATA PREPARATION PHASE")
        print("="*80)
        
        # 1. 划分数据
        self.split_data_by_topic_once()
        
        # 2. 预计算embedding
        self.precompute_embeddings_once()
    
    def train_gate_models(self, seed):
        """为指定seed训练所有门控模型"""
        print(f"\n{'='*80}")
        print(f"Training Gate Models for Seed {seed}")
        print(f"{'='*80}")
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 为每个模型训练Gate
        model_list = MMLU_TRAIN_MODELS
        
        for i, model_name in enumerate(model_list):
            print(f"\nTraining Gate for {model_name} ({i+1}/{len(model_list)})")
            
            # 检查模型是否已经存在
            model_path = os.path.join(
                self.gate_mmlu_dir, 
                f"{model_name}_mmlu_{GATE_TYPE}_seed{seed}.pt"
            )
            
            if os.path.exists(model_path):
                print(f"  Model already exists, skipping...")
                continue
            
            # 创建数据集
            dataset = MMLUEmbeddingDataset(
                self.train_embeddings,
                self.train_data['labels'], 
                model_name,
                self.train_data['data'],
                model_list
            )
            
            # 划分训练集和验证集 (80-20)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            
            # 创建DataLoader
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=True,
                collate_fn=collate_fn_mmlu
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=False,
                collate_fn=collate_fn_mmlu
            )
            
            # 训练
            trainer = GateTrainer("mmlu", model_name, self.embedding_dim, GATE_TYPE)
            trainer.train(train_loader, val_loader)
            
            # 保存模型
            torch.save({
                'model_state_dict': trainer.gate_model.state_dict(),
                'seed': seed,
                'model_name': model_name,
                'gate_type': GATE_TYPE,
                'best_loss': trainer.best_loss
            }, model_path)
            
            print(f"  ✓ Saved gate model to {model_path}")
    
    def train_all_seeds(self):
        """训练所有seed的门控模型"""
        print("\n" + "="*80)
        print("TRAINING PHASE")
        print("="*80)
        
        for seed in self.seeds:
            self.train_gate_models(seed)
    
    def load_gate_models(self, seed):
        """加载指定seed的门控模型"""
        gate_models = {}
        model_list = MMLU_TRAIN_MODELS
        
        for i, model_name in enumerate(model_list):
            model_path = os.path.join(
                self.gate_mmlu_dir,
                f"{model_name}_mmlu_{GATE_TYPE}_seed{seed}.pt"
            )
            
            if not os.path.exists(model_path):
                print(f"Warning: Gate model not found for {model_name} seed {seed}")
                continue
            
            # 初始化Gate模型
            gate = GateNetwork(
                input_dim=self.embedding_dim,
                gate_type=GATE_TYPE,
                hidden_dim=TRAIN_CONFIG["hidden_dim"],
                num_heads=TRAIN_CONFIG.get("num_heads", 8),
                num_blocks=TRAIN_CONFIG.get("num_blocks", 4),
                dropout=TRAIN_CONFIG["dropout"]
            ).to(self.device)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            gate.load_state_dict(checkpoint['model_state_dict'])
            gate.eval()
            
            gate_models[model_name] = gate
        
        return gate_models
    
    def get_gate_scores(self, gate_models, embedding):
        """获取所有Gate的分数"""
        scores = {}
        embedding_batch = embedding.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for model_name, gate_model in gate_models.items():
                s = gate_model(embedding_batch)
                scores[model_name] = s.item()
        
        return scores
    
    def predict_mmlu_ensemble(self, model_predictions, selected_models, idx):
        """MMLU集成预测（软投票）"""
        total_logits = np.zeros(4)
        for model_name in selected_models:
            total_logits += np.exp(model_predictions[model_name][idx])
        
        # 选择概率最高的选项
        pred_class = ['A', 'B', 'C', 'D'][np.argmax(total_logits)]
        return pred_class
    
    def evaluate_threshold(self, seed, threshold):
        """评估特定seed和阈值下的表现"""
        # 加载门控模型
        gate_models = self.load_gate_models(seed)
        if not gate_models:
            print(f"No gate models found for seed {seed}")
            return None
        
        # 设置随机种子用于random baseline
        random.seed(seed)
        np.random.seed(seed)
        
        test_questions = self.test_data['questions']
        test_labels = self.test_data['labels']
        model_predictions = self.test_data['data']
        
        total_samples = len(test_questions)
        correct_ensemble = 0
        correct_random = 0
        
        # 统计每个模型的激活次数
        activation_counts = {model: 0 for model in MMLU_TRAIN_MODELS}
        # 统计每个问题选择的模型数量
        selected_models_counts = []
        
        for idx in tqdm(range(total_samples), desc=f"Seed {seed}, Threshold {threshold}"):
            # 获取当前问题的embedding
            embedding = self.test_embeddings[idx]
            
            # 获取门控分数
            gate_scores = self.get_gate_scores(gate_models, embedding)
            
            # 1. Ensemble策略（基于阈值）
            # 选择门控分数超过阈值的模型
            selected_models = [m for m, s in gate_scores.items() if s > threshold]
            
            # 如果没有模型超过阈值，选择分数最高的模型
            if not selected_models:
                selected_models = [max(gate_scores, key=gate_scores.get)]
            
            # 记录激活次数
            for model in selected_models:
                activation_counts[model] += 1
            
            # 记录每个问题选择的模型数量
            selected_models_counts.append(len(selected_models))
            
            # Ensemble预测
            ensemble_pred = self.predict_mmlu_ensemble(model_predictions, selected_models, idx)
            if ensemble_pred == test_labels[idx]:
                correct_ensemble += 1
            
            # 2. Random baseline（随机选择相同数量的模型）
            k = len(selected_models)
            random_models = np.random.choice(MMLU_TRAIN_MODELS, k, replace=False)
            random_pred = self.predict_mmlu_ensemble(model_predictions, random_models, idx)
            if random_pred == test_labels[idx]:
                correct_random += 1
        
        # 计算准确率
        accuracy_ensemble = correct_ensemble / total_samples
        accuracy_random = correct_random / total_samples
        
        # 计算激活频率（相对于总问题数）
        activation_freq = {
            model: count / total_samples  # 改为除以总问题数
            for model, count in activation_counts.items()
        }
        
        # 计算平均每个问题选择的模型数量
        avg_selected_models = np.mean(selected_models_counts)
        
        return {
            'accuracy_ensemble': accuracy_ensemble,
            'accuracy_random': accuracy_random,
            'activation_freq': activation_freq,
            'avg_selected_models': avg_selected_models,
            'total_samples': total_samples,
            'activation_counts': activation_counts
        }
    
    def evaluate_all_thresholds(self):
        """评估所有阈值"""
        print("\n" + "="*80)
        print("EVALUATION PHASE")
        print("="*80)
        
        results = {}
        
        for threshold in self.thresholds:
            threshold_key = f"threshold_{threshold:.1f}"
            results[threshold_key] = {
                'seed_results': {},
                'acc_ensemble_mean': 0,
                'acc_ensemble_std': 0,
                'acc_random_mean': 0,
                'acc_random_std': 0,
                'avg_selected_models_mean': 0,  # 新增：平均选择的模型数量
                'avg_selected_models_std': 0,
                'activation_freqs': {model: [] for model in MMLU_TRAIN_MODELS},
                'activation_counts': {model: [] for model in MMLU_TRAIN_MODELS}
            }
            
            # 对每个seed进行评估
            seed_acc_ensemble = []
            seed_acc_random = []
            seed_avg_selected = []
            
            for seed in self.seeds:
                print(f"\nEvaluating: Threshold={threshold:.1f}, Seed={seed}")
                result = self.evaluate_threshold(seed, threshold)
                
                if result is not None:
                    results[threshold_key]['seed_results'][seed] = result
                    
                    seed_acc_ensemble.append(result['accuracy_ensemble'])
                    seed_acc_random.append(result['accuracy_random'])
                    seed_avg_selected.append(result['avg_selected_models'])
                    
                    # 收集激活频率和激活次数
                    for model, freq in result['activation_freq'].items():
                        results[threshold_key]['activation_freqs'][model].append(freq)
                    
                    for model, count in result['activation_counts'].items():
                        results[threshold_key]['activation_counts'][model].append(count)
            
            # 计算统计量
            if seed_acc_ensemble:
                results[threshold_key]['acc_ensemble_mean'] = np.mean(seed_acc_ensemble)
                results[threshold_key]['acc_ensemble_std'] = np.std(seed_acc_ensemble, ddof=1)
                
                results[threshold_key]['acc_random_mean'] = np.mean(seed_acc_random)
                results[threshold_key]['acc_random_std'] = np.std(seed_acc_random, ddof=1)
                
                results[threshold_key]['avg_selected_models_mean'] = np.mean(seed_avg_selected)
                results[threshold_key]['avg_selected_models_std'] = np.std(seed_avg_selected, ddof=1)
        
        return results
    
    def generate_csv_report(self, results):
        """生成CSV报告"""
        csv_data = []
        
        for threshold in self.thresholds:
            threshold_key = f"threshold_{threshold:.1f}"
            result = results[threshold_key]
            
            # 基本信息
            row = {
                'threshold': threshold,
                'acc_ensemble_mean': f"{result['acc_ensemble_mean']:.4f}",
                'acc_ensemble_std': f"{result['acc_ensemble_std']:.4f}",
                'acc_random_mean': f"{result['acc_random_mean']:.4f}",
                'acc_random_std': f"{result['acc_random_std']:.4f}",
                'avg_selected_models': f"{result['avg_selected_models_mean']:.2f} ± {result['avg_selected_models_std']:.2f}",
                'ensemble_formatted': f"{result['acc_ensemble_mean']:.4f} ± {result['acc_ensemble_std']:.4f}",
                'random_formatted': f"{result['acc_random_mean']:.4f} ± {result['acc_random_std']:.4f}"
            }
            
            # 每个模型的激活频率（平均±标准差）
            for model in MMLU_TRAIN_MODELS:
                freqs = result['activation_freqs'][model]
                if freqs:
                    mean_freq = np.mean(freqs)
                    std_freq = np.std(freqs, ddof=1)
                    row[f'{model}_freq'] = f"{mean_freq:.4f} ± {std_freq:.4f}"
                else:
                    row[f'{model}_freq'] = "0.0000 ± 0.0000"
            
            csv_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(csv_data)
        
        # 重新排序列
        columns = ['threshold', 'acc_ensemble_mean', 'acc_ensemble_std', 
                  'acc_random_mean', 'acc_random_std', 'avg_selected_models',
                  'ensemble_formatted', 'random_formatted']
        for model in MMLU_TRAIN_MODELS:
            columns.append(f'{model}_freq')
        
        df = df[columns]
        
        # 保存CSV
        csv_file = os.path.join(self.results_dir, "mmlu_main_results.csv")
        df.to_csv(csv_file, index=False)
        
        return csv_file, df
    
    def save_detailed_results(self, results):
        """保存详细结果到JSON文件"""
        output_file = os.path.join(self.results_dir, "mmlu_main_detailed_results.json")
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return output_file
    
    def print_summary(self, results, df):
        """打印结果摘要"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print(f"\nOverall Results:")
        summary_columns = ['threshold', 'ensemble_formatted', 'random_formatted', 'avg_selected_models']
        print(df[summary_columns].to_string(index=False))
        
        # 找出最佳阈值
        best_row = df.loc[df['acc_ensemble_mean'].astype(float).idxmax()]
        print(f"\nBest threshold: {best_row['threshold']}")
        print(f"Best ensemble accuracy: {best_row['ensemble_formatted']}")
        print(f"Corresponding random baseline: {best_row['random_formatted']}")
        print(f"Average models selected per question: {best_row['avg_selected_models']}")
        
        print(f"\nModel activation frequencies at best threshold:")
        best_threshold = best_row['threshold']
        threshold_key = f"threshold_{best_threshold:.1f}"
        result = results[threshold_key]
        
        activation_means = {}
        for model in MMLU_TRAIN_MODELS:
            freqs = result['activation_freqs'][model]
            if freqs:
                activation_means[model] = np.mean(freqs)
        
        # 按激活频率排序
        sorted_models = sorted(activation_means.items(), key=lambda x: x[1], reverse=True)
        for model, freq in sorted_models:
            print(f"  {model:<30}: {freq:.4f}")
        
        # 打印激活频率总和（应该大于0，但可能小于1）
        total_freq = sum(activation_means.values())
        print(f"\n  Total activation frequency (should be >0, may be <1): {total_freq:.4f}")
        print(f"  Note: Sum may be less than 1.0 because multiple models can be selected per question")
        
        # 计算并打印平均每个问题选择的模型数量
        total_samples = 0
        total_activations = 0
        for seed in self.seeds:
            if seed in result['seed_results']:
                total_samples += result['seed_results'][seed]['total_samples']
                # 所有模型的激活次数总和
                for count in result['seed_results'][seed]['activation_counts'].values():
                    total_activations += count
        
        avg_models_per_question = total_activations / total_samples if total_samples > 0 else 0
        print(f"  Average models selected per question (calculated): {avg_models_per_question:.2f}")
    
    def run_full_experiment(self):
        """运行完整实验"""
        print("="*80)
        print("MMLU MAIN EXPERIMENT - Complete Pipeline")
        print("="*80)
        print(f"Seeds: {self.seeds}")
        print(f"Thresholds: {self.thresholds}")
        print(f"Models: {len(MMLU_TRAIN_MODELS)}")
        print(f"Gate type: {GATE_TYPE}")
        print(f"Embedding: {SELECTED_EMBEDDING}")
        print("="*80)
        
        # 阶段1: 数据准备
        self.prepare_all_data()
        
        # 阶段2: 训练
        self.train_all_seeds()
        
        # 阶段3: 评估
        results = self.evaluate_all_thresholds()
        
        # 阶段4: 生成报告
        print("\n" + "="*80)
        print("REPORT GENERATION")
        print("="*80)
        
        # 保存详细结果
        json_file = self.save_detailed_results(results)
        print(f"✓ Detailed results saved to {json_file}")
        
        # 生成CSV报告
        csv_file, df = self.generate_csv_report(results)
        print(f"✓ CSV report saved to {csv_file}")
        
        # 打印摘要
        self.print_summary(results, df)
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return results, df
    
    def run_specific_threshold(self, threshold):
        """只运行特定阈值的评估"""
        if threshold not in self.thresholds:
            print(f"Error: Threshold must be one of {self.thresholds}")
            return
        
        print("="*80)
        print(f"MMLU EXPERIMENT - Threshold {threshold} only")
        print("="*80)
        
        # 阶段1: 数据准备
        self.prepare_all_data()
        
        # 阶段2: 训练（如果需要）
        for seed in self.seeds:
            self.train_gate_models(seed)
        
        # 阶段3: 评估特定阈值
        results = {}
        threshold_key = f"threshold_{threshold:.1f}"
        results[threshold_key] = {
            'seed_results': {},
            'acc_ensemble_mean': 0,
            'acc_ensemble_std': 0,
            'acc_random_mean': 0,
            'acc_random_std': 0,
            'avg_selected_models_mean': 0,
            'avg_selected_models_std': 0,
            'activation_freqs': {model: [] for model in MMLU_TRAIN_MODELS},
            'activation_counts': {model: [] for model in MMLU_TRAIN_MODELS}
        }
        
        seed_acc_ensemble = []
        seed_acc_random = []
        seed_avg_selected = []
        
        for seed in self.seeds:
            print(f"\nEvaluating: Threshold={threshold:.1f}, Seed={seed}")
            result = self.evaluate_threshold(seed, threshold)
            
            if result is not None:
                results[threshold_key]['seed_results'][seed] = result
                seed_acc_ensemble.append(result['accuracy_ensemble'])
                seed_acc_random.append(result['accuracy_random'])
                seed_avg_selected.append(result['avg_selected_models'])
                
                for model, freq in result['activation_freq'].items():
                    results[threshold_key]['activation_freqs'][model].append(freq)
                
                for model, count in result['activation_counts'].items():
                    results[threshold_key]['activation_counts'][model].append(count)
        
        if seed_acc_ensemble:
            results[threshold_key]['acc_ensemble_mean'] = np.mean(seed_acc_ensemble)
            results[threshold_key]['acc_ensemble_std'] = np.std(seed_acc_ensemble, ddof=1)
            results[threshold_key]['acc_random_mean'] = np.mean(seed_acc_random)
            results[threshold_key]['acc_random_std'] = np.std(seed_acc_random, ddof=1)
            results[threshold_key]['avg_selected_models_mean'] = np.mean(seed_avg_selected)
            results[threshold_key]['avg_selected_models_std'] = np.std(seed_avg_selected, ddof=1)
        
        # 打印结果
        print(f"\n{'='*80}")
        print(f"RESULTS FOR THRESHOLD {threshold}")
        print(f"{'='*80}")
        print(f"Ensemble Accuracy: {results[threshold_key]['acc_ensemble_mean']:.4f} ± {results[threshold_key]['acc_ensemble_std']:.4f}")
        print(f"Random Baseline:   {results[threshold_key]['acc_random_mean']:.4f} ± {results[threshold_key]['acc_random_std']:.4f}")
        print(f"Average models selected per question: {results[threshold_key]['avg_selected_models_mean']:.2f} ± {results[threshold_key]['avg_selected_models_std']:.2f}")
        
        # 打印激活频率
        print(f"\nModel Activation Frequencies:")
        activation_means = {}
        for model in MMLU_TRAIN_MODELS:
            freqs = results[threshold_key]['activation_freqs'][model]
            if freqs:
                activation_means[model] = np.mean(freqs)
        
        for model, freq in sorted(activation_means.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:<30}: {freq:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='MMLU Main Experiment - Complete Pipeline')
    parser.add_argument('--mode', default='full', choices=['full', 'data_only', 'train_only', 'eval_only', 'threshold'],
                       help='运行模式: full(完整), data_only(只准备数据), train_only(只训练), eval_only(只评估), threshold(特定阈值)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='特定阈值评估时的阈值 (0.1-0.9)')
    parser.add_argument('--skip_data', action='store_true',
                       help='跳过数据准备阶段 (使用已有的数据)')
    parser.add_argument('--skip_train', action='store_true',
                       help='跳过训练阶段 (使用已有的模型)')
    
    args = parser.parse_args()
    
    # 创建实验对象
    experiment = MMLUCombinedExperiment()
    
    if args.mode == 'full':
        # 完整实验
        experiment.run_full_experiment()
    
    elif args.mode == 'data_only':
        # 只准备数据
        experiment.prepare_all_data()
        print("\n✓ Data preparation completed!")
    
    elif args.mode == 'train_only':
        # 只训练
        if not args.skip_data:
            experiment.prepare_all_data()
        experiment.train_all_seeds()
        print("\n✓ Training completed!")
    
    elif args.mode == 'eval_only':
        # 只评估
        if not args.skip_data:
            experiment.prepare_all_data()
        
        # 检查模型是否存在
        models_exist = True
        for seed in experiment.seeds:
            for model_name in MMLU_TRAIN_MODELS:
                model_path = os.path.join(
                    experiment.gate_mmlu_dir,
                    f"{model_name}_mmlu_{GATE_TYPE}_seed{seed}.pt"
                )
                if not os.path.exists(model_path):
                    models_exist = False
                    break
        
        if not models_exist and not args.skip_train:
            print("Models not found, training first...")
            experiment.train_all_seeds()
        
        results = experiment.evaluate_all_thresholds()
        csv_file, df = experiment.generate_csv_report(results)
        experiment.print_summary(results, df)
    
    elif args.mode == 'threshold':
        # 特定阈值评估
        if not args.skip_data:
            experiment.prepare_all_data()
        
        # 检查模型是否存在
        models_exist = True
        for seed in experiment.seeds:
            for model_name in MMLU_TRAIN_MODELS:
                model_path = os.path.join(
                    experiment.gate_mmlu_dir,
                    f"{model_name}_mmlu_{GATE_TYPE}_seed{seed}.pt"
                )
                if not os.path.exists(model_path):
                    models_exist = False
                    break
        
        if not models_exist and not args.skip_train:
            print("Models not found, training first...")
            experiment.train_all_seeds()
        
        experiment.run_specific_threshold(args.threshold)
    
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()