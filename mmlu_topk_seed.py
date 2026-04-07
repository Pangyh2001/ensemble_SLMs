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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_loader import load_mmlu_data, normalize_question
from embedding_manager import get_embedding_manager
from gate_model import GateNetwork
from trainer import GateTrainer, MMLUEmbeddingDataset, collate_fn_mmlu

class MMLUTopKExperiment:
    """MMLU Top-K实验（按领域划分，使用Top4选择策略）"""
    
    def __init__(self):
        # 固定的领域划分
        self.train_subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_european_history",
            "high_school_geography", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_statistics",
            "high_school_us_history", "high_school_world_history",
            "human_sexuality", "international_law", "jurisprudence",
            "logical_fallacies", "machine_learning", "management",
            "marketing", "medical_genetics", "moral_disputes",
            "moral_scenarios", "nutrition", "philosophy", "prehistory",
            "professional_accounting", "professional_law",
            "professional_medicine", "public_relations", "security_studies",
            "virology", "world_religions"
        ]
        
        self.test_subjects = [
            "electrical_engineering", "high_school_government_and_politics",
            "high_school_psychology", "human_aging", "miscellaneous",
            "professional_psychology", "sociology", "us_foreign_policy"
        ]
        
        self.seeds = [42, 123, 0]  # 固定的3个seed
        self.top_k = 4  # Top4选择策略
        self.gate_mmlu_dir = "gate_mmlu_topk"  # 专门存放MMLU的门控模型
        self.split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits_topk")  # 实验数据划分目录
        self.results_dir = "results_mmlu_topk"  # 结果目录
        
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
        
    def split_data_by_subject(self):
        """
        按固定领域划分数据
        训练领域：self.train_subjects
        测试领域：self.test_subjects
        """
        print("="*80)
        print("STEP 1: Splitting MMLU data by fixed subjects")
        print("="*80)
        
        # 检查是否已经划分过
        train_path = os.path.join(self.split_dir, "mmlu_topk_train.pkl")
        test_path = os.path.join(self.split_dir, "mmlu_topk_test.pkl")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("Data already split, loading from cache...")
            with open(train_path, 'rb') as f:
                self.train_data = pkl.load(f)
            with open(test_path, 'rb') as f:
                self.test_data = pkl.load(f)
            
            # 确保 test_data 有 all_indices
            if 'all_indices' not in self.test_data:
                print("Warning: test_data missing 'all_indices', creating it...")
                # 从 by_subject 重建 all_indices
                all_indices = []
                for subject, data in self.test_data['by_subject'].items():
                    if 'indices' in data:
                        all_indices.extend(data['indices'])
                # 去重并排序
                all_indices = list(sorted(set(all_indices)))
                self.test_data['all_indices'] = all_indices
                print(f"  Created all_indices with {len(all_indices)} indices")
            
            return
        
        print("Performing subject-based data split...")
        
        # 加载原始数据
        print("Loading MMLU data...")
        data_dict, questions, labels, topics = load_mmlu_data(MMLU_TRAIN_MODELS)
        
        # 获取训练和测试的索引
        train_indices = []
        test_indices = []
        
        for idx, topic in enumerate(topics):
            if topic in self.train_subjects:
                train_indices.append(idx)
            elif topic in self.test_subjects:
                test_indices.append(idx)
            else:
                # 如果topic不在任何一个列表中，打印警告
                print(f"Warning: Topic '{topic}' not in train or test subjects. Adding to train.")
                train_indices.append(idx)
        
        # 随机打乱训练集和测试集（领域之间混合）
        print("\nShuffling data within train and test sets...")
        np.random.seed(42)  # 固定随机种子确保可重现
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # 统计信息
        print(f"\nTotal statistics:")
        print(f"  Train subjects: {len(self.train_subjects)}")
        print(f"  Test subjects: {len(self.test_subjects)}")
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Test samples: {len(test_indices)}")
        print(f"  Total samples: {len(train_indices) + len(test_indices)}")
        
        # 按主题统计
        train_topic_counts = Counter([topics[i] for i in train_indices])
        test_topic_counts = Counter([topics[i] for i in test_indices])
        
        print(f"\nTrain topic distribution (top 10):")
        for i, (topic, count) in enumerate(train_topic_counts.most_common(10)):
            print(f"  {i+1:2d}. {topic}: {count} samples")
        
        print(f"\nTest topic distribution:")
        for i, (topic, count) in enumerate(test_topic_counts.most_common()):
            print(f"  {i+1:2d}. {topic}: {count} samples")
        
        # 准备训练数据
        self.train_data = {
            'data': {m: data_dict[m][train_indices] for m in MMLU_TRAIN_MODELS},
            'questions': [questions[i] for i in train_indices],
            'labels': [labels[i] for i in train_indices],
            'topics': [topics[i] for i in train_indices],
            'indices': train_indices,
            'subject_counts': train_topic_counts
        }
        
        # 准备测试数据（按主题组织）
        test_data_by_subject = {}
        for subject in self.test_subjects:
            subject_indices = [i for i in test_indices if topics[i] == subject]
            if subject_indices:
                test_data_by_subject[subject] = {
                    'data': {m: data_dict[m][subject_indices] for m in MMLU_TRAIN_MODELS},
                    'questions': [questions[i] for i in subject_indices],
                    'labels': [labels[i] for i in subject_indices],
                    'indices': subject_indices,
                    'count': len(subject_indices)
                }
        
        self.test_data = {
            'all_data': {m: data_dict[m][test_indices] for m in MMLU_TRAIN_MODELS},
            'all_questions': [questions[i] for i in test_indices],
            'all_labels': [labels[i] for i in test_indices],
            'all_topics': [topics[i] for i in test_indices],
            'all_indices': test_indices,  # 修复：添加all_indices
            'by_subject': test_data_by_subject,
            'subject_counts': test_topic_counts
        }
        
        # 保存划分
        with open(train_path, 'wb') as f:
            pkl.dump(self.train_data, f)
        with open(test_path, 'wb') as f:
            pkl.dump(self.test_data, f)
        
        print(f"\n✓ Data saved to {self.split_dir}")
        
        # 保存划分信息为CSV便于查看
        self.save_split_info(questions, labels, topics, train_indices, test_indices)
    
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
            f"mmlu_topk_{SELECTED_EMBEDDING}_embeddings.pkl"
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
            test_questions = self.test_data['all_questions']
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
        
        # 为每个测试领域创建embedding映射
        self.test_embeddings_by_subject = {}
        for subject, data in self.test_data['by_subject'].items():
            indices = data['indices']
            
            # 获取这些索引在完整测试集中的位置
            full_indices = []
            for idx in indices:
                # 找到在完整测试集中的位置（使用all_indices）
                try:
                    # 确保 test_data 有 all_indices
                    if 'all_indices' not in self.test_data:
                        # 如果没有，从 by_subject 重建
                        all_indices_list = []
                        for s, d in self.test_data['by_subject'].items():
                            if 'indices' in d:
                                all_indices_list.extend(d['indices'])
                        # 去重并排序
                        self.test_data['all_indices'] = list(sorted(set(all_indices_list)))
                        print(f"  Reconstructed all_indices with {len(self.test_data['all_indices'])} indices")
                    
                    full_idx = self.test_data['all_indices'].index(idx)
                    full_indices.append(full_idx)
                except ValueError:
                    print(f"  Warning: Index {idx} not found in all_indices for subject '{subject}'")
                except KeyError:
                    print(f"  Error: 'all_indices' not found in test_data")
                    # 创建临时的映射
                    full_indices = list(range(len(indices)))
                    break
            
            if full_indices:
                self.test_embeddings_by_subject[subject] = self.test_embeddings[full_indices]
                print(f"  Subject '{subject}': {len(full_indices)} samples")
            else:
                self.test_embeddings_by_subject[subject] = torch.tensor([])
                print(f"  Subject '{subject}': 0 samples (no embeddings)")
    
    def prepare_all_data(self):
        """准备所有数据（划分 + embedding）"""
        print("\n" + "="*80)
        print("DATA PREPARATION PHASE")
        print("="*80)
        
        # 1. 划分数据
        self.split_data_by_subject()
        
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
    
    def select_top_k_models(self, gate_scores, k=4):
        """选择Top-K个模型"""
        # 按分数降序排序
        sorted_models = sorted(gate_scores.items(), key=lambda x: x[1], reverse=True)
        # 选择前k个
        top_k_models = [model for model, _ in sorted_models[:k]]
        return top_k_models
    
    def predict_mmlu_ensemble(self, model_predictions, selected_models, idx):
        """MMLU集成预测（软投票）"""
        total_logits = np.zeros(4)
        for model_name in selected_models:
            total_logits += np.exp(model_predictions[model_name][idx])
        
        # 选择概率最高的选项
        pred_class = ['A', 'B', 'C', 'D'][np.argmax(total_logits)]
        return pred_class
    
    def evaluate_subject(self, seed, subject):
        """评估特定seed和特定测试领域"""
        # 加载门控模型
        gate_models = self.load_gate_models(seed)
        if not gate_models:
            print(f"No gate models found for seed {seed}")
            return None
        
        # 获取该领域的数据
        if subject not in self.test_data['by_subject']:
            print(f"Warning: Subject '{subject}' not in test data")
            return None
        
        subject_data = self.test_data['by_subject'][subject]
        
        # 检查是否有该领域的embedding
        if subject not in self.test_embeddings_by_subject:
            print(f"Warning: No embeddings found for subject '{subject}'")
            return None
        
        subject_embeddings = self.test_embeddings_by_subject[subject]
        
        if len(subject_embeddings) == 0:
            print(f"Warning: Empty embeddings for subject '{subject}'")
            return None
        
        test_questions = subject_data['questions']
        test_labels = subject_data['labels']
        model_predictions = subject_data['data']
        
        total_samples = len(test_questions)
        if total_samples == 0:
            print(f"Warning: No samples for subject '{subject}'")
            return None
        
        correct_ensemble = 0
        correct_random = 0
        
        # 统计每个模型的激活次数（在top4中被选中的次数）
        activation_counts = {model: 0 for model in MMLU_TRAIN_MODELS}
        
        # 设置随机种子用于random baseline
        random.seed(seed)
        np.random.seed(seed)
        
        for idx in range(total_samples):
            # 获取当前问题的embedding
            if idx < len(subject_embeddings):
                embedding = subject_embeddings[idx]
            else:
                print(f"Warning: Embedding index {idx} out of range for subject '{subject}'")
                continue
            
            # 获取门控分数
            gate_scores = self.get_gate_scores(gate_models, embedding)
            
            # 1. Top-K策略（选择分数最高的4个模型）
            selected_models_topk = self.select_top_k_models(gate_scores, k=self.top_k)
            
            # 记录激活次数
            for model in selected_models_topk:
                activation_counts[model] += 1
            
            # Ensemble预测
            ensemble_pred = self.predict_mmlu_ensemble(model_predictions, selected_models_topk, idx)
            if ensemble_pred == test_labels[idx]:
                correct_ensemble += 1
            
            # 2. Random baseline（随机选择4个模型）
            random_models = np.random.choice(MMLU_TRAIN_MODELS, self.top_k, replace=False)
            random_pred = self.predict_mmlu_ensemble(model_predictions, random_models, idx)
            if random_pred == test_labels[idx]:
                correct_random += 1
        
        # 计算准确率
        accuracy_ensemble = correct_ensemble / total_samples if total_samples > 0 else 0
        accuracy_random = correct_random / total_samples if total_samples > 0 else 0
        
        # 计算激活频率（相对于该领域总问题数）
        activation_freq = {
            model: count / total_samples if total_samples > 0 else 0
            for model, count in activation_counts.items()
        }
        
        print(f"    Results: Ensemble={accuracy_ensemble:.4f}, Random={accuracy_random:.4f}, Samples={total_samples}")
        
        return {
            'accuracy_ensemble': accuracy_ensemble,
            'accuracy_random': accuracy_random,
            'activation_freq': activation_freq,
            'total_samples': total_samples,
            'activation_counts': activation_counts
        }
    
    def evaluate_all_subjects(self):
        """评估所有测试领域"""
        print("\n" + "="*80)
        print("EVALUATION PHASE - Top-K Strategy")
        print("="*80)
        
        results = {}
        
        # 对每个测试领域进行评估
        for subject in self.test_subjects:
            print(f"\nEvaluating subject: {subject}")
            
            subject_key = subject
            results[subject_key] = {
                'seed_results': {},
                'acc_ensemble_mean': 0,
                'acc_ensemble_std': 0,
                'acc_random_mean': 0,
                'acc_random_std': 0,
                'activation_freqs': {model: [] for model in MMLU_TRAIN_MODELS},
                'sample_counts': []
            }
            
            # 对每个seed进行评估
            seed_acc_ensemble = []
            seed_acc_random = []
            
            for seed in self.seeds:
                result = self.evaluate_subject(seed, subject)
                
                if result is not None:
                    results[subject_key]['seed_results'][seed] = result
                    
                    seed_acc_ensemble.append(result['accuracy_ensemble'])
                    seed_acc_random.append(result['accuracy_random'])
                    results[subject_key]['sample_counts'].append(result['total_samples'])
                    
                    # 收集激活频率
                    for model, freq in result['activation_freq'].items():
                        results[subject_key]['activation_freqs'][model].append(freq)
                else:
                    print(f"  Seed {seed}: Failed to evaluate")
            
            # 计算统计量
            if seed_acc_ensemble:
                results[subject_key]['acc_ensemble_mean'] = np.mean(seed_acc_ensemble)
                results[subject_key]['acc_ensemble_std'] = np.std(seed_acc_ensemble, ddof=1)
                
                results[subject_key]['acc_random_mean'] = np.mean(seed_acc_random)
                results[subject_key]['acc_random_std'] = np.std(seed_acc_random, ddof=1)
        
        # 计算总体统计（所有测试领域的平均值）
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS (averaged across all test subjects)")
        print(f"{'='*80}")
        
        all_ensemble_means = []
        all_ensemble_stds = []
        all_random_means = []
        all_random_stds = []
        
        for subject in self.test_subjects:
            if subject in results:
                all_ensemble_means.append(results[subject]['acc_ensemble_mean'])
                all_ensemble_stds.append(results[subject]['acc_ensemble_std'])
                all_random_means.append(results[subject]['acc_random_mean'])
                all_random_stds.append(results[subject]['acc_random_std'])
        
        if all_ensemble_means:
            overall_results = {
                'acc_ensemble_mean': np.mean(all_ensemble_means),
                'acc_ensemble_std': np.mean(all_ensemble_stds),  # 平均标准差
                'acc_random_mean': np.mean(all_random_means),
                'acc_random_std': np.mean(all_random_stds)
            }
            results['OVERALL'] = overall_results
            
            print(f"Ensemble Accuracy: {overall_results['acc_ensemble_mean']:.4f} ± {overall_results['acc_ensemble_std']:.4f}")
            print(f"Random Baseline:   {overall_results['acc_random_mean']:.4f} ± {overall_results['acc_random_std']:.4f}")
        
        return results
    
    def generate_csv_report(self, results):
        """生成CSV报告"""
        csv_data = []
        
        # 为每个测试领域创建一行
        for subject in self.test_subjects:
            if subject in results:
                result = results[subject]
                
                # 获取样本数量（使用第一个seed的样本数）
                sample_count = result.get('sample_counts', [0])[0] if result.get('sample_counts') else 0
                
                # 基本信息
                row = {
                    'subject': subject,
                    'samples': sample_count,
                    'acc_ensemble_mean': f"{result['acc_ensemble_mean']:.4f}",
                    'acc_ensemble_std': f"{result['acc_ensemble_std']:.4f}",
                    'acc_random_mean': f"{result['acc_random_mean']:.4f}",
                    'acc_random_std': f"{result['acc_random_std']:.4f}",
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
        
        # 添加总体统计行
        if 'OVERALL' in results:
            overall = results['OVERALL']
            row = {
                'subject': 'OVERALL',
                'samples': 'N/A',
                'acc_ensemble_mean': f"{overall['acc_ensemble_mean']:.4f}",
                'acc_ensemble_std': f"{overall['acc_ensemble_std']:.4f}",
                'acc_random_mean': f"{overall['acc_random_mean']:.4f}",
                'acc_random_std': f"{overall['acc_random_std']:.4f}",
                'ensemble_formatted': f"{overall['acc_ensemble_mean']:.4f} ± {overall['acc_ensemble_std']:.4f}",
                'random_formatted': f"{overall['acc_random_mean']:.4f} ± {overall['acc_random_std']:.4f}"
            }
            
            # 为模型频率列添加占位符
            for model in MMLU_TRAIN_MODELS:
                row[f'{model}_freq'] = "N/A"
            
            csv_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(csv_data)
        
        # 重新排序列
        columns = ['subject', 'samples', 'acc_ensemble_mean', 'acc_ensemble_std', 
                  'acc_random_mean', 'acc_random_std', 'ensemble_formatted', 'random_formatted']
        for model in MMLU_TRAIN_MODELS:
            columns.append(f'{model}_freq')
        
        df = df[columns]
        
        # 保存CSV
        csv_file = os.path.join(self.results_dir, "mmlu_topk_results.csv")
        df.to_csv(csv_file, index=False)
        
        return csv_file, df
    
    def save_detailed_results(self, results):
        """保存详细结果到JSON文件"""
        output_file = os.path.join(self.results_dir, "mmlu_topk_detailed_results.json")
        
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
        
        print(f"\nResults by Subject (Top-K Strategy):")
        print(df[['subject', 'samples', 'ensemble_formatted', 'random_formatted']].to_string(index=False))
        
        # 找出最佳和最差的领域
        if len(self.test_subjects) > 1:
            subject_results = df[df['subject'] != 'OVERALL'].copy()
            subject_results['acc_ensemble_mean'] = subject_results['acc_ensemble_mean'].astype(float)
            
            best_row = subject_results.loc[subject_results['acc_ensemble_mean'].idxmax()]
            worst_row = subject_results.loc[subject_results['acc_ensemble_mean'].idxmin()]
            
            print(f"\nBest subject: {best_row['subject']} (Accuracy: {best_row['ensemble_formatted']})")
            print(f"Worst subject: {worst_row['subject']} (Accuracy: {worst_row['ensemble_formatted']})")
        
        # 总体结果
        if 'OVERALL' in results:
            overall = results['OVERALL']
            print(f"\nOverall Results (averaged across all test subjects):")
            print(f"  Ensemble Accuracy: {overall['acc_ensemble_mean']:.4f} ± {overall['acc_ensemble_std']:.4f}")
            print(f"  Random Baseline:   {overall['acc_random_mean']:.4f} ± {overall['acc_random_std']:.4f}")
            
            # 计算相对提升
            if overall['acc_random_mean'] > 0:
                relative_improvement = (overall['acc_ensemble_mean'] - overall['acc_random_mean']) / overall['acc_random_mean'] * 100
                print(f"  Relative Improvement: {relative_improvement:+.2f}%")
        
        # 模型激活频率分析（总体平均）
        print(f"\nOverall Model Activation Frequencies (averaged across all test subjects and seeds):")
        
        # 计算所有测试领域的平均激活频率
        activation_means_overall = {model: [] for model in MMLU_TRAIN_MODELS}
        
        for subject in self.test_subjects:
            if subject in results:
                for model in MMLU_TRAIN_MODELS:
                    freqs = results[subject]['activation_freqs'][model]
                    if freqs:
                        activation_means_overall[model].extend(freqs)
        
        # 计算每个模型的总体平均激活频率
        overall_means = {}
        for model, freqs in activation_means_overall.items():
            if freqs:
                overall_means[model] = np.mean(freqs)
        
        # 按激活频率排序
        sorted_models = sorted(overall_means.items(), key=lambda x: x[1], reverse=True)
        for model, freq in sorted_models:
            print(f"  {model:<30}: {freq:.4f}")
    
    def run_full_experiment(self):
        """运行完整实验"""
        print("="*80)
        print("MMLU TOP-K EXPERIMENT - Complete Pipeline")
        print("="*80)
        print(f"Training Subjects: {len(self.train_subjects)}")
        print(f"Test Subjects: {len(self.test_subjects)}")
        print(f"Seeds: {self.seeds}")
        print(f"Top-K: {self.top_k}")
        print(f"Models: {len(MMLU_TRAIN_MODELS)}")
        print(f"Gate type: {GATE_TYPE}")
        print(f"Embedding: {SELECTED_EMBEDDING}")
        print("="*80)
        
        # 阶段1: 数据准备
        self.prepare_all_data()
        
        # 阶段2: 训练
        self.train_all_seeds()
        
        # 阶段3: 评估
        results = self.evaluate_all_subjects()
        
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


def main():
    parser = argparse.ArgumentParser(description='MMLU Top-K Experiment - Subject-based split with Top4 selection')
    parser.add_argument('--mode', default='full', choices=['full', 'data_only', 'train_only', 'eval_only'],
                       help='运行模式: full(完整), data_only(只准备数据), train_only(只训练), eval_only(只评估)')
    parser.add_argument('--skip_data', action='store_true',
                       help='跳过数据准备阶段 (使用已有的数据)')
    parser.add_argument('--skip_train', action='store_true',
                       help='跳过训练阶段 (使用已有的模型)')
    
    args = parser.parse_args()
    
    # 创建实验对象
    experiment = MMLUTopKExperiment()
    
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
        
        results = experiment.evaluate_all_subjects()
        csv_file, df = experiment.generate_csv_report(results)
        experiment.print_summary(results, df)
    
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()