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
from data_loader import load_gsm8k_raw_predictions, compute_gsm8k_accuracy
from data_split import prepare_gsm8k_data
from embedding_manager import get_embedding_manager
from gate_model import GateNetwork
from trainer import GateTrainer, GSM8KEmbeddingDataset, collate_fn_gsm8k

class GSM8KMultiSeedExperiment:
    """GSM8K多seed多阈值实验"""
    
    def __init__(self, base_seed=42, num_runs=3):
        self.base_seed = base_seed
        self.num_runs = num_runs
        self.seeds = [42, 123, 0]  # 固定的3个seed
        self.thresholds = [i/10 for i in range(1, 10)]  # 0.1-0.9
        self.gate_gsm8k_dir = "gate_gsm8k"  # 专门存放GSM8K的门控模型
        
        # 创建必要的目录
        os.makedirs(self.gate_gsm8k_dir, exist_ok=True)
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 全局embedding manager
        self.emb_manager = get_embedding_manager(SELECTED_EMBEDDING, self.device)
        
    def prepare_data_once(self):
        """只准备一次数据，确保所有实验使用相同的数据划分"""
        print("="*80)
        print("Preparing GSM8K data (one-time only)")
        print("="*80)
        
        # 检查数据是否已经存在
        split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
        train_path = os.path.join(split_dir, "gsm8k_train.pkl")
        test_path = os.path.join(split_dir, "gsm8k_test.pkl")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("Data already prepared, loading from cache...")
        else:
            print("Preparing data for the first time...")
            prepare_gsm8k_data()
        
        # 加载数据
        with open(train_path, 'rb') as f:
            self.train_data = pkl.load(f)
        with open(test_path, 'rb') as f:
            self.test_data = pkl.load(f)
        
        print(f"Train data: {len(self.train_data['questions'])} samples")
        print(f"Test data: {len(self.test_data['questions'])} samples")
        print(f"Predictions shape: {self.test_data['raw_predictions'].shape}")
        
        # 预计算embedding
        print("\nPrecomputing embeddings...")
        self.train_embeddings, self.test_embeddings = self.emb_manager.precompute_embeddings(
            "gsm8k", force_recompute=False
        )
        
        self.embedding_dim = self.emb_manager.get_encoder_dim()
        print(f"Embedding dimension: {self.embedding_dim}")
        
    def load_gate_models(self, seed):
        """加载指定seed的门控模型"""
        gate_models = {}
        model_list = GSM8K_TRAIN_MODELS
        
        for i, model_name in enumerate(model_list):
            model_path = os.path.join(
                self.gate_gsm8k_dir,
                f"{model_name}_gsm8k_{GATE_TYPE}_seed{seed}.pt"
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
        raw_predictions = self.test_data['raw_predictions']  # (samples, models, runs)
        
        total_samples = len(test_questions)
        correct_ensemble = 0
        correct_random = 0
        
        # 统计每个模型的激活次数（被选中的次数）
        activation_counts = {model: 0 for model in GSM8K_TRAIN_MODELS}
        
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
            
            # Ensemble预测（多数投票）
            ensemble_pred = self._predict_ensemble(idx, selected_models, raw_predictions)
            if self._check_correct(ensemble_pred, test_labels[idx]):
                correct_ensemble += 1
            
            # 2. Random baseline（随机选择相同数量的模型）
            k = len(selected_models)
            random_models = np.random.choice(GSM8K_TRAIN_MODELS, k, replace=False)
            random_pred = self._predict_ensemble(idx, random_models, raw_predictions)
            if self._check_correct(random_pred, test_labels[idx]):
                correct_random += 1
        
        # 计算准确率
        accuracy_ensemble = correct_ensemble / total_samples if total_samples > 0 else 0
        accuracy_random = correct_random / total_samples if total_samples > 0 else 0
        
        # 计算激活频率（被选中次数 / 总问题数）
        activation_freq = {
            model: count / total_samples if total_samples > 0 else 0
            for model, count in activation_counts.items()
        }
        
        # 计算平均每个问题选择的模型数量
        avg_selected_models = np.mean(selected_models_counts) if selected_models_counts else 0
        std_selected_models = np.std(selected_models_counts, ddof=1) if selected_models_counts else 0
        
        return {
            'accuracy_ensemble': accuracy_ensemble,
            'accuracy_random': accuracy_random,
            'activation_freq': activation_freq,
            'total_samples': total_samples,
            'avg_selected_models': avg_selected_models,
            'std_selected_models': std_selected_models,
            'selected_models_counts': selected_models_counts
        }
    
    def _predict_ensemble(self, idx, selected_models, raw_predictions):
        """集成预测（多数投票）"""
        model_indices = [GSM8K_TRAIN_MODELS.index(m) for m in selected_models]
        
        answers = []
        for model_idx in model_indices:
            model_runs = raw_predictions[idx, model_idx, :]
            valid_answers = [a for a in model_runs if not np.isnan(a)]
            
            if valid_answers:
                # 统计每个答案的出现频率
                answer_counts = Counter([str(a) for a in valid_answers])
                # 选择最常见的答案
                most_common = float(answer_counts.most_common(1)[0][0])
                answers.append(most_common)
        
        if not answers:
            return np.nan
        
        # 选择所有被选中模型中最常见的答案
        final_counts = Counter(answers)
        return float(final_counts.most_common(1)[0][0])
    
    def _check_correct(self, pred, label):
        """检查预测是否正确"""
        if np.isnan(pred) or np.isnan(label):
            return False
        try:
            return abs(float(pred) - float(label)) < 1e-4
        except:
            return False
    
    def check_models_exist(self):
        """检查所有需要的模型是否存在"""
        all_exist = True
        for seed in self.seeds:
            for model_name in GSM8K_TRAIN_MODELS:
                model_path = os.path.join(
                    self.gate_gsm8k_dir,
                    f"{model_name}_gsm8k_{GATE_TYPE}_seed{seed}.pt"
                )
                if not os.path.exists(model_path):
                    print(f"Missing model: {model_path}")
                    all_exist = False
        
        return all_exist
    
    def run_evaluation_only(self):
        """只运行评估，跳过训练"""
        print("="*80)
        print("GSM8K Multi-Seed Multi-Threshold Evaluation (Skip Training)")
        print("="*80)
        
        # 1. 准备数据
        self.prepare_data_once()
        
        # 2. 检查模型是否存在
        if not self.check_models_exist():
            print("\n❌ Some models are missing! Please train models first.")
            return None
        
        # 3. 为每个阈值进行评估
        print("\n" + "="*80)
        print("Evaluating for all thresholds and seeds")
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
                'avg_selected_models_mean': 0,  # 平均激活模型数量
                'avg_selected_models_std': 0,   # 激活模型数量标准差
                'selected_models_counts_all': [],  # 收集所有seed的模型数量数据
                'activation_freqs': {model: [] for model in GSM8K_TRAIN_MODELS}
            }
            
            # 对每个seed进行评估
            seed_acc_ensemble = []
            seed_acc_random = []
            seed_avg_selected = []  # 每个seed的平均激活模型数量
            
            for seed in self.seeds:
                print(f"\nEvaluating: Threshold={threshold:.1f}, Seed={seed}")
                result = self.evaluate_threshold(seed, threshold)
                
                if result is not None:
                    results[threshold_key]['seed_results'][seed] = result
                    
                    seed_acc_ensemble.append(result['accuracy_ensemble'])
                    seed_acc_random.append(result['accuracy_random'])
                    seed_avg_selected.append(result['avg_selected_models'])
                    
                    # 收集激活频率
                    for model, freq in result['activation_freq'].items():
                        results[threshold_key]['activation_freqs'][model].append(freq)
                    
                    # 收集所有问题的模型数量数据（用于整体统计）
                    if 'selected_models_counts' in result:
                        results[threshold_key]['selected_models_counts_all'].extend(
                            result['selected_models_counts']
                        )
            
            # 计算统计量
            if seed_acc_ensemble:
                results[threshold_key]['acc_ensemble_mean'] = np.mean(seed_acc_ensemble)
                results[threshold_key]['acc_ensemble_std'] = np.std(seed_acc_ensemble, ddof=1)
                
                results[threshold_key]['acc_random_mean'] = np.mean(seed_acc_random)
                results[threshold_key]['acc_random_std'] = np.std(seed_acc_random, ddof=1)
                
                # 计算平均激活模型数量的统计
                results[threshold_key]['avg_selected_models_mean'] = np.mean(seed_avg_selected)
                results[threshold_key]['avg_selected_models_std'] = np.std(seed_avg_selected, ddof=1)
        
        # 4. 保存详细结果
        self.save_detailed_results(results)
        
        # 5. 生成CSV报告
        csv_file, df = self.generate_csv_report(results)
        
        # 6. 打印摘要
        self.print_summary(results, df)
        
        return results
    
    def save_detailed_results(self, results):
        """保存详细结果到JSON文件"""
        output_file = os.path.join(RESULT_DIR, "gsm8k_detailed_results.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
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
        
        print(f"\n✓ Detailed results saved to {output_file}")
    
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
            for model in GSM8K_TRAIN_MODELS:
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
        for model in GSM8K_TRAIN_MODELS:
            columns.append(f'{model}_freq')
        
        df = df[columns]
        
        # 保存CSV
        csv_file = "gsm8k_results.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n✓ CSV report saved to {csv_file}")
        
        return csv_file, df
    
    def print_summary(self, results, df):
        """打印结果摘要"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print(f"\nResults Summary:")
        summary_columns = ['threshold', 'ensemble_formatted', 'random_formatted', 'avg_selected_models']
        print(df[summary_columns].to_string(index=False))
        
        # 找出最佳阈值
        best_row = df.loc[df['acc_ensemble_mean'].astype(float).idxmax()]
        print(f"\nBest threshold: {best_row['threshold']}")
        print(f"Best ensemble accuracy: {best_row['ensemble_formatted']}")
        print(f"Corresponding random baseline: {best_row['random_formatted']}")
        print(f"Average models selected per question: {best_row['avg_selected_models']}")
        
        # 打印激活频率最高的3个模型
        print(f"\nModel Activation Frequencies at best threshold ({best_row['threshold']}):")
        best_threshold = best_row['threshold']
        threshold_key = f"threshold_{best_threshold:.1f}"
        result = results[threshold_key]
        
        activation_means = {}
        for model in GSM8K_TRAIN_MODELS:
            freqs = result['activation_freqs'][model]
            if freqs:
                activation_means[model] = np.mean(freqs)
        
        top_models = sorted(activation_means.items(), key=lambda x: x[1], reverse=True)[:5]
        for model, freq in top_models:
            print(f"  {model:<30}: {freq:.4f}")


def main():
    parser = argparse.ArgumentParser(description="GSM8K Multi-Seed Multi-Threshold Experiment")
    parser.add_argument("--mode", default="eval_only", choices=["full", "eval_only"],
                       help="运行模式: full(完整), eval_only(只评估)")
    parser.add_argument("--threshold", type=float, default=None,
                       help="只评估特定阈值 (0.1-0.9)")
    args = parser.parse_args()
    
    # 创建实验对象
    experiment = GSM8KMultiSeedExperiment()
    
    if args.mode == "full":
        print("完整模式需要训练模型，但原训练代码已移除。")
        print("请使用 --mode eval_only 进行直接评估。")
        return
    
    elif args.mode == "eval_only":
        # 只进行评估
        if args.threshold is not None:
            # 只评估特定阈值
            if args.threshold in experiment.thresholds:
                print(f"\n只评估阈值: {args.threshold}")
                # 准备数据
                experiment.prepare_data_once()
                
                # 检查模型是否存在
                if not experiment.check_models_exist():
                    print("\n❌ Some models are missing! Please ensure models are in gate_gsm8k/")
                    return
                
                # 只评估这个阈值
                threshold_key = f"threshold_{args.threshold:.1f}"
                results = {threshold_key: {
                    'seed_results': {},
                    'acc_ensemble_mean': 0,
                    'acc_ensemble_std': 0,
                    'acc_random_mean': 0,
                    'acc_random_std': 0,
                    'avg_selected_models_mean': 0,
                    'avg_selected_models_std': 0,
                    'selected_models_counts_all': [],
                    'activation_freqs': {model: [] for model in GSM8K_TRAIN_MODELS}
                }}
                
                seed_acc_ensemble = []
                seed_acc_random = []
                seed_avg_selected = []
                
                for seed in experiment.seeds:
                    print(f"\nEvaluating: Threshold={args.threshold:.1f}, Seed={seed}")
                    result = experiment.evaluate_threshold(seed, args.threshold)
                    
                    if result is not None:
                        results[threshold_key]['seed_results'][seed] = result
                        seed_acc_ensemble.append(result['accuracy_ensemble'])
                        seed_acc_random.append(result['accuracy_random'])
                        seed_avg_selected.append(result['avg_selected_models'])
                        
                        for model, freq in result['activation_freq'].items():
                            results[threshold_key]['activation_freqs'][model].append(freq)
                        
                        if 'selected_models_counts' in result:
                            results[threshold_key]['selected_models_counts_all'].extend(
                                result['selected_models_counts']
                            )
                
                if seed_acc_ensemble:
                    results[threshold_key]['acc_ensemble_mean'] = np.mean(seed_acc_ensemble)
                    results[threshold_key]['acc_ensemble_std'] = np.std(seed_acc_ensemble, ddof=1)
                    results[threshold_key]['acc_random_mean'] = np.mean(seed_acc_random)
                    results[threshold_key]['acc_random_std'] = np.std(seed_acc_random, ddof=1)
                    results[threshold_key]['avg_selected_models_mean'] = np.mean(seed_avg_selected)
                    results[threshold_key]['avg_selected_models_std'] = np.std(seed_avg_selected, ddof=1)
                
                # 生成临时的DataFrame用于打印
                temp_data = [{
                    'threshold': args.threshold,
                    'acc_ensemble_mean': f"{results[threshold_key]['acc_ensemble_mean']:.4f}",
                    'acc_ensemble_std': f"{results[threshold_key]['acc_ensemble_std']:.4f}",
                    'acc_random_mean': f"{results[threshold_key]['acc_random_mean']:.4f}",
                    'acc_random_std': f"{results[threshold_key]['acc_random_std']:.4f}",
                    'avg_selected_models': f"{results[threshold_key]['avg_selected_models_mean']:.2f} ± {results[threshold_key]['avg_selected_models_std']:.2f}",
                    'ensemble_formatted': f"{results[threshold_key]['acc_ensemble_mean']:.4f} ± {results[threshold_key]['acc_ensemble_std']:.4f}",
                    'random_formatted': f"{results[threshold_key]['acc_random_mean']:.4f} ± {results[threshold_key]['acc_random_std']:.4f}"
                }]
                
                temp_df = pd.DataFrame(temp_data)
                print("\nResults:")
                print(temp_df[['threshold', 'ensemble_formatted', 'random_formatted', 'avg_selected_models']].to_string(index=False))
                
                # 打印激活频率
                print(f"\nModel Activation Frequencies:")
                activation_means = {}
                for model in GSM8K_TRAIN_MODELS:
                    freqs = results[threshold_key]['activation_freqs'][model]
                    if freqs:
                        activation_means[model] = np.mean(freqs)
                
                top_models = sorted(activation_means.items(), key=lambda x: x[1], reverse=True)[:5]
                for model, freq in top_models:
                    print(f"  {model:<30}: {freq:.4f}")
                
            else:
                print(f"阈值必须是: {experiment.thresholds}")
        else:
            # 评估所有阈值
            results = experiment.run_evaluation_only()
            if results:
                # 已经打印了完整摘要
                pass


if __name__ == "__main__":
    main()