import os
import json
import pickle as pkl
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import argparse
import pandas as pd
from datetime import datetime

# 导入项目模块
from config import *
from data_loader import load_mmlu_data
from gate_model import GateNetwork
from trainer import GateTrainer, MMLUDataset, collate_fn_mmlu
from torch.utils.data import DataLoader, random_split


class SimpleMMLUTester:
    """
    简化的MMLU测试器 - 直接使用已有的数据集
    """
    
    def __init__(self, embedding_key="bert", k=3, device='cuda'):
        """
        初始化测试器
        
        Args:
            embedding_key: 使用的embedding模型
            k: top-k中k的值
            device: 计算设备
        """
        self.embedding_key = embedding_key
        self.k = k
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 存储模型
        self.gate_models = {}
        self.model_accuracies = {}
        
        print(f"Simple MMLU Tester initialized")
        print(f"Embedding: {embedding_key}, k={k}, device={device}")
    
    def load_existing_split(self, split_dir):
        """
        加载已有的划分数据
        
        Args:
            split_dir: 划分数据目录（包含domain_train.pkl和domain_test.pkl）
        """
        print(f"\nLoading existing split from: {split_dir}")
        
        # 加载训练数据
        train_path = os.path.join(split_dir, "domain_train.pkl")
        if not os.path.exists(train_path):
            # 尝试相对路径
            train_path = os.path.join(CUR_DIR, DATA_DIR, "splits", split_dir, "domain_train.pkl")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        with open(train_path, "rb") as f:
            self.train_data = pkl.load(f)
        
        # 加载测试数据
        test_path = os.path.join(split_dir, "domain_test.pkl")
        if not os.path.exists(test_path):
            test_path = os.path.join(CUR_DIR, DATA_DIR, "splits", split_dir, "domain_test.pkl")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        with open(test_path, "rb") as f:
            self.test_data_by_subject = pkl.load(f)
        
        # 显示数据集信息
        train_samples = len(self.train_data['questions'])
        train_subjects = len(self.train_data.get('subjects', []))
        train_groups = len(self.train_data.get('groups', []))
        
        test_samples = sum(len(data['questions']) for data in self.test_data_by_subject.values())
        test_subjects = len(self.test_data_by_subject)
        
        print(f"✓ Dataset loaded:")
        print(f"  Training: {train_samples} samples, {train_subjects} subjects, {train_groups} groups")
        print(f"  Testing: {test_samples} samples, {test_subjects} subjects")
        
        return self.train_data, self.test_data_by_subject
    
    def train_gate_models(self):
        """训练所有门控模型"""
        print(f"\n{'='*60}")
        print("TRAINING GATE MODELS")
        print(f"{'='*60}")
        
        for model_name in MMLU_TRAIN_MODELS:
            print(f"\nTraining Gate for: {model_name}")
            print("-" * 40)
            
            # 检查模型是否有数据
            if model_name not in self.train_data['data']:
                print(f"  Skipping {model_name} - no data available")
                continue
            
            # 创建数据集
            dataset = MMLUDataset(
                self.train_data['data'], 
                self.train_data['questions'], 
                self.train_data['labels'], 
                model_name
            )
            
            # 划分训练集和验证集
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=True,
                collate_fn=collate_fn_mmlu
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=TRAIN_CONFIG["batch_size"],
                shuffle=False,
                collate_fn=collate_fn_mmlu
            )
            
            # 训练模型
            trainer = GateTrainer("mmlu", model_name, self.embedding_key)
            trainer.device = self.device
            trainer.gate_model = trainer.gate_model.to(self.device)
            
            trainer.train(train_loader, val_loader)
            
            # 保存模型
            self.gate_models[model_name] = trainer.gate_model
            
            print(f"  ✓ Gate for {model_name} trained")
    
    def load_pretrained_gates(self, gate_dir=GATE_DIR):
        """加载预训练的门控模型"""
        print(f"\nLoading pretrained gate models from: {gate_dir}")
        
        for model_name in MMLU_TRAIN_MODELS:
            model_path = os.path.join(gate_dir, f"{model_name}_mmlu.pt")
            if os.path.exists(model_path):
                gate_model = GateNetwork(
                    embedding_key=self.embedding_key,
                    hidden_dim=TRAIN_CONFIG["hidden_dim"],
                    dropout=TRAIN_CONFIG["dropout"]
                ).to(self.device)
                
                checkpoint = torch.load(model_path, map_location=self.device)
                gate_model.load_state_dict(checkpoint['model_state_dict'])
                gate_model.eval()
                
                self.gate_models[model_name] = gate_model
                print(f"  ✓ Loaded gate for {model_name}")
            else:
                print(f"  ✗ Gate model not found for {model_name}")
    
    def evaluate_single_models(self):
        """评估所有单模型在各个领域的表现"""
        print(f"\n{'='*60}")
        print("SINGLE MODEL EVALUATION")
        print(f"{'='*60}")
        
        model_accuracies = defaultdict(lambda: defaultdict(float))
        
        for subject, test_data in self.test_data_by_subject.items():
            questions = test_data['questions']
            labels = test_data['labels']
            model_predictions = test_data['data']
            
            print(f"\nSubject: {subject} ({len(questions)} samples)")
            
            subject_results = []
            
            for model_name in MMLU_TRAIN_MODELS:
                if model_name not in model_predictions:
                    continue
                    
                preds = model_predictions[model_name]
                correct = 0
                
                for i, (pred, label) in enumerate(zip(preds, labels)):
                    pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
                    if pred_class == label:
                        correct += 1
                
                accuracy = correct / len(labels)
                model_accuracies[model_name][subject] = accuracy
                subject_results.append((model_name, accuracy))
            
            # 显示该领域每个模型的准确率（前3名）
            subject_results.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top models:")
            for i, (model_name, acc) in enumerate(subject_results[:3], 1):
                print(f"    {i}. {model_name:25s}: {acc:.4f}")
            
            if len(subject_results) > 3:
                avg_rest = np.mean([acc for _, acc in subject_results[3:]])
                print(f"    ... {len(subject_results) - 3} other models, average: {avg_rest:.4f}")
        
        self.model_accuracies = model_accuracies
        return model_accuracies
    
    def evaluate_top_k_strategy(self):
        """
        评估top-k集成策略 - 按测试领域单独计算指标
        """
        print(f"\n{'='*60}")
        print(f"TOP-K ENSEMBLE EVALUATION (k={self.k})")
        print(f"{'='*60}")
        
        # 存储每个领域的结果
        subject_results = {}
        
        # 存储每个领域的详细数据
        subject_detailed_results = {}
        
        # 记录整体统计
        total_correct_ensemble = 0
        total_correct_baseline = 0
        total_samples = 0
        
        for subject, test_data in tqdm(self.test_data_by_subject.items(), desc="Testing subjects"):
            print(f"\nSubject: {subject} ({len(test_data['questions'])} samples)")
            
            questions = test_data['questions']
            labels = test_data['labels']
            model_predictions = test_data['data']
            
            # 初始化该领域的统计
            subject_correct_ensemble = 0
            subject_correct_baseline = 0
            subject_samples = len(questions)
            
            # 记录该领域每个模型的激活次数
            subject_model_activation_counts = defaultdict(int)
            
            # 存储每个问题的详细预测结果
            detailed_predictions = []
            
            for i in range(subject_samples):
                q_text = questions[i]
                true_label = labels[i]
                
                # 1. 获取门控分数
                gate_scores = {}
                for model_name in MMLU_TRAIN_MODELS:
                    if model_name not in self.gate_models:
                        gate_scores[model_name] = 0.0
                    else:
                        # 单个问题评分
                        with torch.no_grad():
                            score = self.gate_models[model_name]([q_text])
                            gate_scores[model_name] = score.item()
                
                # 2. 选择top-k模型
                sorted_models = sorted(gate_scores.items(), key=lambda x: x[1], reverse=True)
                top_k_models = [model for model, _ in sorted_models[:self.k]]
                
                # 记录激活
                for model in top_k_models:
                    subject_model_activation_counts[model] += 1
                
                # 3. 集成预测（硬投票）
                ensemble_votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                for model_name in top_k_models:
                    if model_name in model_predictions:
                        pred = model_predictions[model_name][i]
                        pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
                        ensemble_votes[pred_class] += 1
                
                ensemble_pred = max(ensemble_votes.items(), key=lambda x: x[1])[0]
                
                # 4. Baseline: 随机选择k个模型
                available_models = [m for m in MMLU_TRAIN_MODELS if m in model_predictions]
                if len(available_models) >= self.k:
                    random_models = np.random.choice(available_models, self.k, replace=False)
                    baseline_votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                    for model_name in random_models:
                        pred = model_predictions[model_name][i]
                        pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
                        baseline_votes[pred_class] += 1
                    
                    baseline_pred = max(baseline_votes.items(), key=lambda x: x[1])[0]
                else:
                    baseline_pred = 'A'  # 如果没有足够的模型，使用默认预测
                
                # 5. 检查正确性
                is_correct_ensemble = ensemble_pred == true_label
                is_correct_baseline = baseline_pred == true_label
                
                if is_correct_ensemble:
                    subject_correct_ensemble += 1
                    total_correct_ensemble += 1
                
                if is_correct_baseline:
                    subject_correct_baseline += 1
                    total_correct_baseline += 1
                
                total_samples += 1
                
                # 存储详细预测结果
                detailed_predictions.append({
                    'question_id': i,
                    'question': q_text[:100] + "..." if len(q_text) > 100 else q_text,
                    'true_label': true_label,
                    'ensemble_prediction': ensemble_pred,
                    'baseline_prediction': baseline_pred,
                    'is_correct_ensemble': is_correct_ensemble,
                    'is_correct_baseline': is_correct_baseline,
                    'top_k_models': top_k_models,
                    'gate_scores': {m: s for m, s in sorted_models}
                })
            
            # 计算该subject的指标
            subject_accuracy_ensemble = subject_correct_ensemble / subject_samples
            subject_accuracy_baseline = subject_correct_baseline / subject_samples
            
            # 计算每个模型的激活概率
            model_activation_probs = {}
            for model_name in MMLU_TRAIN_MODELS:
                activation_prob = subject_model_activation_counts[model_name] / subject_samples if subject_samples > 0 else 0
                model_activation_probs[model_name] = activation_prob
            
            subject_results[subject] = {
                'samples': subject_samples,
                'ensemble_accuracy': subject_accuracy_ensemble,
                'baseline_accuracy': subject_accuracy_baseline,
                'improvement': subject_accuracy_ensemble - subject_accuracy_baseline,
                'model_activation_probs': model_activation_probs
            }
            
            subject_detailed_results[subject] = {
                'summary': subject_results[subject],
                'detailed_predictions': detailed_predictions
            }
            
            print(f"  Ensemble: {subject_accuracy_ensemble:.4f}")
            print(f"  Baseline: {subject_accuracy_baseline:.4f}")
            print(f"  Improvement: {subject_accuracy_ensemble - subject_accuracy_baseline:+.4f}")
            
            # 显示该领域的模型激活率（前5名）
            print(f"  Top model activation probabilities:")
            sorted_activation = sorted(model_activation_probs.items(), key=lambda x: x[1], reverse=True)
            for j, (model_name, prob) in enumerate(sorted_activation[:5], 1):
                print(f"    {j}. {model_name:25s}: {prob:.4f}")
        
        # 计算整体指标
        overall_accuracy_ensemble = total_correct_ensemble / total_samples if total_samples > 0 else 0
        overall_accuracy_baseline = total_correct_baseline / total_samples if total_samples > 0 else 0
        
        # 计算整体模型激活率
        overall_model_activation_counts = defaultdict(int)
        for subject_result in subject_results.values():
            for model_name, prob in subject_result['model_activation_probs'].items():
                overall_model_activation_counts[model_name] += prob * subject_result['samples']
        
        overall_activation_rates = {}
        for model_name in MMLU_TRAIN_MODELS:
            rate = overall_model_activation_counts[model_name] / total_samples if total_samples > 0 else 0
            overall_activation_rates[model_name] = rate
        
        # 显示整体结果
        print(f"\n{'='*60}")
        print("OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Total test samples: {total_samples}")
        print(f"Ensemble accuracy: {overall_accuracy_ensemble:.4f}")
        print(f"Baseline accuracy: {overall_accuracy_baseline:.4f}")
        print(f"Improvement: {overall_accuracy_ensemble - overall_accuracy_baseline:+.4f}")
        
        # 显示模型激活率
        print(f"\nOverall model activation rates:")
        print("-" * 40)
        sorted_models = sorted(overall_activation_rates.items(), key=lambda x: x[1], reverse=True)
        for model_name, rate in sorted_models:
            print(f"  {model_name:25s}: {rate:.4f}")
        
        # 按subject显示结果
        print(f"\nSubject-wise results:")
        print("-" * 80)
        print(f"{'Subject':30s} {'Samples':>8s} {'Ensemble':>10s} {'Baseline':>10s} {'Improvement':>12s}")
        print("-" * 80)
        
        for subject, result in subject_results.items():
            print(f"{subject:30s} {result['samples']:8d} {result['ensemble_accuracy']:10.4f} "
                  f"{result['baseline_accuracy']:10.4f} {result['improvement']:+12.4f}")
        
        return {
            'overall': {
                'ensemble_accuracy': overall_accuracy_ensemble,
                'baseline_accuracy': overall_accuracy_baseline,
                'improvement': overall_accuracy_ensemble - overall_accuracy_baseline,
                'total_samples': total_samples,
                'model_activation_rates': dict(overall_activation_rates)
            },
            'subject_results': subject_results,
            'subject_detailed_results': subject_detailed_results
        }
    
    def save_results(self, results, output_dir, split_dir, embedding_key, k):
        """
        保存详细的评测结果到文件
        
        Args:
            results: 评测结果
            output_dir: 输出目录
            split_dir: 数据集划分目录
            embedding_key: embedding模型
            k: top-k值
        """
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{split_dir.replace('/', '_')}_k{k}_{embedding_key}_{timestamp}"
        exp_dir = os.path.join(output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"\nSaving results to: {exp_dir}")
        
        # 1. 保存整体结果
        overall_file = os.path.join(exp_dir, "overall_results.json")
        with open(overall_file, 'w', encoding='utf-8') as f:
            json.dump(results['overall'], f, indent=2, ensure_ascii=False)
        
        # 2. 保存每个领域的详细结果
        subject_results = results['subject_results']
        subject_summary_file = os.path.join(exp_dir, "subject_summary.csv")
        
        # 创建汇总表格
        summary_data = []
        for subject, result in subject_results.items():
            row = {
                'subject': subject,
                'samples': result['samples'],
                'ensemble_accuracy': result['ensemble_accuracy'],
                'baseline_accuracy': result['baseline_accuracy'],
                'improvement': result['improvement']
            }
            
            # 添加每个模型的激活概率
            for model_name, prob in result['model_activation_probs'].items():
                row[f'activation_prob_{model_name}'] = prob
            
            summary_data.append(row)
        
        # 转换为DataFrame并保存
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(subject_summary_file, index=False, encoding='utf-8-sig')
        
        # 3. 保存每个领域的激活概率矩阵
        activation_matrix = []
        model_names = list(MMLU_TRAIN_MODELS)
        
        for subject, result in subject_results.items():
            row = {'subject': subject}
            for model_name in model_names:
                row[model_name] = result['model_activation_probs'].get(model_name, 0.0)
            activation_matrix.append(row)
        
        activation_df = pd.DataFrame(activation_matrix)
        activation_file = os.path.join(exp_dir, "model_activation_matrix.csv")
        activation_df.to_csv(activation_file, index=False, encoding='utf-8-sig')
        
        # 4. 保存详细预测结果（每个领域一个文件）
        if 'subject_detailed_results' in results:
            detailed_dir = os.path.join(exp_dir, "detailed_predictions")
            os.makedirs(detailed_dir, exist_ok=True)
            
            for subject, data in results['subject_detailed_results'].items():
                # 创建安全的文件名
                safe_subject = subject.replace('/', '_').replace('\\', '_')
                detailed_file = os.path.join(detailed_dir, f"{safe_subject}.json")
                
                # 只保存前100个样本的详细预测，避免文件过大
                max_samples = min(100, len(data['detailed_predictions']))
                detailed_data = {
                    'summary': data['summary'],
                    'sample_predictions': data['detailed_predictions'][:max_samples]
                }
                
                with open(detailed_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        # 5. 保存配置信息
        config_info = {
            'timestamp': timestamp,
            'split_dir': split_dir,
            'embedding_model': embedding_key,
            'k_value': k,
            'device': str(self.device),
            'train_models': MMLU_TRAIN_MODELS,
            'test_subjects': list(subject_results.keys())
        }
        
        config_file = os.path.join(exp_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        
        # 6. 创建README文件
        readme_file = os.path.join(exp_dir, "README.md")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f"# MMLU测试结果\n\n")
            f.write(f"## 实验信息\n")
            f.write(f"- 时间: {timestamp}\n")
            f.write(f"- 数据集划分: {split_dir}\n")
            f.write(f"- Embedding模型: {embedding_key}\n")
            f.write(f"- Top-K值: {k}\n")
            f.write(f"- 测试领域数: {len(subject_results)}\n")
            f.write(f"- 总样本数: {results['overall']['total_samples']}\n\n")
            
            f.write(f"## 整体结果\n")
            f.write(f"- Ensemble准确率: {results['overall']['ensemble_accuracy']:.4f}\n")
            f.write(f"- Baseline准确率: {results['overall']['baseline_accuracy']:.4f}\n")
            f.write(f"- 提升: {results['overall']['improvement']:+.4f}\n\n")
            
            f.write(f"## 文件说明\n")
            f.write(f"- `overall_results.json`: 整体结果\n")
            f.write(f"- `subject_summary.csv`: 各领域结果汇总\n")
            f.write(f"- `model_activation_matrix.csv`: 模型激活概率矩阵\n")
            f.write(f"- `config.json`: 实验配置\n")
            f.write(f"- `detailed_predictions/`: 各领域的详细预测结果\n")
            f.write(f"- `README.md`: 本说明文件\n")
        
        print(f"✓ Results saved in: {exp_dir}")
        print(f"  - Overall results: {overall_file}")
        print(f"  - Subject summary: {subject_summary_file}")
        print(f"  - Activation matrix: {activation_file}")
        print(f"  - Config info: {config_file}")
        print(f"  - README: {readme_file}")
        
        return exp_dir
    
    def run_test(self, split_dir, train_gates=True, output_dir="./results"):
        """
        运行完整的测试流程
        
        Args:
            split_dir: 已有划分数据的目录
            train_gates: 是否训练门控模型（True=训练，False=加载已有模型）
            output_dir: 结果输出目录
        """
        print("="*80)
        print("SIMPLE MMLU TESTING PIPELINE")
        print("Using existing dataset split")
        print("="*80)
        
        # 1. 加载已有数据集
        self.load_existing_split(split_dir)
        
        # 2. 训练或加载门控模型
        if train_gates:
            self.train_gate_models()
        else:
            self.load_pretrained_gates()
        
        # 3. 评估单模型性能
        self.evaluate_single_models()
        
        # 4. 评估top-k集成策略（按领域计算）
        results = self.evaluate_top_k_strategy()
        
        # 5. 保存详细结果
        results_dir = self.save_results(
            results, 
            output_dir, 
            os.path.basename(split_dir.rstrip('/')),
            self.embedding_key,
            self.k
        )
        
        print("\n" + "="*80)
        print("TESTING COMPLETED")
        print(f"Results saved to: {results_dir}")
        print("="*80)
        
        return results, results_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化的MMLU测试脚本 - 直接使用已有的数据集')
    parser.add_argument('--split-dir', type=str, required=True,
                       help='已有划分数据的目录（包含domain_train.pkl和domain_test.pkl）')
    parser.add_argument('--embedding', type=str, default='bert', 
                       choices=['bert', 'e5-base', 'e5-large', 'gte-large', 'minilm'],
                       help='使用的embedding模型')
    parser.add_argument('--k', type=int, default=4,
                       help='top-k中的k值')
    parser.add_argument('--train-gates', action='store_true', default=True,
                       help='训练门控模型（默认）')
    parser.add_argument('--load-gates', dest='train_gates', action='store_false',
                       help='加载已有的门控模型（不训练）')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='结果输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建测试器并运行
    tester = SimpleMMLUTester(
        embedding_key=args.embedding,
        k=args.k,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results, results_dir = tester.run_test(
        split_dir=args.split_dir,
        train_gates=args.train_gates,
        output_dir=args.output_dir
    )
    
    # 打印最终摘要
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Split directory: {args.split_dir}")
    print(f"Embedding Model: {args.embedding}")
    print(f"Top-K Value: {args.k}")
    print(f"Training gates: {args.train_gates}")
    print(f"Results directory: {results_dir}")
    print(f"\nOverall Results:")
    print(f"  Ensemble Accuracy: {results['overall']['ensemble_accuracy']:.4f}")
    print(f"  Baseline Accuracy: {results['overall']['baseline_accuracy']:.4f}")
    print(f"  Improvement: {results['overall']['improvement']:+.4f}")
    print(f"  Total test samples: {results['overall']['total_samples']}")
    
    # 显示每个领域的结果
    print(f"\nPer-Subject Results:")
    print("-" * 80)
    for subject, result in results['subject_results'].items():
        print(f"{subject:30s}: Ensemble={result['ensemble_accuracy']:.4f}, "
              f"Baseline={result['baseline_accuracy']:.4f}, "
              f"Improvement={result['improvement']:+.4f}")
    print("="*80)


if __name__ == "__main__":
    main()