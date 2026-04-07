import os
import sys
import json
import pickle as pkl
import numpy as np
import pandas as pd
import random
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CUR_DIR, DATA_DIR, GATE_DIR, TRAIN_CONFIG, EMBEDDING_MODELS, MMLU_TRAIN_MODELS
from gate_model import GateNetwork

# 创建特定的门控文件夹
SPECIFIC_GATE_DIR = os.path.join(CUR_DIR, "gate_mmlu_1")
os.makedirs(SPECIFIC_GATE_DIR, exist_ok=True)

class MMLU_EnsembleTester:
    """MMLU集成测试器（精简版）"""
    def __init__(self, test_data, embedding_key="bert"):
        self.test_data = test_data
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载所有门控模型
        self.gate_models = {}
        print(f"\nLoading gate models from {SPECIFIC_GATE_DIR}...")
        
        for model_name in MMLU_TRAIN_MODELS:
            model_path = os.path.join(SPECIFIC_GATE_DIR, f"{model_name}.pt")
            if not os.path.exists(model_path):
                print(f"  Warning: Gate model for {model_name} not found at {model_path}")
                continue
                
            # 创建门控网络
            gate_model = GateNetwork(
                embedding_key=embedding_key,
                hidden_dim=TRAIN_CONFIG["hidden_dim"],
                dropout=TRAIN_CONFIG["dropout"]
            ).to(self.device)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            gate_model.load_state_dict(checkpoint['model_state_dict'])
            gate_model.eval()
            
            self.gate_models[model_name] = gate_model
        
        if not self.gate_models:
            raise ValueError("No gate models loaded!")
        
        print(f"Loaded {len(self.gate_models)} gate models")
    
    def get_gate_scores(self, question):
        """获取单个问题的门控分数"""
        scores = {}
        with torch.no_grad():
            for model_name, gate_model in self.gate_models.items():
                s = gate_model([question])
                scores[model_name] = s.item()  # 标量分数
        return scores
    
    def test_single_threshold(self, threshold, use_random_baseline=False, seed=42):
        """测试单个阈值（可选择是否使用随机基线）"""
        questions = self.test_data['questions']
        labels = self.test_data['labels']
        model_predictions = self.test_data['data']
        
        total_samples = len(questions)
        total_correct = 0
        
        # 用于统计激活频率
        activation_counts = {model_name: 0 for model_name in self.gate_models.keys()}
        
        # 设置随机种子（用于random baseline）
        if use_random_baseline:
            random.seed(seed)
            np.random.seed(seed)
        
        # 预先计算所有问题的门控分数
        print(f"  Pre-computing gate scores...")
        all_gate_scores = []
        for idx in tqdm(range(total_samples), desc="Computing gate scores", leave=False):
            q_text = questions[idx]
            gate_scores = self.get_gate_scores(q_text)
            all_gate_scores.append(gate_scores)
        
        print(f"  Testing with {'random baseline' if use_random_baseline else 'gate'} threshold={threshold}...")
        for idx in tqdm(range(total_samples), desc="Testing", leave=False):
            true_label = labels[idx]
            gate_scores = all_gate_scores[idx]
            
            if use_random_baseline:
                # Random baseline: 随机选择激活的模型（数量与门控方法相同）
                # 首先用门控方法确定应该激活多少模型
                activated_models = [m for m, s in gate_scores.items() if s > threshold]
                if not activated_models:
                    # 如果没有激活，选择分数最高的模型
                    activated_count = 1
                else:
                    activated_count = len(activated_models)
                
                # 随机选择相同数量的模型
                all_models = list(self.gate_models.keys())
                activated_models = list(np.random.choice(all_models, activated_count, replace=False))
            else:
                # 正常的门控方法
                activated_models = [m for m, s in gate_scores.items() if s > threshold]
                if not activated_models:
                    max_model = max(gate_scores, key=gate_scores.get)
                    activated_models = [max_model]
            
            # 统计激活频率（只在门控方法时统计）
            if not use_random_baseline:
                for model_name in activated_models:
                    activation_counts[model_name] += 1
            
            # 集成预测
            ensemble_logits = np.zeros(4)
            for model_name in activated_models:
                model_pred = model_predictions[model_name][idx]
                model_probs = np.exp(model_pred) / np.sum(np.exp(model_pred))
                ensemble_logits += model_probs
            
            # 获取最终预测
            pred_idx = np.argmax(ensemble_logits)
            pred_label = ['A', 'B', 'C', 'D'][pred_idx]
            
            # 检查是否正确
            if pred_label == true_label:
                total_correct += 1
        
        # 计算准确率
        accuracy = total_correct / total_samples
        
        # 计算激活频率（转换为百分比）
        if not use_random_baseline:
            activation_freq = {model: count/total_samples for model, count in activation_counts.items()}
        else:
            activation_freq = None
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'activation_frequency': activation_freq,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'method': 'random_baseline' if use_random_baseline else 'gate'
        }
    
    def test_multiple_thresholds(self, thresholds):
        """测试多个阈值，同时获取门控方法和随机基线结果"""
        print(f"\nTesting {len(thresholds)} thresholds...")
        
        gate_results = {}
        random_results = {}
        
        for threshold in thresholds:
            print(f"\n--- Testing threshold = {threshold:.2f} ---")
            
            # 测试门控方法
            gate_result = self.test_single_threshold(threshold, use_random_baseline=False)
            gate_results[threshold] = gate_result
            print(f"  Gate method accuracy: {gate_result['accuracy']:.4f}")
            
            # 测试随机基线
            random_result = self.test_single_threshold(threshold, use_random_baseline=True, seed=42)
            random_results[threshold] = random_result
            print(f"  Random baseline accuracy: {random_result['accuracy']:.4f}")
        
        return {
            'gate_results': gate_results,
            'random_results': random_results
        }


def load_test_data():
    """加载测试数据"""
    split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits_mmlu_specific")
    test_path = os.path.join(split_dir, "mmlu_test.pkl")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. Please run data split first.")
    
    print(f"Loading test data from {test_path}...")
    with open(test_path, "rb") as f:
        test_data = pkl.load(f)
    
    print(f"  Test samples: {len(test_data['questions'])}")
    return test_data


def test_single_models_baseline(test_data):
    """测试单模型作为参考基线"""
    print("\n" + "="*80)
    print("Single Model Baseline Performance")
    print("="*80)
    
    questions = test_data['questions']
    labels = test_data['labels']
    model_predictions = test_data['data']
    
    single_model_results = {}
    
    for model_name in MMLU_TRAIN_MODELS:
        if model_name not in model_predictions:
            print(f"  Warning: No predictions for {model_name}")
            continue
        
        predictions = model_predictions[model_name]
        correct = 0
        total = len(questions)
        
        for i in range(total):
            pred_logits = predictions[i]
            pred_idx = np.argmax(pred_logits)
            pred_label = ['A', 'B', 'C', 'D'][pred_idx]
            
            if pred_label == labels[i]:
                correct += 1
        
        accuracy = correct / total
        single_model_results[model_name] = accuracy
        print(f"  {model_name:30s}: {accuracy:.4f}")
    
    return single_model_results


def format_results_for_display(all_results, single_model_results):
    """格式化结果以便显示"""
    print("\n" + "="*80)
    print("FOCUSED RESULTS SUMMARY")
    print("="*80)
    
    # 1. 单模型基线
    print("\n1. SINGLE MODEL BASELINE:")
    print("-" * 50)
    avg_single = np.mean(list(single_model_results.values())) if single_model_results else 0
    print(f"Average single model accuracy: {avg_single:.4f}")
    
    # 2. 阈值对比表格
    print("\n2. THRESHOLD COMPARISON (Gate vs Random Baseline):")
    print("-" * 80)
    print(f"{'Threshold':<10} {'Gate Accuracy':<15} {'Random Baseline':<15} {'Difference':<15}")
    print("-" * 80)
    
    gate_results = all_results['gate_results']
    random_results = all_results['random_results']
    
    for threshold in sorted(gate_results.keys()):
        gate_acc = gate_results[threshold]['accuracy']
        random_acc = random_results[threshold]['accuracy']
        diff = gate_acc - random_acc
        
        print(f"{threshold:<10.2f} {gate_acc:<15.4f} {random_acc:<15.4f} {diff:<+15.4f}")
    
    print("-" * 80)
    
    # 3. 最佳阈值
    best_threshold = max(gate_results.keys(), key=lambda t: gate_results[t]['accuracy'])
    best_gate_acc = gate_results[best_threshold]['accuracy']
    best_random_acc = random_results[best_threshold]['accuracy']
    
    print(f"\n3. BEST THRESHOLD: {best_threshold:.2f}")
    print(f"   Gate method accuracy: {best_gate_acc:.4f}")
    print(f"   Random baseline accuracy: {best_random_acc:.4f}")
    print(f"   Improvement over random: {best_gate_acc - best_random_acc:+.4f}")
    
    if single_model_results:
        improvement_over_avg = best_gate_acc - avg_single
        print(f"   Improvement over avg single model: {improvement_over_avg:+.4f}")
    
    # 4. SLM激活频率表（只显示最佳阈值）
    print(f"\n4. SLM ACTIVATION FREQUENCY (at threshold {best_threshold:.2f}):")
    print("-" * 60)
    print(f"{'Model':<30} {'Activation Freq':<15} {'Single Acc':<15}")
    print("-" * 60)
    
    activation_freq = gate_results[best_threshold]['activation_frequency']
    if activation_freq:
        # 按激活频率排序
        sorted_models = sorted(activation_freq.items(), key=lambda x: x[1], reverse=True)
        
        for model_name, freq in sorted_models:
            single_acc = single_model_results.get(model_name, 0)
            print(f"{model_name:<30} {freq:<15.4f} {single_acc:<15.4f}")
    
    print("-" * 60)


def save_detailed_results(all_results, single_model_results, thresholds, embedding):
    """保存详细结果"""
    result_dir = os.path.join(CUR_DIR, "results_mmlu_focused")
    os.makedirs(result_dir, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存JSON格式的完整结果
    json_file = os.path.join(result_dir, f"focused_results_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump({
            'configuration': {
                'thresholds': thresholds,
                'embedding': embedding,
                'test_samples': all_results['gate_results'][thresholds[0]]['total_samples'],
                'timestamp': timestamp
            },
            'single_model_baseline': single_model_results,
            'gate_results': {t: all_results['gate_results'][t] for t in thresholds},
            'random_baseline_results': {t: all_results['random_results'][t] for t in thresholds},
            'summary': {
                'avg_single_model_accuracy': np.mean(list(single_model_results.values())) if single_model_results else 0,
                'best_threshold': max(all_results['gate_results'].keys(), 
                                    key=lambda t: all_results['gate_results'][t]['accuracy']),
                'best_gate_accuracy': max(r['accuracy'] for r in all_results['gate_results'].values()),
                'best_random_accuracy': max(r['accuracy'] for r in all_results['random_results'].values())
            }
        }, f, indent=2)
    
    # 2. 保存CSV格式便于分析
    csv_file = os.path.join(result_dir, f"threshold_comparison_{timestamp}.csv")
    
    # 准备CSV数据
    csv_data = []
    for threshold in sorted(all_results['gate_results'].keys()):
        gate_acc = all_results['gate_results'][threshold]['accuracy']
        random_acc = all_results['random_results'][threshold]['accuracy']
        
        # 收集激活频率（如果有）
        activation_info = all_results['gate_results'][threshold]['activation_frequency']
        if activation_info:
            activation_str = ";".join([f"{m}:{f:.3f}" for m, f in sorted(activation_info.items())])
        else:
            activation_str = ""
        
        csv_data.append({
            'threshold': threshold,
            'gate_accuracy': gate_acc,
            'random_baseline_accuracy': random_acc,
            'difference': gate_acc - random_acc,
            'activation_frequencies': activation_str
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    # 3. 保存激活频率矩阵（所有阈值）
    freq_matrix_file = os.path.join(result_dir, f"activation_matrix_{timestamp}.csv")
    
    # 构建激活频率矩阵
    freq_matrix = []
    for threshold in sorted(all_results['gate_results'].keys()):
        activation_freq = all_results['gate_results'][threshold]['activation_frequency']
        if activation_freq:
            row = {'threshold': threshold}
            row.update(activation_freq)
            freq_matrix.append(row)
    
    if freq_matrix:
        freq_df = pd.DataFrame(freq_matrix)
        freq_df.to_csv(freq_matrix_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  JSON results: {json_file}")
    print(f"  CSV comparison: {csv_file}")
    if freq_matrix:
        print(f"  Activation matrix: {freq_matrix_file}")


def main():
    parser = argparse.ArgumentParser(description='MMLU聚焦测试脚本')
    parser.add_argument('--thresholds', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                       help='要测试的阈值列表，用逗号分隔 (默认: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)')
    parser.add_argument('--embedding', type=str, default='bert',
                       choices=['bert', 'e5-base', 'e5-large', 'gte-large', 'minilm'],
                       help='Embedding模型选择 (默认: bert)')
    parser.add_argument('--test-only', action='store_true',
                       help='仅测试模式（跳过单模型基线测试）')
    
    args = parser.parse_args()
    
    # 解析阈值列表
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    
    print("="*80)
    print("MMLU FOCUSED TEST SCRIPT")
    print("="*80)
    print(f"Testing thresholds: {thresholds}")
    print(f"Embedding: {args.embedding}")
    print(f"Test only mode: {args.test_only}")
    print("="*80)
    
    try:
        # 1. 加载测试数据
        test_data = load_test_data()
        
        # 2. 测试单模型基线（可选）
        single_model_results = {}
        if not args.test_only:
            single_model_results = test_single_models_baseline(test_data)
        
        # 3. 测试多阈值集成系统
        print("\n" + "="*80)
        print("Testing Ensemble System with Multiple Thresholds")
        print("="*80)
        
        tester = MMLU_EnsembleTester(test_data, args.embedding)
        all_results = tester.test_multiple_thresholds(thresholds)
        
        # 4. 格式化显示结果
        format_results_for_display(all_results, single_model_results)
        
        # 5. 保存详细结果
        save_detailed_results(all_results, single_model_results, thresholds, args.embedding)
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()