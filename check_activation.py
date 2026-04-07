import torch
import os
import pickle as pkl
import numpy as np
import argparse
import json
from tqdm import tqdm
from collections import Counter
import random
from config import *
from gate_model import GateNetwork
from data_loader import compute_gsm8k_accuracy, load_gsm8k_data, normalize_question

def calculate_activation_stats(task_type, embedding_key="bert"):
    """
    计算指定任务在测试集上的：
    1. 平均激活 SLM 数量
    2. 每个 SLM 的激活频率
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 获取任务对应的阈值配置
    threshold = THRESHOLD_CONFIG.get(task_type, 0.6)
    
    print(f"\n{'='*80}")
    print(f"ACTIVATION ANALYSIS: {task_type.upper()}")
    print(f"Embedding: {embedding_key}")
    print(f"Threshold: {threshold}")
    print(f"{'='*80}")

    # 2. 确定模型列表
    if task_type == "mmlu":
        model_list = MMLU_TRAIN_MODELS
    else:
        model_list = GSM8K_TRAIN_MODELS

    # 3. 加载所有训练好的 Gate 模型
    print("Loading Gate models...")
    gate_models = {}
    for model_name in model_list:
        gate = GateNetwork(
            embedding_key=embedding_key,
            hidden_dim=TRAIN_CONFIG["hidden_dim"],
            dropout=TRAIN_CONFIG["dropout"]
        ).to(device)
        
        model_path = os.path.join(GATE_DIR, f"{model_name}_{task_type}.pt")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            gate.load_state_dict(checkpoint['model_state_dict'])
            gate.eval()
            gate_models[model_name] = gate
        else:
            print(f"Warning: Gate model not found for {model_name}, skipping...")
    
    if not gate_models:
        print("Error: No gate models loaded. Please train the gates first.")
        return

    # 初始化每个模型的计数器
    model_activation_counts = {name: 0 for name in gate_models.keys()}

    # 4. 加载测试数据集
    split_path = os.path.join(CUR_DIR, DATA_DIR, "splits", f"{task_type}_test.pkl")
    if not os.path.exists(split_path):
        print(f"Error: Test data not found at {split_path}. Please run data_split.py first.")
        return
        
    print(f"Loading test data from {split_path}...")
    with open(split_path, "rb") as f:
        test_data = pkl.load(f)
    
    if task_type == "mmlu":
        questions = test_data['questions']
    else:
        questions = test_data['questions']
        raw_predictions = test_data['raw_predictions']
        labels = test_data['labels']
    
    total_questions = len(questions)
    total_activated_count = 0 # 累积所有问题的激活总数
    
    # 5. 遍历问题计算激活数
    print("Calculating activations...")
    
    with torch.no_grad():
        for idx in tqdm(range(total_questions), desc=f"Scanning {task_type}"):
            q_text = questions[idx]
            
            # 获取所有 Gate 的打分
            scores = {}
            for name, gate in gate_models.items():
                s = gate([q_text]).item()
                scores[name] = s
            
            # 筛选超过阈值的模型
            activated_models = [name for name, s in scores.items() if s > threshold]
            
            # 兜底逻辑：如果没有模型超过阈值，选择分数最高的那个
            if not activated_models:
                max_model = max(scores, key=scores.get)
                activated_models = [max_model]
            
            # 更新总激活数
            total_activated_count += len(activated_models)
            
            # 更新每个模型的激活计数
            for model_name in activated_models:
                model_activation_counts[model_name] += 1

    # 6. 计算统计结果
    avg_activation = total_activated_count / total_questions
    
    print(f"\n{'-'*40}")
    print(f"RESULTS SUMMARY - {task_type.upper()}")
    print(f"{'-'*40}")
    print(f"Total Questions: {total_questions}")
    print(f"Average Activated SLMs: {avg_activation:.4f}")
    
    print(f"\n{'-'*40}")
    print(f"PER-MODEL ACTIVATION FREQUENCY")
    print(f"(Sorted by frequency)")
    print(f"{'-'*40}")
    print(f"{'Model Name':<30} | {'Count':<8} | {'Frequency':<10}")
    print("-" * 54)
    
    # 按激活频率排序
    sorted_models = sorted(model_activation_counts.items(), key=lambda x: x[1], reverse=True)
    
    for name, count in sorted_models:
        freq = (count / total_questions) * 100
        print(f"{name:<30} | {count:<8} | {freq:.2f}%")
    print("-" * 54)


def test_gsm8k_multiple_thresholds(embedding_key="bert"):
    """
    专门测试GSM8K任务在多个阈值下的性能：
    1. 门控方法的准确率
    2. threshold_random (baseline) 方法的准确率
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"GSM8K MULTI-THRESHOLD EVALUATION")
    print(f"Embedding: {embedding_key}")
    print(f"{'='*80}")
    
    # 1. 加载GSM8K测试数据
    print("Loading GSM8K test data...")
    split_path = os.path.join(CUR_DIR, DATA_DIR, "splits", "gsm8k_test.pkl")
    if not os.path.exists(split_path):
        print(f"Error: Test data not found at {split_path}. Please run data_split.py first.")
        return
    
    with open(split_path, "rb") as f:
        test_data = pkl.load(f)
    
    questions = test_data['questions']
    raw_predictions = test_data['raw_predictions']  # (samples, models, runs)
    labels = test_data['labels']
    
    total_samples = len(questions)
    print(f"Loaded {total_samples} test samples")
    
    # 2. 加载所有训练好的 Gate 模型
    print("\nLoading Gate models...")
    gate_models = {}
    for model_name in GSM8K_TRAIN_MODELS:
        gate = GateNetwork(
            embedding_key=embedding_key,
            hidden_dim=TRAIN_CONFIG["hidden_dim"],
            dropout=TRAIN_CONFIG["dropout"]
        ).to(device)
        
        model_path = os.path.join(GATE_DIR, f"{model_name}_gsm8k.pt")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            gate.load_state_dict(checkpoint['model_state_dict'])
            gate.eval()
            gate_models[model_name] = gate
            print(f"  ✓ Loaded gate for {model_name}")
        else:
            print(f"  ✗ Gate model not found for {model_name}")
    
    if not gate_models:
        print("Error: No gate models loaded. Please train the gates first.")
        return
    
    # 3. 定义要测试的阈值
    thresholds = [i/10 for i in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
    print(f"\nTesting thresholds: {thresholds}")
    
    # 4. 预先计算所有问题的门控分数（加速）
    print("\nPre-computing gate scores for all questions...")
    all_gate_scores = []
    
    with torch.no_grad():
        for idx in tqdm(range(total_samples), desc="Computing gate scores"):
            q_text = questions[idx]
            scores = {}
            for name, gate in gate_models.items():
                s = gate([q_text]).item()
                scores[name] = s
            all_gate_scores.append(scores)
    
    # 5. 测试每个阈值
    gate_results = {}
    random_results = {}
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold = {threshold:.1f} ---")
        
        # 5.1 门控方法测试
        gate_correct = 0
        gate_activation_counts = Counter()
        
        for idx in tqdm(range(total_samples), desc="Gate method", leave=False):
            gate_scores = all_gate_scores[idx]
            true_label = labels[idx]
            
            # 选择激活的模型（分数 > 阈值）
            activated_models = [m for m, s in gate_scores.items() if s > threshold]
            
            # 如果没有模型激活，选择分数最高的模型
            if not activated_models:
                max_model = max(gate_scores, key=gate_scores.get)
                activated_models = [max_model]
            
            # 统计激活数量
            gate_activation_counts[len(activated_models)] += 1
            
            # GSM8K集成：多数投票
            model_answers = []
            model_idx_map = {name: i for i, name in enumerate(GSM8K_TRAIN_MODELS)}
            
            for model_name in activated_models:
                if model_name in model_idx_map:
                    model_idx = model_idx_map[model_name]
                    model_runs = raw_predictions[idx, model_idx, :]
                    
                    # 获取模型最可能的答案（取众数）
                    valid_answers = [a for a in model_runs if not np.isnan(a)]
                    if valid_answers:
                        most_common = Counter(valid_answers).most_common(1)[0][0]
                        model_answers.append(most_common)
            
            if model_answers:
                # 多数投票决定最终答案
                final_answer = Counter(model_answers).most_common(1)[0][0]
                
                # 检查是否正确
                try:
                    if abs(float(final_answer) - float(true_label)) < 1e-4:
                        gate_correct += 1
                except:
                    pass
        
        gate_accuracy = gate_correct / total_samples
        
        # 5.2 threshold_random (baseline) 方法测试
        random.seed(42)  # 固定随机种子保证可重复性
        random_correct = 0
        
        for idx in tqdm(range(total_samples), desc="Random baseline", leave=False):
            gate_scores = all_gate_scores[idx]
            true_label = labels[idx]
            
            # 首先确定门控方法会激活多少模型
            activated_models = [m for m, s in gate_scores.items() if s > threshold]
            if not activated_models:
                activated_count = 1
            else:
                activated_count = len(activated_models)
            
            # 随机选择相同数量的模型
            all_models = list(gate_models.keys())
            random_models = random.sample(all_models, min(activated_count, len(all_models)))
            
            # 用随机选择的模型进行集成
            model_answers = []
            model_idx_map = {name: i for i, name in enumerate(GSM8K_TRAIN_MODELS)}
            
            for model_name in random_models:
                if model_name in model_idx_map:
                    model_idx = model_idx_map[model_name]
                    model_runs = raw_predictions[idx, model_idx, :]
                    
                    valid_answers = [a for a in model_runs if not np.isnan(a)]
                    if valid_answers:
                        most_common = Counter(valid_answers).most_common(1)[0][0]
                        model_answers.append(most_common)
            
            if model_answers:
                final_answer = Counter(model_answers).most_common(1)[0][0]
                
                try:
                    if abs(float(final_answer) - float(true_label)) < 1e-4:
                        random_correct += 1
                except:
                    pass
        
        random_accuracy = random_correct / total_samples
        
        # 5.3 存储结果
        gate_results[threshold] = {
            'accuracy': gate_accuracy,
            'total_samples': total_samples,
            'correct': gate_correct,
            'activation_distribution': dict(gate_activation_counts)
        }
        
        random_results[threshold] = {
            'accuracy': random_accuracy,
            'total_samples': total_samples,
            'correct': random_correct
        }
        
        print(f"  Gate method accuracy: {gate_accuracy:.4f}")
        print(f"  Random baseline accuracy: {random_accuracy:.4f}")
        print(f"  Difference: {gate_accuracy - random_accuracy:+.4f}")
    
    # 6. 输出汇总结果
    print(f"\n{'='*80}")
    print(f"GSM8K MULTI-THRESHOLD RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Threshold':<10} {'Gate Acc':<12} {'Random Acc':<12} {'Difference':<12}")
    print("-" * 46)
    
    for threshold in sorted(thresholds):
        gate_acc = gate_results[threshold]['accuracy']
        random_acc = random_results[threshold]['accuracy']
        diff = gate_acc - random_acc
        print(f"{threshold:<10.1f} {gate_acc:<12.4f} {random_acc:<12.4f} {diff:<+12.4f}")
    
    print("-" * 46)
    
    # 找出最佳阈值
    best_threshold = max(gate_results.keys(), key=lambda t: gate_results[t]['accuracy'])
    best_gate_acc = gate_results[best_threshold]['accuracy']
    best_random_acc = random_results[best_threshold]['accuracy']
    
    print(f"\nBest threshold: {best_threshold:.1f}")
    print(f"  Gate method: {best_gate_acc:.4f}")
    print(f"  Random baseline: {best_random_acc:.4f}")
    print(f"  Improvement: {best_gate_acc - best_random_acc:+.4f}")
    
    # 7. 保存详细结果
    result_dir = os.path.join(CUR_DIR, "results_gsm8k_threshold")
    os.makedirs(result_dir, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"gsm8k_threshold_results_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump({
            'configuration': {
                'embedding': embedding_key,
                'thresholds': thresholds,
                'total_samples': total_samples,
                'timestamp': timestamp
            },
            'gate_results': gate_results,
            'random_results': random_results,
            'summary': {
                'best_threshold': best_threshold,
                'best_gate_accuracy': best_gate_acc,
                'best_random_accuracy': best_random_acc,
                'improvement': best_gate_acc - best_random_acc
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {result_file}")
    
    # 8. 激活分布分析（最佳阈值）
    print(f"\n{'='*80}")
    print(f"ACTIVATION DISTRIBUTION (Best threshold: {best_threshold:.1f})")
    print(f"{'='*80}")
    
    activation_dist = gate_results[best_threshold]['activation_distribution']
    total_activated = sum(k * v for k, v in activation_dist.items())
    avg_activated = total_activated / total_samples
    
    print(f"Average activated models: {avg_activated:.2f}")
    print(f"\n{'# Models':<10} {'Count':<10} {'Percentage':<10}")
    print("-" * 30)
    
    for n_models in sorted(activation_dist.keys()):
        count = activation_dist[n_models]
        percentage = (count / total_samples) * 100
        print(f"{n_models:<10} {count:<10} {percentage:.1f}%")
    
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算平均激活SLM数量及频率（扩展版）')
    
    parser.add_argument("--task", default="all", choices=["mmlu", "gsm8k", "all"],
                       help="选择任务: mmlu, gsm8k 或 all (默认: all)")
    
    parser.add_argument("--embedding", default="bert", 
                       help="使用的 Embedding 模型 (默认: bert)")
    
    parser.add_argument("--gsm8k-thresholds", action="store_true",
                       help="为GSM8K任务运行多阈值测试")
    
    args = parser.parse_args()

    if args.gsm8k_thresholds:
        # 专门运行GSM8K的多阈值测试
        test_gsm8k_multiple_thresholds(embedding_key=args.embedding)
    else:
        # 原来的功能
        if args.task == "all":
            calculate_activation_stats("mmlu", embedding_key=args.embedding)
            calculate_activation_stats("gsm8k", embedding_key=args.embedding)
        else:
            calculate_activation_stats(args.task, embedding_key=args.embedding)