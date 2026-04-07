import os
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from collections import defaultdict
from config import *
from data_loader import load_mmlu_data, load_mmlu_single_model


def create_single_split(model_names, seed=42):
    """
    一次性划分数据：每个领域8:2，然后混合打乱
    """
    print(f"Creating split with seed={seed}")
    
    # 加载所有数据
    data_dict, questions, labels, topics = load_mmlu_data(model_names)
    unique_topics = list(set(topics))
    
    # 按领域划分
    train_indices = []
    test_indices = []
    
    for topic in unique_topics:
        topic_indices = [i for i, t in enumerate(topics) if t == topic]
        
        if len(topic_indices) < 2:
            train_indices.extend(topic_indices)
            continue
        
        topic_train, topic_test = train_test_split(
            topic_indices, test_size=0.2, random_state=seed, 
            stratify=[labels[i] for i in topic_indices]
        )
        train_indices.extend(topic_train)
        test_indices.extend(topic_test)
    
    # 打乱
    np.random.seed(seed)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return train_indices, test_indices


def evaluate_model_on_split(model_name, data_dict, questions, labels, indices):
    """
    在指定划分上评估模型
    """
    if model_name in data_dict:
        # 训练模型：直接评估
        predictions = data_dict[model_name][indices]
        split_labels = [labels[i] for i in indices]
        
        correct = 0
        for pred, label in zip(predictions, split_labels):
            pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
            if pred_class == label:
                correct += 1
        
        return correct / len(indices)
    else:
        # Baseline模型：需要单独加载
        try:
            all_preds, all_questions, all_labels = load_mmlu_single_model(model_name)
            from data_loader import normalize_question
            
            # 创建映射
            q_to_idx = {normalize_question(q): i for i, q in enumerate(all_questions)}
            
            correct = 0
            for idx in indices:
                q = questions[idx]
                norm_q = normalize_question(q)
                true_label = labels[idx]
                
                if norm_q in q_to_idx:
                    pred_idx = q_to_idx[norm_q]
                    pred = all_preds[pred_idx]
                    pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
                    
                    if pred_class == true_label:
                        correct += 1
            
            return correct / len(indices)
        except:
            return 0.0


def run_all_seeds(seeds=[42, 123, 0], output_csv="mmlu_results.csv"):
    """
    运行所有seed的实验
    """
    print("="*60)
    print("MMLU EVALUATION - MULTIPLE SEEDS")
    print(f"Seeds: {seeds}")
    print("="*60)
    
    # 所有要评估的模型
    train_models = MMLU_TRAIN_MODELS
    baseline_models = MMLU_BASELINE_MODELS
    all_models = train_models + baseline_models
    
    print(f"\nModels to evaluate:")
    print(f"  Training models: {len(train_models)}")
    print(f"  Baseline models: {len(baseline_models)}")
    
    # 先加载一次数据（所有模型）
    print("\nLoading all training model data...")
    data_dict, questions, labels, topics = load_mmlu_data(train_models)
    
    # 存储结果
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Seed: {seed}")
        print(f"{'='*40}")
        
        # 创建划分
        train_idx, test_idx = create_single_split(train_models, seed)
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")
        
        # 评估所有模型
        seed_results = {}
        
        for model in all_models:
            # 在测试集上评估
            test_acc = evaluate_model_on_split(model, data_dict, questions, labels, test_idx)
            seed_results[model] = test_acc
            
            print(f"  {model:<30}: {test_acc:.4f}")
        
        results[seed] = seed_results
    
    # 计算统计
    print(f"\n{'='*60}")
    print("CALCULATING STATISTICS")
    print(f"{'='*60}")
    
    stats_data = []
    
    for model in all_models:
        accuracies = [results[seed][model] for seed in seeds]
        
        stats_row = {
            'model': model,
            'type': 'baseline' if model in baseline_models else 'training',
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies)
        }
        
        # 添加每个seed的准确率
        for i, seed in enumerate(seeds, 1):
            stats_row[f'seed_{i}'] = accuracies[i-1]
        
        stats_data.append(stats_row)
    
    # 创建DataFrame并排序
    df = pd.DataFrame(stats_data)
    df = df.sort_values('mean', ascending=False)
    
    # 保存到CSV
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # 打印结果
    print(f"\n{'='*80}")
    print("FINAL RESULTS (Sorted by Mean Accuracy)")
    print(f"{'='*80}")
    
    print(f"\n{'Model':<35} {'Type':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 77)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<35} {row['type']:<10} {row['mean']:<8.4f} "
              f"{row['std']:<8.4f} {row['min']:<8.4f} {row['max']:<8.4f}")
    
    # 分组统计
    train_df = df[df['type'] == 'training']
    baseline_df = df[df['type'] == 'baseline']
    
    print(f"\n{'='*60}")
    print("GROUP SUMMARY")
    print(f"{'='*60}")
    
    if not train_df.empty:
        print(f"\nTraining Models ({len(train_df)}):")
        print(f"  Mean: {train_df['mean'].mean():.4f} ± {train_df['mean'].std():.4f}")
        print(f"  Best: {train_df.iloc[0]['model']} ({train_df.iloc[0]['mean']:.4f})")
        print(f"  Worst: {train_df.iloc[-1]['model']} ({train_df.iloc[-1]['mean']:.4f})")
    
    if not baseline_df.empty:
        print(f"\nBaseline Models ({len(baseline_df)}):")
        print(f"  Mean: {baseline_df['mean'].mean():.4f} ± {baseline_df['mean'].std():.4f}")
        print(f"  Best: {baseline_df.iloc[0]['model']} ({baseline_df.iloc[0]['mean']:.4f})")
        print(f"  Worst: {baseline_df.iloc[-1]['model']} ({baseline_df.iloc[-1]['mean']:.4f})")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {os.path.abspath(output_csv)}")
    print(f"{'='*60}")
    
    return df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MMLU多seed评估脚本')
    parser.add_argument('--seeds', type=str, default='42,123,0',
                       help='随机种子列表，用逗号分隔')
    parser.add_argument('--output', type=str, default='mmlu_results.csv',
                       help='输出CSV文件名')
    
    args = parser.parse_args()
    
    # 解析种子
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    print(f"MMLU Evaluation")
    print(f"Seeds: {seeds}")
    print(f"Output: {args.output}")
    
    # 运行所有seed
    results = run_all_seeds(seeds, args.output)
    
    return results


if __name__ == "__main__":
    main()