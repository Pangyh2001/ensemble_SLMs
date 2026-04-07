import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import argparse
from config import MMLU_TRAIN_MODELS, CUR_DIR, DATA_DIR
from data_loader import normalize_question

def load_mmlu_subject_data(subject, data_dir=None):
    """
    加载特定主题的MMLU数据
    """
    if data_dir is None:
        data_dir = os.path.join(CUR_DIR, DATA_DIR, "mmlu_hf")
    
    subject_data = {}
    subject_questions = []
    subject_labels = []
    
    csv_file = f"{subject}.csv"
    
    for model_name in MMLU_TRAIN_MODELS:
        model_dir = os.path.join(data_dir, model_name)
        csv_path = os.path.join(model_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            continue
        
        df = pd.read_csv(csv_path)
        model_predictions = []
        
        for idx, row in df.iterrows():
            if model_name == MMLU_TRAIN_MODELS[0]:
                subject_questions.append(row['question'])
                subject_labels.append(row['label'])
            
            pred_str = row['prediction']
            try:
                pred_list = eval(pred_str)
                model_predictions.append(pred_list)
            except:
                model_predictions.append([0, 0, 0, 0])
        
        subject_data[model_name] = np.array(model_predictions)
    
    return subject_data, subject_questions, subject_labels

def compute_model_accuracy(predictions, labels):
    """
    计算模型在特定主题上的准确率
    """
    correct = 0
    total = len(labels)
    
    for pred, label in zip(predictions, labels):
        pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
        if pred_class == label:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def compute_majority_vote_accuracy(subject_data, labels):
    """
    计算多数投票集成在特定主题上的准确率
    """
    model_names = list(subject_data.keys())
    num_models = len(model_names)
    num_samples = len(labels)
    
    if num_models == 0 or num_samples == 0:
        return 0.0
    
    correct = 0
    
    for sample_idx in range(num_samples):
        # 收集所有模型的预测
        votes = []
        for model_name in model_names:
            pred = subject_data[model_name][sample_idx]
            pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
            votes.append(pred_class)
        
        # 多数投票
        from collections import Counter
        vote_counts = Counter(votes)
        majority_vote = vote_counts.most_common(1)[0][0]
        
        # 检查是否正确
        if majority_vote == labels[sample_idx]:
            correct += 1
    
    return correct / num_samples

def evaluate_slm_baselines(test_subjects):
    """
    评估SLM在特定主题上的baseline性能
    """
    print("="*80)
    print("MMLU Subject-Specific Baseline Evaluation")
    print(f"Subjects: {', '.join(test_subjects)}")
    print(f"Models: {len(MMLU_TRAIN_MODELS)} SLMs")
    print("="*80)
    
    # 准备结果数据结构
    results_data = []
    
    for subject in tqdm(test_subjects, desc="Processing subjects"):
        print(f"\nEvaluating subject: {subject}")
        
        # 加载该主题的数据
        try:
            subject_data, questions, labels = load_mmlu_subject_data(subject)
        except Exception as e:
            print(f"Error loading subject {subject}: {e}")
            continue
        
        num_samples = len(labels)
        print(f"  Samples: {num_samples}")
        
        if num_samples == 0:
            print(f"  No data found for subject {subject}")
            continue
        
        # 1. 计算每个SLM在该主题上的准确率
        model_accuracies = {}
        for model_name in MMLU_TRAIN_MODELS:
            if model_name in subject_data:
                acc = compute_model_accuracy(subject_data[model_name], labels)
                model_accuracies[model_name] = acc
                
                # 添加到结果数据
                results_data.append({
                    'subject': subject,
                    'model': model_name,
                    'method': 'single_model',
                    'accuracy': acc,
                    'samples': num_samples
                })
        
        # 打印单个模型性能
        print(f"  Individual model accuracies:")
        for model_name, acc in sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"    {model_name:<30}: {acc:.4f}")
        
        # 2. 计算多数投票准确率
        majority_vote_acc = compute_majority_vote_accuracy(subject_data, labels)
        
        # 添加到结果数据
        results_data.append({
            'subject': subject,
            'model': 'majority_vote',
            'method': 'ensemble',
            'accuracy': majority_vote_acc,
            'samples': num_samples
        })
        
        print(f"  Majority vote accuracy: {majority_vote_acc:.4f}")
        
        # 3. 计算平均模型性能（作为参考）
        avg_accuracy = np.mean(list(model_accuracies.values())) if model_accuracies else 0.0
        results_data.append({
            'subject': subject,
            'model': 'average',
            'method': 'reference',
            'accuracy': avg_accuracy,
            'samples': num_samples
        })
        
        # 4. 计算最佳单个模型性能
        best_accuracy = max(model_accuracies.values()) if model_accuracies else 0.0
        best_model = max(model_accuracies, key=model_accuracies.get) if model_accuracies else 'N/A'
        results_data.append({
            'subject': subject,
            'model': f'best_single ({best_model})',
            'method': 'reference',
            'accuracy': best_accuracy,
            'samples': num_samples
        })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results_data)
    
    # 保存到CSV
    output_file = os.path.join(CUR_DIR, "results", "baselines_mmlu_subjects.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    
    # 按主题总结
    subjects_summary = {}
    for subject in test_subjects:
        subject_results = results_df[results_df['subject'] == subject]
        
        # 单个模型平均
        single_models = subject_results[subject_results['method'] == 'single_model']
        avg_single = single_models['accuracy'].mean() if not single_models.empty else 0.0
        
        # 多数投票
        majority = subject_results[subject_results['model'] == 'majority_vote']
        majority_acc = majority['accuracy'].iloc[0] if not majority.empty else 0.0
        
        # 最佳单个
        best_single = subject_results[subject_results['model'].str.contains('best_single')]
        best_acc = best_single['accuracy'].iloc[0] if not best_single.empty else 0.0
        
        subjects_summary[subject] = {
            'avg_single': avg_single,
            'majority_vote': majority_acc,
            'best_single': best_acc,
            'num_models': len(single_models),
            'samples': subject_results['samples'].iloc[0] if not subject_results.empty else 0
        }
    
    # 打印详细总结
    print("\nSubject-wise Performance:")
    print("-" * 80)
    print(f"{'Subject':<35} {'Samples':<8} {'Models':<6} {'Avg Single':<12} {'Best Single':<12} {'Majority Vote':<12}")
    print("-" * 80)
    
    for subject, stats in subjects_summary.items():
        print(f"{subject:<35} {stats['samples']:<8} {stats['num_models']:<6} "
              f"{stats['avg_single']:.4f}      {stats['best_single']:.4f}      "
              f"{stats['majority_vote']:.4f}")
    
    # 计算总体统计
    all_single_results = results_df[results_df['method'] == 'single_model']
    overall_avg_single = all_single_results['accuracy'].mean() if not all_single_results.empty else 0.0
    
    all_majority_results = results_df[results_df['model'] == 'majority_vote']
    overall_majority = all_majority_results['accuracy'].mean() if not all_majority_results.empty else 0.0
    
    print("\nOverall Statistics:")
    print(f"  Average single model accuracy across all subjects: {overall_avg_single:.4f}")
    print(f"  Average majority vote accuracy across all subjects: {overall_majority:.4f}")
    
    if overall_majority > overall_avg_single:
        improvement = (overall_majority - overall_avg_single) / overall_avg_single * 100
        print(f"  Majority vote improves over average single model by: {improvement:.2f}%")
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="MMLU Subject-Specific Baseline Evaluation")
    parser.add_argument("--subjects", nargs="+", default=None,
                       help="Specific subjects to evaluate (default: predefined 8 subjects)")
    parser.add_argument("--output", default=None,
                       help="Output CSV file path (default: results/baselines_mmlu_subjects.csv)")
    
    args = parser.parse_args()
    
    # 预定义的8个主题
    predefined_subjects = [
        "electrical_engineering",
        "high_school_government_and_politics",
        "high_school_psychology",
        "human_aging",
        "miscellaneous",
        "professional_psychology",
        "sociology",
        "us_foreign_policy"
    ]
    
    # 使用用户指定的主题或预定义主题
    test_subjects = args.subjects if args.subjects else predefined_subjects
    
    # 检查数据目录是否存在
    data_dir = os.path.join(CUR_DIR, DATA_DIR, "mmlu_hf")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure MMLU data is downloaded and organized correctly.")
        return
    
    # 运行评估
    results_df = evaluate_slm_baselines(test_subjects)
    
    # 如果有指定输出文件，额外保存一份
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nAdditional copy saved to: {args.output}")

if __name__ == "__main__":
    main()