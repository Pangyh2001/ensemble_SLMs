import os
import json
import pickle as pkl
import numpy as np
from collections import Counter
from tqdm import tqdm
from config import CUR_DIR, DATA_DIR, MMLU_TRAIN_MODELS, GSM8K_TRAIN_MODELS

def check_correct_gsm8k(pred, label):
    """检查GSM8K预测是否正确 (浮点数比较)"""
    if np.isnan(label):
        return False
    try:
        return abs(float(pred) - float(label)) < 1e-4
    except:
        return False

def evaluate_mmlu():
    print("\n" + "="*50)
    print("Evaluating MMLU (Majority Vote & Oracle)")
    print("="*50)
    
    # 加载测试集
    split_path = os.path.join(CUR_DIR, DATA_DIR, "splits", "mmlu_test.pkl")
    if not os.path.exists(split_path):
        print(f"Error: {split_path} not found. Please run 'python main.py --mode split' first.")
        return None

    with open(split_path, "rb") as f:
        test_data = pkl.load(f)

    questions = test_data['questions']
    labels = test_data['labels']
    model_data = test_data['data'] # dict: {model_name: logits_array}
    
    total = len(questions)
    maj_correct = 0
    oracle_correct = 0
    
    # 映射索引到字母
    idx_to_char = ['A', 'B', 'C', 'D']
    
    for i in tqdm(range(total), desc="MMLU Processing"):
        label = labels[i]
        votes = []
        
        # 收集所有模型的预测
        for model_name in MMLU_TRAIN_MODELS:
            logits = model_data[model_name][i]
            pred_idx = np.argmax(logits) # 取最大概率的索引
            pred_char = idx_to_char[pred_idx]
            votes.append(pred_char)
        
        # 1. Majority Voting
        # 如果票数相同，Counter默认按出现顺序返回，这里简单处理
        most_common_ans = Counter(votes).most_common(1)[0][0]
        if most_common_ans == label:
            maj_correct += 1
            
        # 2. Oracle (只要有一个模型答对)
        if label in votes:
            oracle_correct += 1
            
    return {
        "majority_voting": maj_correct / total,
        "oracle": oracle_correct / total
    }

def evaluate_gsm8k():
    print("\n" + "="*50)
    print("Evaluating GSM8K (Majority Vote & Oracle)")
    print("="*50)
    
    # 加载测试集
    split_path = os.path.join(CUR_DIR, DATA_DIR, "splits", "gsm8k_test.pkl")
    if not os.path.exists(split_path):
        print(f"Error: {split_path} not found. Please run 'python main.py --mode split' first.")
        return None

    with open(split_path, "rb") as f:
        test_data = pkl.load(f)
        
    # raw_predictions shape: (num_samples, num_models, num_runs)
    raw_preds = test_data['raw_predictions']
    labels = test_data['labels']
    
    total = len(labels)
    maj_correct = 0
    oracle_correct = 0
    
    for i in tqdm(range(total), desc="GSM8K Processing"):
        label = labels[i]
        
        # 收集每个模型的最终答案
        # 注意：GSM8K每个模型有多次采样(runs)，我们先取每个模型内部的多数投票作为该模型的答案
        model_final_answers = []
        
        for m_idx in range(len(GSM8K_TRAIN_MODELS)):
            runs = raw_preds[i, m_idx, :]
            # 过滤掉无效值 (nan)
            valid_runs = [r for r in runs if not np.isnan(r)]
            
            if valid_runs:
                # 该模型内部投票
                model_ans = Counter(valid_runs).most_common(1)[0][0]
                model_final_answers.append(model_ans)
            # 如果该模型全是nan，则该模型弃权，不加入model_final_answers
        
        if not model_final_answers:
            continue
            
        # 1. Majority Voting (所有模型的答案再投一次票)
        system_ans = Counter(model_final_answers).most_common(1)[0][0]
        if check_correct_gsm8k(system_ans, label):
            maj_correct += 1
            
        # 2. Oracle (只要有一个模型答对)
        # 检查是否有任何一个模型的最终答案与标签匹配
        if any(check_correct_gsm8k(ans, label) for ans in model_final_answers):
            oracle_correct += 1

    return {
        "majority_voting": maj_correct / total,
        "oracle": oracle_correct / total
    }

def main():
    results = {}
    
    # 运行 MMLU
    mmlu_res = evaluate_mmlu()
    if mmlu_res:
        results["mmlu"] = mmlu_res
        print(f"MMLU Results: {mmlu_res}")
        
    # 运行 GSM8K
    gsm8k_res = evaluate_gsm8k()
    if gsm8k_res:
        results["gsm8k"] = gsm8k_res
        print(f"GSM8K Results: {gsm8k_res}")
    
    # 保存结果
    output_file = "comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()