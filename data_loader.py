import os
import pandas as pd
import numpy as np
import pickle as pkl
import re
import json
from config import *
from collections import Counter

def load_mmlu_data(model_names, data_dir=None):
    """加载MMLU数据集"""
    if data_dir is None:
        data_dir = os.path.join(CUR_DIR, DATA_DIR, "mmlu_hf")
        
    all_data = {model: [] for model in model_names}
    questions = []
    labels = []
    topics = []
    
    first_model_dir = os.path.join(data_dir, model_names[0])
    if not os.path.exists(first_model_dir):
        raise FileNotFoundError(f"Directory not found: {first_model_dir}")
        
    csv_files = [f for f in os.listdir(first_model_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        topic = csv_file.replace('.csv', '')
        
        for model_name in model_names:
            model_dir = os.path.join(data_dir, model_name)
            csv_path = os.path.join(model_dir, csv_file)
            
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found.")
                continue
                
            df = pd.read_csv(csv_path)
            
            for idx, row in df.iterrows():
                if model_name == model_names[0]:
                    questions.append(row['question'])
                    labels.append(row['label'])
                    topics.append(topic)
                
                pred_str = row['prediction']
                try:
                    pred_list = eval(pred_str)
                    all_data[model_name].append(pred_list)
                except:
                    all_data[model_name].append([0,0,0,0])

    data_dict = {}
    for model_name in model_names:
        data_dict[model_name] = np.array(all_data[model_name])
    
    return data_dict, questions, labels, topics


def load_mmlu_single_model(model_name, data_dir=None):
    """加载单个MMLU模型的数据（用于baseline评估）"""
    if data_dir is None:
        data_dir = os.path.join(CUR_DIR, DATA_DIR, "mmlu_hf")
    
    model_dir = os.path.join(data_dir, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    predictions = []
    questions = []
    labels = []
    
    csv_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.csv')])
    
    for csv_file in csv_files:
        csv_path = os.path.join(model_dir, csv_file)
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            questions.append(row['question'])
            labels.append(row['label'])
            try:
                pred_list = eval(row['prediction'])
                predictions.append(pred_list)
            except:
                predictions.append([0, 0, 0, 0])
    
    return np.array(predictions), questions, labels


def normalize_question(text):
    """标准化问题文本，用于对齐"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除可能的 "Answer:" 后缀
    text = re.sub(r'\s*Answer:\s*$', '', text, flags=re.IGNORECASE)
    return text.lower()


def load_gsm8k_data(dataset_name="train"):
    """
    加载 GSM8K 的问题和标签 - 支持多种格式
    返回标准化的问题文本用于对齐
    """
    data_path = os.path.join(CUR_DIR, DATA_DIR, "gsm8k", f"{dataset_name}.jsonl")
    
    if not os.path.exists(data_path):
        data_path = os.path.join(CUR_DIR, DATA_DIR, "gsm8k", f"{dataset_name}.json")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"GSM8K data file not found. Tried:\n"
            f"  - {os.path.join(CUR_DIR, DATA_DIR, 'gsm8k', dataset_name)}.jsonl\n"
            f"  - {os.path.join(CUR_DIR, DATA_DIR, 'gsm8k', dataset_name)}.json"
        )

    print(f"Loading GSM8K data from: {data_path}")

    data_df = None
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data_df = pd.DataFrame(data)
            print(f"  ✓ Loaded as JSON array: {len(data)} samples")
    except Exception as e:
        print(f"  ✗ Could not load as JSON array: {e}")
        
        try:
            data_df = pd.read_json(data_path, lines=True)
            print(f"  ✓ Loaded as JSONL: {len(data_df)} samples")
        except Exception as e2:
            print(f"  ✗ Could not load as JSONL: {e2}")
            raise ValueError(f"Could not load data from {data_path}")
    
    if data_df is None:
        raise ValueError(f"Could not load data from {data_path}")

    questions, labels = [], []
    
    print(f"  Detected columns: {list(data_df.columns)}")
    
    if 'question' in data_df.columns and 'answer' in data_df.columns:
        print("  Format: Standard GSM8K (question/answer)")
        pattern = r'####\s*(\S+)'
        for i, row in data_df.iterrows():
            # 标准化问题文本
            q = normalize_question(row['question'])
            questions.append(q)
            
            matches = re.findall(pattern, str(row['answer']))
            if matches:
                try:
                    labels.append(float(matches[0].replace(",", "")))
                except:
                    labels.append(np.nan)
            else:
                labels.append(np.nan)
                
    elif 'instruction' in data_df.columns and 'ground_truth' in data_df.columns:
        print("  Format: Custom format (instruction/ground_truth)")
        pattern = r'####\s*(\S+)'
        
        for i, row in data_df.iterrows():
            # 标准化问题文本
            q = normalize_question(row['instruction'])
            questions.append(q)
            
            ground_truth = str(row['ground_truth'])
            
            matches = re.findall(pattern, ground_truth)
            if matches:
                try:
                    labels.append(float(matches[-1].replace(",", "")))
                except:
                    labels.append(np.nan)
            else:
                try:
                    numbers = re.findall(r'-?\d+\.?\d*', ground_truth)
                    if numbers:
                        labels.append(float(numbers[-1].replace(",", "")))
                    else:
                        labels.append(np.nan)
                except:
                    labels.append(np.nan)
    else:
        raise ValueError(
            f"Unknown data format. Expected columns: "
            f"(question/answer) or (instruction/ground_truth), "
            f"but got: {list(data_df.columns)}"
        )

    valid_count = len([l for l in labels if not np.isnan(l)])
    print(f"  ✓ Extracted {len(questions)} questions, {valid_count} valid labels")
    
    if valid_count == 0:
        print("\n  WARNING: No valid labels found!")
        print(f"  Sample ground_truth: {data_df['ground_truth'].iloc[0] if 'ground_truth' in data_df.columns else 'N/A'}")
    
    return questions, labels


def load_gsm8k_raw_predictions(input_dir, model_names, num_samples=None,
                                num_runs=10, dataset_name="train"):
    """
    加载GSM8K原始预测数据（答案值，非概率）
    
    重要：PKL文件只包含模型输出，不包含原始问题！
    预测值和test.json是按索引一一对应的，不需要文本匹配。
    
    返回:
        all_predictions: (num_samples, num_models, num_runs) - 原始答案值
        questions: List[str] - 原始问题文本
        labels: List[float]
    """
    print(f"\n  Loading {dataset_name} predictions for {len(model_names)} models...")
    
    # 1. 先加载ground truth数据（问题和标签）
    questions_gt, labels_gt = load_gsm8k_data(dataset_name=dataset_name)
    num_gt_samples = len(questions_gt)
    
    # 2. 加载每个模型的预测数组
    model_predictions = []
    min_samples = num_gt_samples
    
    for model_n in model_names:
        results_dir = os.path.join(input_dir, model_n, dataset_name)
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Model result directory not found: {results_dir}")
        
        # 找到最大的run ID
        all_files_id = [int(fn.split("_")[1]) for fn in
                        os.listdir(results_dir) if "npy" in fn and "run_" in fn]
        if not all_files_id:
            raise ValueError(f"No npy files found in {results_dir}")
        max_id = max(all_files_id)
        
        pred_path = os.path.join(results_dir, f"run_{max_id}_predictions.npy")
        
        # 加载预测值: (samples, runs)
        pred_arr = np.load(pred_path)[:, :num_runs]
        
        # 检查样本数
        if pred_arr.shape[0] < min_samples:
            min_samples = pred_arr.shape[0]
        
        # 确保预测数量和ground truth对齐
        # 如果预测数量大于ground truth，截断；如果小于，填充nan
        if pred_arr.shape[0] < num_gt_samples:
            # 填充nan
            padded = np.full((num_gt_samples, num_runs), np.nan)
            padded[:pred_arr.shape[0]] = pred_arr
            model_predictions.append(padded)
            print(f"    {model_n}: {pred_arr.shape[0]} samples (padded to {num_gt_samples})")
        else:
            # 截断到ground truth大小
            model_predictions.append(pred_arr[:num_gt_samples])
            print(f"    {model_n}: {pred_arr.shape[0]} samples (using first {num_gt_samples})")
    
    # 3. 确定实际使用的样本数
    # 使用所有模型和ground truth都有的最小样本数
    if num_samples is not None:
        final_samples = min(num_samples, min_samples, num_gt_samples)
    else:
        final_samples = min(min_samples, num_gt_samples)
    
    # 4. 提取对齐的数据
    all_predictions = np.stack([pred[:final_samples] for pred in model_predictions], axis=1)
    questions = questions_gt[:final_samples]
    labels = labels_gt[:final_samples]
    
    print(f"  ✓ Final data: {final_samples} samples across {len(model_names)} models")
    print(f"    Shape: {all_predictions.shape}")
    
    return all_predictions, questions, labels


def compute_gsm8k_accuracy(predictions, label):
    """
    计算GSM8K准确率：10次采样中正确的比例
    
    Args:
        predictions: (num_runs,) 原始答案值
        label: float 正确答案
    
    Returns:
        accuracy: 0到1之间的准确率
    """
    if np.isnan(label):
        return 0.0
    
    correct_count = 0
    for pred in predictions:
        try:
            if abs(float(pred) - float(label)) < 1e-4:
                correct_count += 1
        except:
            pass
    
    return correct_count / len(predictions)


def load_gsm8k_single_model(model_name, input_dir, num_samples=None,
                             num_runs=10, dataset_name="test"):
    """
    加载单个GSM8K模型的数据（用于baseline评估）
    
    重要：PKL文件只包含模型输出，不包含原始问题！
    预测值和test.json是按索引一一对应的。
    """
    results_dir = os.path.join(input_dir, model_name, dataset_name)
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Model directory not found: {results_dir}")
    
    # 找到最大的run ID
    all_files_id = [int(fn.split("_")[1]) for fn in
                    os.listdir(results_dir) if "npy" in fn and "run_" in fn]
    if not all_files_id:
        raise ValueError(f"No npy files found in {results_dir}")
    max_id = max(all_files_id)
    
    pred_path = os.path.join(results_dir, f"run_{max_id}_predictions.npy")
    
    # 加载预测: (samples, runs)
    predictions = np.load(pred_path)[:, :num_runs]
    
    # 加载ground truth（问题和标签）
    questions_gt, labels_gt = load_gsm8k_data(dataset_name=dataset_name)
    
    # 确保对齐
    min_len = min(len(predictions), len(questions_gt))
    
    if num_samples is not None:
        min_len = min(min_len, num_samples)
    
    predictions = predictions[:min_len]
    questions = questions_gt[:min_len]
    labels = labels_gt[:min_len]
    
    return predictions, questions, labels