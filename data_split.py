import numpy as np
from sklearn.model_selection import train_test_split
import pickle as pkl
import os
from config import *
from data_loader import load_mmlu_data

def split_mmlu_data(test_size=0.2):
    """划分MMLU数据集"""
    print("Loading MMLU data...")
    data_dict, questions, labels, topics = load_mmlu_data(MMLU_TRAIN_MODELS)
    
    indices = np.arange(len(questions))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=labels
    )
    
    train_data = {
        'data': {m: data_dict[m][train_idx] for m in MMLU_TRAIN_MODELS},
        'questions': [questions[i] for i in train_idx],
        'labels': [labels[i] for i in train_idx],
        'topics': [topics[i] for i in train_idx]
    }
    test_data = {
        'data': {m: data_dict[m][test_idx] for m in MMLU_TRAIN_MODELS},
        'questions': [questions[i] for i in test_idx],
        'labels': [labels[i] for i in test_idx],
        'topics': [topics[i] for i in test_idx]
    }
    
    save_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "mmlu_train.pkl"), "wb") as f:
        pkl.dump(train_data, f)
    with open(os.path.join(save_dir, "mmlu_test.pkl"), "wb") as f:
        pkl.dump(test_data, f)
    
    print(f"MMLU split done: {len(train_idx)} train, {len(test_idx)} test samples")


def prepare_gsm8k_data():
    """
    准备GSM8K数据 - 不进行拆分，直接使用原始的train/test
    只需要组织数据格式以便训练和评估使用
    """
    print("Preparing GSM8K data (using original train/test split)...")
    from data_loader import load_gsm8k_raw_predictions
    
    input_dir = os.path.join(CUR_DIR, DATA_DIR, "gsm8k")
    
    # 处理训练集
    print("\n  Loading train data...")
    try:
        train_predictions, train_questions, train_labels = load_gsm8k_raw_predictions(
            input_dir, GSM8K_TRAIN_MODELS,
            dataset_name="train",
            num_runs=GSM8K_CONFIG["num_runs"],
            num_samples=None  # 使用全部数据
        )
        
        train_data = {
            'raw_predictions': train_predictions,
            'questions': train_questions,
            'labels': train_labels
        }
        
        print(f"  ✓ Train: {len(train_questions)} samples, shape: {train_predictions.shape}")
    except Exception as e:
        print(f"  ✗ Error loading train data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 处理测试集
    print("\n  Loading test data...")
    try:
        test_predictions, test_questions, test_labels = load_gsm8k_raw_predictions(
            input_dir, GSM8K_TRAIN_MODELS,
            dataset_name="test",
            num_runs=GSM8K_CONFIG["num_runs"],
            num_samples=None  # 使用全部数据
        )
        
        test_data = {
            'raw_predictions': test_predictions,
            'questions': test_questions,
            'labels': test_labels
        }
        
        print(f"  ✓ Test: {len(test_questions)} samples, shape: {test_predictions.shape}")
    except Exception as e:
        print(f"  ✗ Error loading test data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存
    save_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "gsm8k_train.pkl"), "wb") as f:
        pkl.dump(train_data, f)
    with open(os.path.join(save_dir, "gsm8k_test.pkl"), "wb") as f:
        pkl.dump(test_data, f)
    
    print(f"\n✓ GSM8K data prepared: {len(train_questions)} train, {len(test_questions)} test samples")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="both", choices=["mmlu", "gsm8k", "both"])
    args = parser.parse_args()
    
    print("="*80)
    print("Data Preparation")
    print("="*80)
    
    if args.task in ["mmlu", "both"]:
        print("\n1. MMLU: Splitting data...")
        split_mmlu_data()
    
    if args.task in ["gsm8k", "both"]:
        print("\n2. GSM8K: Preparing data (no split needed)...")
        prepare_gsm8k_data()
    
    print("\n" + "="*80)
    print("Data preparation complete!")
    print("="*80)