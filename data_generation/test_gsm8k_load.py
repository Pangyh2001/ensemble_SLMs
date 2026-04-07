#!/usr/bin/env python3
"""
测试GSM8K数据加载是否正常
"""
import sys

import pandas as pd
sys.path.insert(0, '/data2/pyh/ensembleLLM/Claude_code')

from data_loader import load_gsm8k_data
import json

print("="*80)
print("Testing GSM8K Data Loading")
print("="*80)

# 测试train split
print("\n1. Loading TRAIN split...")
try:
    questions_train, labels_train = load_gsm8k_data("train")
    print(f"\n✓ Successfully loaded TRAIN data")
    print(f"  Total questions: {len(questions_train)}")
    print(f"  Valid labels: {len([l for l in labels_train if not pd.isna(l)])}")
    print(f"  Invalid labels: {len([l for l in labels_train if pd.isna(l)])}")
    
    # 显示第一个样本
    print(f"\n  First sample:")
    print(f"    Question: {questions_train[0][:150]}...")
    print(f"    Label: {labels_train[0]}")
    
    # 显示统计
    import numpy as np
    valid_labels = [l for l in labels_train if not np.isnan(l)]
    if valid_labels:
        print(f"\n  Label statistics:")
        print(f"    Min: {min(valid_labels)}")
        print(f"    Max: {max(valid_labels)}")
        print(f"    Mean: {np.mean(valid_labels):.2f}")
    
except Exception as e:
    print(f"✗ Error loading TRAIN data: {e}")
    import traceback
    traceback.print_exc()

# 测试test split
print("\n" + "="*80)
print("2. Loading TEST split...")
try:
    questions_test, labels_test = load_gsm8k_data("test")
    print(f"\n✓ Successfully loaded TEST data")
    print(f"  Total questions: {len(questions_test)}")
    print(f"  Valid labels: {len([l for l in labels_test if not pd.isna(l)])}")
    print(f"  Invalid labels: {len([l for l in labels_test if pd.isna(l)])}")
    
    # 显示第一个样本
    print(f"\n  First sample:")
    print(f"    Question: {questions_test[0][:150]}...")
    print(f"    Label: {labels_test[0]}")
    
except Exception as e:
    print(f"✗ Error loading TEST data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Test Complete")
print("="*80)

# 显示原始数据示例
print("\n3. Checking raw data format...")
try:
    with open('/data2/pyh/ensembleLLM/Claude_code/dataset/gsm8k/train.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n  First raw item from train.json:")
    first_item = data[0]
    for key, value in first_item.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"    {key}: {value[:100]}...")
        else:
            print(f"    {key}: {value}")
            
except Exception as e:
    print(f"  Could not load raw data: {e}")