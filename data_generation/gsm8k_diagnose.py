#!/usr/bin/env python3
"""
诊断GSM8K数据对齐问题
"""
import os
import sys
import json
import pickle
import numpy as np
import re

# 添加项目路径
sys.path.insert(0, '/data2/pyh/ensembleLLM/Claude_code')

from data_loader import normalize_question

def load_test_json():
    """加载test.json文件"""
    path = "/data2/pyh/ensembleLLM/Claude_code/dataset/gsm8k/test.json"
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_model_pkl(model_name):
    """加载模型的pkl文件"""
    pkl_path = f"/data2/pyh/ensembleLLM/Claude_code/dataset/gsm8k/{model_name}/test/run_1318_outputs.pkl"
    
    if not os.path.exists(pkl_path):
        print(f"❌ PKL file not found: {pkl_path}")
        return None
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_question_from_pkl_item(item):
    """从pkl item中提取问题"""
    # 尝试多种可能的格式
    if isinstance(item, dict):
        if 'question' in item:
            return item['question']
        elif 'instruction' in item:
            return item['instruction']
    
    if isinstance(item, (list, tuple)) and len(item) > 0:
        first = item[0]
        if isinstance(first, dict):
            if 'question' in first:
                return first['question']
            elif 'instruction' in first:
                return first['instruction']
        elif isinstance(first, str):
            return first
    
    return None

def main():
    print("="*80)
    print("GSM8K Data Alignment Diagnostics")
    print("="*80)
    
    # 1. 加载test.json
    print("\n1. Loading test.json...")
    test_data = load_test_json()
    print(f"   Total samples: {len(test_data)}")
    
    # 提取前3个问题
    test_questions = []
    for i, item in enumerate(test_data[:3]):
        q = item.get('instruction', '')
        test_questions.append(q)
        print(f"\n   Sample {i+1} (raw):")
        print(f"   {q[:200]}...")
        print(f"\n   Sample {i+1} (normalized):")
        print(f"   {normalize_question(q)[:200]}...")
    
    # 2. 加载模型pkl
    models_to_check = ["Llama-2-70b-chat-hf", "gemma-7b"]
    
    for model_name in models_to_check:
        print(f"\n{'='*80}")
        print(f"2. Checking model: {model_name}")
        print('='*80)
        
        pkl_data = load_model_pkl(model_name)
        if pkl_data is None:
            continue
        
        print(f"   PKL data type: {type(pkl_data)}")
        print(f"   PKL data length: {len(pkl_data) if isinstance(pkl_data, (list, tuple)) else 'N/A'}")
        
        if isinstance(pkl_data, (list, tuple)) and len(pkl_data) > 0:
            print(f"\n   First item type: {type(pkl_data[0])}")
            print(f"   First item structure:")
            
            first_item = pkl_data[0]
            if isinstance(first_item, dict):
                print(f"   Keys: {list(first_item.keys())}")
            elif isinstance(first_item, (list, tuple)):
                print(f"   Length: {len(first_item)}")
                if len(first_item) > 0:
                    print(f"   first_item[0] type: {type(first_item[0])}")
                    if isinstance(first_item[0], dict):
                        print(f"   first_item[0] keys: {list(first_item[0].keys())}")
            
            # 尝试提取前3个问题
            print(f"\n   Extracted questions from PKL:")
            for i in range(min(3, len(pkl_data))):
                q = extract_question_from_pkl_item(pkl_data[i])
                if q:
                    print(f"\n   Sample {i+1} (raw):")
                    print(f"   {q[:200]}...")
                    print(f"\n   Sample {i+1} (normalized):")
                    print(f"   {normalize_question(q)[:200]}...")
                else:
                    print(f"\n   Sample {i+1}: Could not extract question")
            
            # 3. 对比匹配
            print(f"\n   Checking alignment with test.json:")
            pkl_questions_norm = []
            for i in range(min(len(pkl_data), len(test_data))):
                q = extract_question_from_pkl_item(pkl_data[i])
                if q:
                    pkl_questions_norm.append(normalize_question(q))
                else:
                    pkl_questions_norm.append("")
            
            test_questions_norm = [normalize_question(item.get('instruction', '')) 
                                   for item in test_data[:len(pkl_questions_norm)]]
            
            # 检查直接匹配
            direct_matches = sum(1 for p, t in zip(pkl_questions_norm, test_questions_norm) if p == t)
            print(f"   Direct matches (same order): {direct_matches}/{len(pkl_questions_norm)}")
            
            # 检查是否有任何问题在test.json中
            test_q_set = set(normalize_question(item.get('instruction', '')) for item in test_data)
            pkl_in_test = sum(1 for q in pkl_questions_norm if q in test_q_set and q != "")
            print(f"   PKL questions found in test.json: {pkl_in_test}/{len(pkl_questions_norm)}")
            
            # 显示一些不匹配的例子
            if direct_matches < len(pkl_questions_norm):
                print(f"\n   First mismatch example:")
                for i in range(min(5, len(pkl_questions_norm))):
                    if pkl_questions_norm[i] != test_questions_norm[i]:
                        print(f"\n   Position {i}:")
                        print(f"   PKL:  {pkl_questions_norm[i][:100]}...")
                        print(f"   Test: {test_questions_norm[i][:100]}...")
                        break
    
    # 4. 检查test split的问题
    print(f"\n{'='*80}")
    print("3. Checking test split data structure")
    print('='*80)
    
    split_path = "/data2/pyh/ensembleLLM/Claude_code/dataset/splits/gsm8k_test.pkl"
    if os.path.exists(split_path):
        with open(split_path, 'rb') as f:
            test_split = pickle.load(f)
        
        print(f"   Keys in test_split: {test_split.keys()}")
        print(f"   Number of questions: {len(test_split['questions'])}")
        
        print(f"\n   First 3 questions from test_split:")
        for i in range(min(3, len(test_split['questions']))):
            q = test_split['questions'][i]
            print(f"\n   Sample {i+1} (raw):")
            print(f"   {q[:200]}...")
            print(f"\n   Sample {i+1} (normalized):")
            print(f"   {normalize_question(q)[:200]}...")
    else:
        print(f"   ❌ Test split file not found: {split_path}")
    
    print(f"\n{'='*80}")
    print("Diagnostics Complete")
    print('='*80)

if __name__ == "__main__":
    main()