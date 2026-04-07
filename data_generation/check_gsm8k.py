#!/usr/bin/env python3
"""
诊断GSM8K数据文件的问题
"""
import os
import json

# 设置路径
gsm8k_dir = "/data2/pyh/ensembleLLM/Claude_code/dataset/gsm8k"

print("="*80)
print("GSM8K Data File Diagnostics")
print("="*80)

# 1. 检查目录存在
if not os.path.exists(gsm8k_dir):
    print(f"❌ Directory does not exist: {gsm8k_dir}")
    exit(1)
else:
    print(f"✓ Directory exists: {gsm8k_dir}\n")

# 2. 列出所有文件
print("Files in directory:")
files = os.listdir(gsm8k_dir)
for f in sorted(files):
    if not os.path.isdir(os.path.join(gsm8k_dir, f)):
        size = os.path.getsize(os.path.join(gsm8k_dir, f))
        print(f"  {f:30s} ({size:,} bytes)")
print()

# 3. 检查数据文件
for split in ["train", "test"]:
    print(f"\n{'='*80}")
    print(f"Checking {split} split")
    print('='*80)
    
    # 尝试 .jsonl
    jsonl_path = os.path.join(gsm8k_dir, f"{split}.jsonl")
    json_path = os.path.join(gsm8k_dir, f"{split}.json")
    
    file_path = None
    file_format = None
    
    if os.path.exists(jsonl_path):
        file_path = jsonl_path
        file_format = "JSONL"
    elif os.path.exists(json_path):
        file_path = json_path
        file_format = "JSON"
    else:
        print(f"❌ No data file found for {split} split")
        print(f"   Tried: {split}.jsonl and {split}.json")
        continue
    
    print(f"✓ Found: {os.path.basename(file_path)} (format: {file_format})")
    print(f"  Size: {os.path.getsize(file_path):,} bytes")
    
    # 读取前3行
    print(f"\nFirst 3 lines:")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                line = line.strip()
                if line:
                    print(f"  Line {i+1}: {line[:150]}{'...' if len(line) > 150 else ''}")
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        continue
    
    # 尝试解析
    print(f"\nTrying to parse:")
    
    # 方法1: JSONL (每行一个JSON)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        success_count = 0
        for i, line in enumerate(lines[:5]):  # 只测试前5行
            try:
                obj = json.loads(line)
                success_count += 1
            except json.JSONDecodeError as e:
                print(f"  ❌ Line {i+1} is not valid JSON: {e}")
                print(f"     Content: {line[:100]}")
        
        if success_count > 0:
            print(f"  ✓ Successfully parsed {success_count}/{min(5, len(lines))} lines as JSONL")
            # 显示第一个对象的结构
            first_obj = json.loads(lines[0])
            print(f"  ✓ Keys in first object: {list(first_obj.keys())}")
            print(f"  ✓ Total lines in file: {len(lines)}")
        else:
            print(f"  ❌ Could not parse any lines as JSONL")
    except Exception as e:
        print(f"  ❌ JSONL parsing failed: {e}")
    
    # 方法2: JSON数组
    print(f"\n  Trying as JSON array...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"  ✓ Successfully parsed as JSON array")
            print(f"  ✓ Number of items: {len(data)}")
            if len(data) > 0:
                print(f"  ✓ Keys in first item: {list(data[0].keys())}")
        else:
            print(f"  ⚠ File is valid JSON but not an array")
    except Exception as e:
        print(f"  ✗ JSON array parsing failed: {e}")

print(f"\n{'='*80}")
print("Diagnostics Complete")
print('='*80)

# 4. 给出建议
print("\nRecommendations:")
print("1. GSM8K standard format is JSONL (one JSON per line)")
print("2. Each line should have 'question' and 'answer' fields")
print("3. Answer format: 'reasoning text #### numeric_answer'")
print("\nExample valid line:")
print('{"question": "Janet has 16 eggs...", "answer": "She eats 3... #### 18"}')