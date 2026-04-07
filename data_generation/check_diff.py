import os
import pandas as pd
from collections import defaultdict

def analyze_csv_content(file_path):
    """
    读取并分析CSV内容
    假设 MMLU 格式：第一列或某列是题目内容
    """
    try:
        # header=None 因为原始MMLU通常没有表头
        df = pd.read_csv(file_path, header=None)
        total_rows = len(df)
        
        # 假设第一列（索引0）是题目文本，以此作为唯一标识
        # 如果你的CSV格式不同，请修改索引
        unique_questions = df[0].nunique() 
        
        return {
            "total": total_rows,
            "unique": unique_questions,
            "has_duplicate": total_rows != unique_questions,
            "data_set": set(df[0].astype(str).tolist()) # 存储题目集合用于做模型间的交集对比
        }
    except Exception as e:
        return None

def main():
    base_dir = "/data2/pyh/ensembleLLM/Claude_code/dataset/mmlu_hf"
    models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # 存储所有数据：subject -> model -> info
    report = defaultdict(dict)
    all_subjects = set()

    print(f"开始深度扫描 {len(models)} 个模型文件夹...")

    for model in models:
        model_path = os.path.join(base_dir, model)
        csv_files = [f for f in os.listdir(model_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            info = analyze_csv_content(os.path.join(model_path, csv_file))
            if info:
                report[csv_file][model] = info
                all_subjects.add(csv_file)

    print("\n" + "="*50)
    print("🔍 数据一致性及重复项报告")
    print("="*50)

    for subject in sorted(list(all_subjects)):
        model_results = report[subject]
        
        # 1. 检查各模型间的“唯一题数”是否统一
        unique_counts = {m: info['unique'] for m, info in model_results.items()}
        counts_set = set(unique_counts.values())
        
        # 2. 检查模型内部是否有重复行
        duplicate_models = [m for m, info in model_results.items() if info['has_duplicate']]

        if len(counts_set) > 1 or len(duplicate_models) > 0 or len(model_results) < len(models):
            print(f"\n📂 主题: {subject}")
            
            if len(model_results) < len(models):
                print(f"   ⚠️ 缺失模型: {set(models) - set(model_results.keys())}")

            if len(duplicate_models) > 0:
                print(f"   ❌ 内部存在重复项的模型:")
                for m in duplicate_models:
                    print(f"      - {m}: 总行数 {model_results[m]['total']}, 唯一题数 {model_results[m]['unique']}")

            if len(counts_set) > 1:
                print(f"   ⚠️ 模型间唯一题数不一致:")
                for m, count in unique_counts.items():
                    print(f"      - {m}: {count} 个唯一题目")
            
            # 3. 检查交集（可选）：看是否所有模型都拥有完全相同的题目
            sets = [info['data_set'] for info in model_results.values()]
            if sets:
                common_questions = set.intersection(*sets)
                all_questions = set.union(*sets)
                if len(common_questions) != len(all_questions):
                    print(f"   💡 数据内容差异: 并集 {len(all_questions)} 题, 全体交集 {len(common_questions)} 题")
                    print(f"      (这意味着不同模型持有的题目内容不完全重合)")

    print("\n扫描完成。")

if __name__ == "__main__":
    main()