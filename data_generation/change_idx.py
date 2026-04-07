import os
import pandas as pd
from functools import reduce

def sanitize_question(text):
    """移除末尾的 Answer: 及其前后的空白符，用于匹配唯一题目"""
    if pd.isna(text): return ""
    text = str(text).strip()
    if text.endswith("Answer:"):
        text = text[:-7].strip()
    return text

def finalize_alignment(root_dir):
    # 获取所有子目录（模型）
    models = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # 获取所有出现过的 csv 文件名
    subjects = set()
    for m in models:
        subjects.update([f for f in os.listdir(os.path.join(root_dir, m)) if f.endswith('.csv')])

    print(f"检测到 {len(models)} 个模型，共 {len(subjects)} 个主题文件。")

    for sub in sorted(list(subjects)):
        model_dfs = {}
        
        # 1. 加载并预清洗
        for m in models:
            path = os.path.join(root_dir, m, sub)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if 'question' not in df.columns:
                        continue
                    
                    # 辅助列用于内容对齐
                    df['clean_q'] = df['question'].apply(sanitize_question)
                    # 内部去重：保留第一次出现的题目
                    df = df.drop_duplicates(subset=['clean_q'], keep='first')
                    model_dfs[m] = df
                except Exception as e:
                    print(f"读取 {path} 失败: {e}")

        if not model_dfs:
            continue

        # 2. 跨模型取题目交集
        common_qs_sets = [set(df['clean_q']) for df in model_dfs.values()]
        common_qs = reduce(lambda x, y: x.intersection(y), common_qs_sets)

        if len(common_qs) == 0:
            print(f"⚠️ 警告: 主题 {sub} 的交集为空，跳过该主题。")
            continue

        # 为了保证所有模型在同一文件中的题目顺序一致，预先排好序
        sorted_common_qs = sorted(list(common_qs))

        # 3. 过滤、统一格式、重置 idx 并保存
        for m in models:
            if m not in model_dfs: continue
            
            df = model_dfs[m]
            # 过滤出交集题目
            df_aligned = df[df['clean_q'].isin(common_qs)].copy()
            
            # --- 修正后的排序逻辑 ---
            # 将 clean_q 转为分类类型，其顺序由 sorted_common_qs 定义
            df_aligned['clean_q'] = pd.Categorical(df_aligned['clean_q'], categories=sorted_common_qs, ordered=True)
            df_aligned = df_aligned.sort_values('clean_q')
            
            # 统一加上 Answer: 后缀
            df_aligned['question'] = df_aligned['clean_q'].astype(str) + "\nAnswer:"
            
            # 重置 idx 从 0 开始
            df_aligned['idx'] = range(len(df_aligned))
            
            # 调整列顺序
            cols = ['idx', 'question', 'prediction', 'label']
            existing_cols = [c for c in cols if c in df_aligned.columns]
            
            final_path = os.path.join(root_dir, m, sub)
            df_aligned[existing_cols].to_csv(final_path, index=False, encoding='utf-8')

        print(f"✅ 已完成对齐与重排: {sub} (共 {len(sorted_common_qs)} 题)")

if __name__ == "__main__":
    # 指定你的路径
    target_path = " "
    finalize_alignment(target_path)
    print("\n🚀 所有处理已完成！现在各模型间的数据已完全对齐且索引重置。")