import os
import pandas as pd
from tqdm import tqdm

def process_csv_files(root_dir):
    # 遍历所有子目录（模型文件夹）
    models = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"开始处理 {len(models)} 个模型的数据...")

    for model in models:
        model_path = os.path.join(root_dir, model)
        csv_files = [f for f in os.listdir(model_path) if f.endswith('.csv')]
        
        print(f"正在处理模型: {model}")
        for csv_file in tqdm(csv_files):
            file_path = os.path.join(model_path, csv_file)
            
            try:
                # 读取CSV
                df = pd.read_csv(file_path)
                
                if 'question' not in df.columns:
                    continue
                
                # 定义转换函数
                def add_answer_suffix(text):
                    if pd.isna(text):
                        return text
                    text = str(text).strip()
                    # 检查是否已经包含 Answer: 结尾
                    if not text.endswith("Answer:"):
                        # 如果最后没有换行，建议加一个换行符，保持格式整洁
                        return text + "\nAnswer:"
                    return text

                # 应用修改
                df['question'] = df['question'].apply(add_answer_suffix)
                
                # 保存回原文件
                df.to_csv(file_path, index=False, encoding='utf-8')
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

if __name__ == "__main__":
    # 请确保你在 mmlu_hf 目录下运行，或者修改此路径
    current_directory = " " 
    process_csv_files(current_directory)
    print("\n✅ 所有模型的 question 格式已统一补全 'Answer:'")