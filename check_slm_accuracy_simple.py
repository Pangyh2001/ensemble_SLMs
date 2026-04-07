# check_slm_accuracy_simple.py
import os
import json
import csv
import glob
import numpy as np
from collections import Counter
from typing import List, Dict, Any
import re

def normalize_text(text: str) -> str:
    """标准化文本"""
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace(",", "").replace(" ", "").lower()
    for suffix in ["$", "%", "°", "ml", "kg", "m", "cm", "km"]:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    return text

def extract_final_answer_text(answer: str) -> str:
    """从GSM8K答案中提取最终答案"""
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()

def majority_vote_text(texts: List[str]) -> str:
    """多数投票"""
    if not texts:
        return ""
    normed = [normalize_text(t) for t in texts]
    counts = Counter(normed)
    if not counts:
        return ""
    most_common_norm = counts.most_common(1)[0][0]
    for t in texts:
        if normalize_text(t) == most_common_norm:
            return t
    return texts[0]

def is_gsm8k_correct(pred: str, gold: str) -> bool:
    """判断GSM8K答案是否正确"""
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if pred_norm == gold_norm:
        return True
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        return abs(pred_num - gold_num) < 1e-6
    except (ValueError, TypeError):
        return False

def parse_prediction_to_label(pred_str: str) -> str:
    """解析MMLU预测为标签"""
    if pred_str is None:
        return ""
    s = str(pred_str).strip()
    try:
        s2 = s.replace("[", "").replace("]", "")
        nums = [float(x) for x in s2.split(",") if str(x).strip() != ""]
        if len(nums) == 0:
            return ""
        idx = int(np.argmax(nums))
        mapping = ["A", "B", "C", "D"]
        if 0 <= idx < len(mapping):
            return mapping[idx]
        return chr(ord("A") + idx)
    except Exception:
        return ""

def extract_gold_answer_from_ground_truth(ground_truth: str) -> str:
    """从ground_truth字段提取最终答案"""
    if not ground_truth:
        return ""
    
    ground_truth = str(ground_truth)
    
    # 方法1: 提取####后面的数字
    if "####" in ground_truth:
        parts = ground_truth.split("####")
        if len(parts) > 1:
            answer = parts[-1].strip()
            # 清理答案中的非数字字符（保留小数点和负号）
            cleaned = re.sub(r'[^\d.-]', '', answer)
            if cleaned:
                return cleaned
    
    # 方法2: 直接提取所有数字，取最后一个
    numbers = re.findall(r'-?\d+\.?\d*', ground_truth)
    if numbers:
        return numbers[-1]
    
    return ""

def check_slm_accuracy(gsm8k_gold_path: str, gsm8k_pred_path: str, mmlu_dir: str):
    """检查SLM在两个数据集上的正确率"""
    
    # 初始化统计
    stats = {
        "gsm8k": {"total": 0, "correct": 0},
        "mmlu": {"total": 0, "correct": 0},
        "all": {"total": 0, "correct": 0}
    }
    
    print("=" * 70)
    print("SLM基础正确率统计（不加门控）")
    print("=" * 70)
    
    # 1. 检查GSM8K - 支持JSON数组格式
    if os.path.exists(gsm8k_gold_path) and os.path.exists(gsm8k_pred_path):
        print(f"\n[GSM8K] 加载数据...")
        print(f"  金标文件: {gsm8k_gold_path}")
        print(f"  预测文件: {gsm8k_pred_path}")
        
        # 加载金标 - 针对JSON数组格式
        gsm_gold = []
        try:
            with open(gsm8k_gold_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # 直接加载整个JSON数组
                
            if not isinstance(data, list):
                print(f"  ✗ 错误: 文件不是JSON数组格式")
                return stats
            
            print(f"  成功读取JSON数组，包含 {len(data)} 个元素")
            
            # 处理每个样本
            valid_count = 0
            for i, obj in enumerate(data):
                ground_truth = obj.get("ground_truth", "")
                gold_answer = extract_gold_answer_from_ground_truth(ground_truth)
                
                if gold_answer:
                    gsm_gold.append({
                        "gold_text": gold_answer,
                        "instruction": obj.get("instruction", ""),
                        "id": obj.get("id", f"sample_{i}"),
                        "original_ground_truth": ground_truth
                    })
                    valid_count += 1
                else:
                    print(f"  警告: 第{i+1}个样本无法提取答案: {ground_truth[:50]}...")
            
            print(f"  成功提取 {valid_count} 个有效答案")
            
        except Exception as e:
            print(f"  ✗ 加载GSM8K数据失败: {e}")
            import traceback
            traceback.print_exc()
            return stats
        
        # 加载预测数据
        try:
            arr = np.load(gsm8k_pred_path, allow_pickle=True)
            print(f"  预测数据形状: {arr.shape}")
            
            # 将预测数据转换为文本列表
            gsm_preds = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        row.append("")
                    else:
                        row.append(str(v))
                gsm_preds.append(row)
            
            # 统计正确率
            min_samples = min(len(gsm_gold), len(gsm_preds))
            print(f"  将对齐 {min_samples} 个样本")
            
            for i in range(min_samples):
                texts = gsm_preds[i]
                final_pred = majority_vote_text(texts) if texts else ""
                is_correct = int(is_gsm8k_correct(final_pred, gsm_gold[i]["gold_text"]))
                
                stats["gsm8k"]["total"] += 1
                stats["all"]["total"] += 1
                if is_correct:
                    stats["gsm8k"]["correct"] += 1
                    stats["all"]["correct"] += 1
            
            print(f"  已处理 {min_samples} 个样本")
            
            # 显示前几个样本的详情
            if min_samples > 0:
                print(f"\n  前3个样本详情:")
                for i in range(min(3, min_samples)):
                    texts = gsm_preds[i]
                    final_pred = majority_vote_text(texts) if texts else ""
                    gold = gsm_gold[i]["gold_text"]
                    is_correct = is_gsm8k_correct(final_pred, gold)
                    print(f"    样本{i+1}: 金标={gold}, 预测={final_pred}, 正确={is_correct}")
                    
        except Exception as e:
            print(f"  ✗ 加载预测数据失败: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print(f"\n[GSM8K] ⚠️ 文件不存在，跳过")
        if not os.path.exists(gsm8k_gold_path):
            print(f"  缺失金标文件: {gsm8k_gold_path}")
        if not os.path.exists(gsm8k_pred_path):
            print(f"  缺失预测文件: {gsm8k_pred_path}")
    
    # 2. 检查MMLU
    if os.path.exists(mmlu_dir):
        print(f"\n[MMLU] 加载数据...")
        print(f"  目录: {mmlu_dir}")
        
        # 加载金标
        gold_map = {}
        files = sorted(glob.glob(os.path.join(mmlu_dir, "*.csv")))
        print(f"  找到 {len(files)} 个CSV文件")
        
        if not files:
            print(f"  ⚠️ 目录中没有CSV文件")
        else:
            for fp in files:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            q = str(row.get("question", "")).strip()
                            lab = str(row.get("label", "")).strip()
                            if q:
                                gold_map[q] = lab
                except Exception as e:
                    print(f"  警告: 读取文件失败 {fp}: {e}")
        
        # 加载预测
        pred_map = {}
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        q = str(row.get("question", "")).strip()
                        pred_field = row.get("prediction")
                        pred_str = "" if pred_field is None else str(pred_field)
                        pred_label = parse_prediction_to_label(pred_str)
                        if q:
                            if q not in pred_map:
                                pred_map[q] = []
                            pred_map[q].append(pred_label)
            except Exception as e:
                print(f"  警告: 读取预测文件失败 {fp}: {e}")
        
        # 统计正确率
        matched_count = 0
        for q, gold_label in gold_map.items():
            pred_labels = pred_map.get(q, [])
            
            if pred_labels:  # 只统计匹配的样本
                matched_count += 1
                voted_label = Counter(pred_labels).most_common(1)[0][0] if pred_labels else ""
                is_correct = int(str(voted_label).strip() == str(gold_label).strip())
                
                stats["mmlu"]["total"] += 1
                stats["all"]["total"] += 1
                if is_correct:
                    stats["mmlu"]["correct"] += 1
                    stats["all"]["correct"] += 1
        
        print(f"  金标问题数: {len(gold_map)}")
        print(f"  成功匹配的问题数: {matched_count} ({matched_count/len(gold_map)*100:.1f}% if gold_map>0 else 0)")
    else:
        print(f"\n[MMLU] ⚠️ 目录不存在，跳过: {mmlu_dir}")
    
    # 3. 输出结果
    print("\n" + "=" * 70)
    print("正确率统计结果")
    print("=" * 70)
    
    for dataset in ["gsm8k", "mmlu", "all"]:
        total = stats[dataset]["total"]
        correct = stats[dataset]["correct"]
        if total > 0:
            accuracy = correct / total
            print(f"{dataset.upper():6s} | 样本数: {total:6d} | 正确数: {correct:6d} | 正确率: {accuracy:.4f} ({accuracy*100:6.2f}%)")
        else:
            print(f"{dataset.upper():6s} | 无数据")
    
    print("=" * 70)
    
    return stats

def analyze_model(gsm8k_gold_path: str, gsm8k_pred_path: str, mmlu_dir: str):
    """
    分析指定模型在两个数据集上的正确率
    
    Args:
        gsm8k_gold_path: GSM8K金标文件路径（JSON数组格式）
        gsm8k_pred_path: GSM8K预测文件路径（.npy格式）
        mmlu_dir: MMLU数据目录路径
    """
    print("正在统计SLM基础正确率...")
    print()
    
    stats = check_slm_accuracy(gsm8k_gold_path, gsm8k_pred_path, mmlu_dir)
    
    # 输出总结
    print("\n📊 总结:")
    if stats["gsm8k"]["total"] > 0:
        acc = stats["gsm8k"]["correct"] / stats["gsm8k"]["total"]
        print(f"  • GSM8K: {acc*100:.2f}% ({stats['gsm8k']['correct']}/{stats['gsm8k']['total']})")
    
    if stats["mmlu"]["total"] > 0:
        acc = stats["mmlu"]["correct"] / stats["mmlu"]["total"]
        print(f"  • MMLU:  {acc*100:.2f}% ({stats['mmlu']['correct']}/{stats['mmlu']['total']})")
    
    if stats["all"]["total"] > 0:
        acc = stats["all"]["correct"] / stats["all"]["total"]
        print(f"  • 总计:  {acc*100:.2f}% ({stats['all']['correct']}/{stats['all']['total']})")
    
    return stats

def check_gsm8k_only(gsm8k_gold_path: str, gsm8k_pred_path: str):
    """只检查GSM8K正确率"""
    print("=" * 70)
    print("GSM8K正确率统计")
    print("=" * 70)
    
    # 使用check_slm_accuracy函数，但传入空的mmlu目录
    stats = check_slm_accuracy(gsm8k_gold_path, gsm8k_pred_path, "")
    
    if stats["gsm8k"]["total"] > 0:
        acc = stats["gsm8k"]["correct"] / stats["gsm8k"]["total"]
        print(f"\n📊 GSM8K结果: {acc*100:.2f}% ({stats['gsm8k']['correct']}/{stats['gsm8k']['total']})")
    else:
        print(f"\n⚠️ 没有处理任何GSM8K样本")
    
    return stats["gsm8k"]["correct"], stats["gsm8k"]["total"]

def main():
    """主函数 - 支持命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="检查SLM在两个数据集上的正确率")
    parser.add_argument("--gsm8k_gold", type=str, default="./dataset/gsm8k/test.json",
                       help="GSM8K金标文件路径（JSON数组格式）")
    parser.add_argument("--gsm8k_pred", type=str, 
                       help="GSM8K预测文件路径（.npy格式，必需）")
    parser.add_argument("--mmlu_dir", type=str, default="",
                       help="MMLU数据目录路径（可选）")
    parser.add_argument("--gsm8k_only", action="store_true",
                       help="只检查GSM8K，忽略MMLU")
    
    args = parser.parse_args()
    
    if args.gsm8k_only or not args.mmlu_dir:
        # 只检查GSM8K
        if not args.gsm8k_pred:
            print("错误: 必须提供 --gsm8k_pred 参数")
            return
        check_gsm8k_only(args.gsm8k_gold, args.gsm8k_pred)
    else:
        # 检查两个数据集
        if not args.gsm8k_pred:
            print("错误: 必须提供 --gsm8k_pred 参数")
            return
        analyze_model(args.gsm8k_gold, args.gsm8k_pred, args.mmlu_dir)

if __name__ == "__main__":
    main()