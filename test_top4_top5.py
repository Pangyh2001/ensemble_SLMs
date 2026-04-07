# evaluate_topk_ensemble.py
import torch
import numpy as np
import os
import pickle as pkl
import csv
from tqdm import tqdm
from collections import Counter
import random

from config import *
from gate_model import GateNetwork
from embedding_manager import get_embedding_manager
from data_loader import load_mmlu_single_model, load_gsm8k_single_model, compute_gsm8k_accuracy
from evaluator import normalize_question

class TopKEnsembleEvaluator:
    def __init__(self, task_type, seed, device='cuda', embedding_key="bert", gate_type="mlp"):
        self.task_type = task_type
        self.seed = seed
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.embedding_key = embedding_key
        self.gate_type = gate_type
        
        # 设置随机种子
        self._set_seed(seed)
        
        # 获取共享的embedding manager
        self.emb_manager = get_embedding_manager(embedding_key, device)
        embedding_dim = self.emb_manager.get_encoder_dim()
        
        # 确定模型列表和门控路径
        if task_type == "mmlu":
            self.train_models = MMLU_TRAIN_MODELS
            # MMLU门控从gate_mmlu_main读取
            self.gate_dir = "gate_mmlu_main"
        else:
            self.train_models = GSM8K_TRAIN_MODELS
            # GSM8K门控从gate_gsm8k读取
            self.gate_dir = "gate_gsm8k"
        
        # 加载所有门控模型（使用带种子的文件名）
        self.gate_models = {}
        print(f"\nLoading {task_type.upper()} gate models (seed={seed}) from {self.gate_dir}/...")
        
        for model_name in self.train_models:
            gate = GateNetwork(
                input_dim=embedding_dim,
                gate_type=gate_type,
                hidden_dim=TRAIN_CONFIG["hidden_dim"],
                num_heads=TRAIN_CONFIG.get("num_heads", 8),
                num_blocks=TRAIN_CONFIG.get("num_blocks", 4),
                dropout=TRAIN_CONFIG["dropout"]
            ).to(self.device)
            
            # 新的命名格式: {model_name}_{task_type}_{gate_type}_seed{seed}.pt
            model_path = os.path.join(self.gate_dir, f"{model_name}_{task_type}_{gate_type}_seed{seed}.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                gate.load_state_dict(checkpoint['model_state_dict'])
                gate.eval()
                self.gate_models[model_name] = gate
                print(f"  ✓ Loaded {model_name}")
            else:
                print(f"  ✗ Warning: Gate not found for {model_name} at {model_path}")
        
        print(f"  Loaded {len(self.gate_models)}/{len(self.train_models)} gates\n")
    
    def _set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def get_gate_scores(self, embedding):
        """获取所有Gate的分数"""
        scores = {}
        embedding_batch = embedding.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for model_name, gate_model in self.gate_models.items():
                s = gate_model(embedding_batch)
                scores[model_name] = s.item()
        return scores
    
    def predict_topk_majority_vote(self, idx, top_k, test_data, gate_scores):
        """
        使用门控分数最高的top_k个模型进行多数投票
        
        Args:
            idx: 问题索引
            top_k: 选择的模型数量 (4或5)
            test_data: 测试数据
            gate_scores: 预计算的门控分数
        
        Returns:
            prediction: 预测结果
        """
        # 按门控分数降序排序，选择top_k个模型
        sorted_models = sorted(gate_scores.items(), key=lambda x: x[1], reverse=True)
        selected_models = [m[0] for m in sorted_models[:top_k]]
        
        if self.task_type == "mmlu":
            # MMLU: 硬投票
            votes = []
            model_predictions = test_data['data']
            for model_name in selected_models:
                pred_logits = model_predictions[model_name][idx]
                pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred_logits))]
                votes.append(pred_class)
            
            # 多数投票
            vote_counts = Counter(votes)
            return vote_counts.most_common(1)[0][0]
        
        else:
            # GSM8K: 多数投票（基于10次采样的最频繁答案）
            raw_preds = test_data['raw_predictions']
            all_answers = []
            
            for model_name in selected_models:
                m_idx = self.train_models.index(model_name)
                model_runs = raw_preds[idx, m_idx, :]
                valid_answers = [a for a in model_runs if not np.isnan(a)]
                if valid_answers:
                    most_common = Counter(valid_answers).most_common(1)[0][0]
                    all_answers.append(most_common)
            
            if not all_answers:
                return np.nan
            return Counter(all_answers).most_common(1)[0][0]
    
    def evaluate_topk(self, test_data, test_embeddings, top_k):
        """
        评估top-k集成的准确率
        """
        questions = test_data['questions']
        labels = test_data['labels']
        total = len(questions)
        correct = 0
        
        print(f"\nEvaluating Top-{top_k} Ensemble on {self.task_type.upper()} (seed={self.seed})...")
        
        for i in tqdm(range(total), desc=f"Top-{top_k}"):
            true_label = labels[i]
            
            # 获取该问题的embedding并计算门控分数
            embedding = test_embeddings[i]
            gate_scores = self.get_gate_scores(embedding)
            
            # Top-k投票预测
            prediction = self.predict_topk_majority_vote(i, top_k, test_data, gate_scores)
            
            # 检查正确性
            if self._check_correct(prediction, true_label):
                correct += 1
        
        accuracy = correct / total
        print(f"  Top-{top_k} Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return accuracy
    
    def _check_correct(self, pred, label):
        """检查预测是否正确"""
        if self.task_type == "mmlu":
            return pred == label
        else:
            try:
                return abs(float(pred) - float(label)) < 1e-4
            except:
                return False


def load_test_data(task_type):
    """加载测试数据"""
    split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
    
    with open(os.path.join(split_dir, f"{task_type}_test.pkl"), "rb") as f:
        test_data = pkl.load(f)
    
    return test_data


def load_test_embeddings(task_type, embedding_key="bert"):
    """加载预计算的测试集embeddings"""
    emb_manager = get_embedding_manager(embedding_key, device='cuda')
    _, test_embeddings = emb_manager.precompute_embeddings(task_type, force_recompute=False)
    return test_embeddings


def run_single_experiment(task_type, seed, test_data, test_embeddings, top_k_values):
    """
    运行单个种子的实验
    """
    embedding_key = SELECTED_EMBEDDING
    gate_type = GATE_TYPE
    
    # 创建评估器
    evaluator = TopKEnsembleEvaluator(task_type, seed, embedding_key=embedding_key, gate_type=gate_type)
    
    # 评估top-4和top-5
    results = {}
    for top_k in top_k_values:
        acc = evaluator.evaluate_topk(test_data, test_embeddings, top_k)
        results[f"top_{top_k}"] = acc
    
    return results


def compute_statistics(results_list):
    """
    计算多次实验的平均值和标准差
    
    Args:
        results_list: List of dict, 每个dict包含不同种子的结果
                    例如: [{'top_4': 0.8, 'top_5': 0.79}, ...]
    
    Returns:
        stats: dict, 包含每个指标的mean和std
    """
    if not results_list:
        return {}
    
    # 获取所有指标名称
    metrics = results_list[0].keys()
    
    stats = {}
    for metric in metrics:
        values = [r[metric] for r in results_list if metric in r]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            stats[metric] = {'mean': mean_val, 'std': std_val}
    
    return stats


def main():
    print("\n" + "="*80)
    print("Top-K Ensemble Evaluation with Multiple Seeds")
    print("="*80)
    print("Using pre-trained gate models:")
    print("  - MMLU gates from: gate_mmlu_main/")
    print("  - GSM8K gates from: gate_gsm8k/")
    print("  - File format: {model_name}_{task}_{gate_type}_seed{seed}.pt")
    print("="*80 + "\n")
    
    # 配置
    seeds = [0, 42, 123]  # 三个随机种子
    top_k_values = [4, 5]
    
    # 存储所有结果
    all_results = {
        "mmlu": [],
        "gsm8k": []
    }
    
    # 先加载测试数据和embeddings（不依赖种子）
    print("Loading test data and embeddings...")
    mmlu_test_data = load_test_data("mmlu")
    mmlu_test_embeddings = load_test_embeddings("mmlu")
    
    gsm8k_test_data = load_test_data("gsm8k")
    gsm8k_test_embeddings = load_test_embeddings("gsm8k")
    print("✓ Data loaded\n")
    
    # 运行MMLU实验（多个种子）
    print("\n" + "="*80)
    print("MMLU EXPERIMENTS")
    print("="*80)
    
    for seed in seeds:
        print(f"\n--- Running MMLU with seed={seed} ---")
        try:
            result = run_single_experiment("mmlu", seed, mmlu_test_data, mmlu_test_embeddings, top_k_values)
            all_results["mmlu"].append(result)
            print(f"✓ MMLU seed={seed} completed: {result}")
        except Exception as e:
            print(f"✗ Error in MMLU seed={seed}: {e}")
            import traceback
            traceback.print_exc()
            all_results["mmlu"].append({f"top_{k}": None for k in top_k_values})
    
    # 运行GSM8K实验（多个种子）
    print("\n" + "="*80)
    print("GSM8K EXPERIMENTS")
    print("="*80)
    
    for seed in seeds:
        print(f"\n--- Running GSM8K with seed={seed} ---")
        try:
            result = run_single_experiment("gsm8k", seed, gsm8k_test_data, gsm8k_test_embeddings, top_k_values)
            all_results["gsm8k"].append(result)
            print(f"✓ GSM8K seed={seed} completed: {result}")
        except Exception as e:
            print(f"✗ Error in GSM8K seed={seed}: {e}")
            import traceback
            traceback.print_exc()
            all_results["gsm8k"].append({f"top_{k}": None for k in top_k_values})
    
    # 计算统计信息
    mmlu_stats = compute_statistics(all_results["mmlu"])
    gsm8k_stats = compute_statistics(all_results["gsm8k"])
    
    # 保存结果到CSV
    csv_path = os.path.join(CUR_DIR, "result_top4_top5.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow(['Task', 'Method', 'Seed0', 'Seed42', 'Seed123', 'Mean', 'Std'])
        
        # 写入MMLU结果
        if all_results["mmlu"]:
            for top_k in top_k_values:
                method = f'Top-{top_k} Majority Vote'
                seed_values = []
                for seed_idx, seed_result in enumerate(all_results["mmlu"]):
                    acc = seed_result.get(f"top_{top_k}", None)
                    seed_values.append(f'{acc:.4f}' if acc is not None else 'N/A')
                
                mean_val = mmlu_stats.get(f"top_{top_k}", {}).get('mean', None)
                std_val = mmlu_stats.get(f"top_{top_k}", {}).get('std', None)
                
                writer.writerow([
                    'MMLU', method, 
                    seed_values[0], seed_values[1], seed_values[2],
                    f'{mean_val:.4f}' if mean_val is not None else 'N/A',
                    f'{std_val:.4f}' if std_val is not None else 'N/A'
                ])
        
        # 写入GSM8K结果
        if all_results["gsm8k"]:
            for top_k in top_k_values:
                method = f'Top-{top_k} Majority Vote'
                seed_values = []
                for seed_idx, seed_result in enumerate(all_results["gsm8k"]):
                    acc = seed_result.get(f"top_{top_k}", None)
                    seed_values.append(f'{acc:.4f}' if acc is not None else 'N/A')
                
                mean_val = gsm8k_stats.get(f"top_{top_k}", {}).get('mean', None)
                std_val = gsm8k_stats.get(f"top_{top_k}", {}).get('std', None)
                
                writer.writerow([
                    'GSM8K', method,
                    seed_values[0], seed_values[1], seed_values[2],
                    f'{mean_val:.4f}' if mean_val is not None else 'N/A',
                    f'{std_val:.4f}' if std_val is not None else 'N/A'
                ])
        
        # 添加总结行
        writer.writerow([])
        writer.writerow(['Summary', '', '', '', '', '', ''])
        
        # 计算平均准确率（跨top-k）
        if mmlu_stats:
            mmlu_accs = [mmlu_stats[f"top_{k}"]['mean'] for k in top_k_values if f"top_{k}" in mmlu_stats]
            if mmlu_accs:
                mmlu_avg = np.mean(mmlu_accs)
                writer.writerow(['MMLU', f'Average over Top-{top_k_values[0]} & Top-{top_k_values[1]}', 
                                '', '', '', f'{mmlu_avg:.4f}', ''])
        
        if gsm8k_stats:
            gsm8k_accs = [gsm8k_stats[f"top_{k}"]['mean'] for k in top_k_values if f"top_{k}" in gsm8k_stats]
            if gsm8k_accs:
                gsm8k_avg = np.mean(gsm8k_accs)
                writer.writerow(['GSM8K', f'Average over Top-{top_k_values[0]} & Top-{top_k_values[1]}', 
                                '', '', '', f'{gsm8k_avg:.4f}', ''])
    
    # 打印最终结果
    print("\n" + "="*80)
    print("FINAL RESULTS (Summary)")
    print("="*80)
    print(f"\nResults saved to: {csv_path}\n")
    
    print("MMLU Results:")
    if mmlu_stats:
        for top_k in top_k_values:
            mean_val = mmlu_stats[f"top_{top_k}"]['mean']
            std_val = mmlu_stats[f"top_{top_k}"]['std']
            print(f"  Top-{top_k} Majority Vote: {mean_val:.4f} ± {std_val:.4f}")
    else:
        print("  No results")
    
    print("\nGSM8K Results:")
    if gsm8k_stats:
        for top_k in top_k_values:
            mean_val = gsm8k_stats[f"top_{top_k}"]['mean']
            std_val = gsm8k_stats[f"top_{top_k}"]['std']
            print(f"  Top-{top_k} Majority Vote: {mean_val:.4f} ± {std_val:.4f}")
    else:
        print("  No results")
    
    # 打印各种子详细结果
    print("\n" + "-"*80)
    print("Detailed Results by Seed:")
    print("-"*80)
    
    print("\nMMLU:")
    for i, seed in enumerate(seeds):
        if i < len(all_results["mmlu"]):
            result = all_results["mmlu"][i]
            print(f"  Seed {seed}: Top-4={result.get('top_4', 'N/A'):.4f}, Top-5={result.get('top_5', 'N/A'):.4f}")
    
    print("\nGSM8K:")
    for i, seed in enumerate(seeds):
        if i < len(all_results["gsm8k"]):
            result = all_results["gsm8k"][i]
            print(f"  Seed {seed}: Top-4={result.get('top_4', 'N/A'):.4f}, Top-5={result.get('top_5', 'N/A'):.4f}")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()