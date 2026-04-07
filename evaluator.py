import torch
import numpy as np
import os
import pickle as pkl
import json
from tqdm import tqdm
from collections import Counter
from config import *
from gate_model import GateNetwork
from embedding_manager import get_embedding_manager
from data_loader import (load_mmlu_single_model, load_gsm8k_single_model, 
                         compute_gsm8k_accuracy, normalize_question)

class EnsembleEvaluator:
    def __init__(self, task_type, device='cuda', embedding_key="bert", gate_type="mlp"):
        self.task_type = task_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.embedding_key = embedding_key
        self.gate_type = gate_type
        
        # 获取共享的embedding manager
        self.emb_manager = get_embedding_manager(embedding_key, device)
        embedding_dim = self.emb_manager.get_encoder_dim()
        
        # 加载训练模型的Gate
        self.gate_models = {}
        if task_type == "mmlu":
            self.train_models = MMLU_TRAIN_MODELS
            self.baseline_models = MMLU_BASELINE_MODELS
        else:
            self.train_models = GSM8K_TRAIN_MODELS
            self.baseline_models = GSM8K_BASELINE_MODELS
        
        for model_name in self.train_models:
            gate = GateNetwork(
                input_dim=embedding_dim,
                gate_type=gate_type,
                hidden_dim=TRAIN_CONFIG["hidden_dim"],
                num_heads=TRAIN_CONFIG.get("num_heads", 8),
                num_blocks=TRAIN_CONFIG.get("num_blocks", 4),
                dropout=TRAIN_CONFIG["dropout"]
            ).to(self.device)
            
            model_path = os.path.join(GATE_DIR, f"{model_name}_{task_type}_{gate_type}.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                gate.load_state_dict(checkpoint['model_state_dict'])
                gate.eval()
                self.gate_models[model_name] = gate
            else:
                print(f"Warning: Gate model not found for {model_name} at {model_path}")

    def get_gate_scores(self, embedding):
        """
        获取所有Gate的分数
        Args:
            embedding: (embed_dim,) 单个问题的embedding
        Returns:
            scores: dict {model_name: score}
        """
        scores = {}
        embedding_batch = embedding.unsqueeze(0).to(self.device)  # (1, embed_dim)
        
        with torch.no_grad():
            for model_name, gate_model in self.gate_models.items():
                s = gate_model(embedding_batch)
                scores[model_name] = s.item()
        return scores

    def evaluate_single_models(self, test_data, test_embeddings):
        """评估所有单模型（训练模型 + baseline模型）"""
        results = {}
        questions = test_data['questions']
        labels = test_data['labels']
        
        # 评估训练模型
        print("\n=== Evaluating Training Models ===")
        if self.task_type == "mmlu":
            model_predictions = test_data['data']
            for model_name in self.train_models:
                if model_name not in model_predictions:
                    continue
                preds = model_predictions[model_name]
                acc = self._compute_accuracy_mmlu(preds, labels)
                results[f"train_model_{model_name}"] = acc
                print(f"  {model_name}: {acc:.4f}")
        else:
            # GSM8K: 使用原始预测
            for i, model_name in enumerate(self.train_models):
                raw_predictions = test_data['raw_predictions'][:, i, :]
                acc = self._compute_accuracy_gsm8k_batch(raw_predictions, labels)
                results[f"train_model_{model_name}"] = acc
                print(f"  {model_name}: {acc:.4f}")
        
        # 评估baseline模型
        print("\n=== Evaluating Baseline Models ===")
        for model_name in self.baseline_models:
            try:
                if self.task_type == "mmlu":
                    # 加载完整baseline数据
                    preds, baseline_questions, baseline_labels = load_mmlu_single_model(model_name)
                    
                    # MMLU需要通过文本对齐（因为可能有不同的subject组合）
                    baseline_q_to_idx = {normalize_question(q): i for i, q in enumerate(baseline_questions)}
                    
                    aligned_preds = []
                    valid_count = 0
                    for q in questions:
                        norm_q = normalize_question(q)
                        if norm_q in baseline_q_to_idx:
                            aligned_preds.append(preds[baseline_q_to_idx[norm_q]])
                            valid_count += 1
                        else:
                            aligned_preds.append(np.zeros(4))
                    
                    if valid_count < len(questions) * 0.5:
                        print(f"  {model_name}: Low alignment rate ({valid_count}/{len(questions)}), skipping")
                        continue
                    
                    aligned_preds = np.array(aligned_preds)
                    acc = self._compute_accuracy_mmlu(aligned_preds, labels)
                    
                else:  # GSM8K
                    input_dir = os.path.join(CUR_DIR, DATA_DIR, "gsm8k")
                    
                    # GSM8K: 预测和问题是按索引对应的，不需要文本匹配
                    # 直接加载并按索引对齐
                    preds_full, baseline_questions, baseline_labels = load_gsm8k_single_model(
                        model_name, input_dir, 
                        num_samples=None,
                        dataset_name="test"
                    )
                    
                    # 确保长度一致
                    min_len = min(len(preds_full), len(questions))
                    aligned_preds = preds_full[:min_len]
                    
                    if min_len < len(questions) * 0.9:
                        print(f"  {model_name}: Warning - only {min_len}/{len(questions)} samples available")
                    
                    # 只评估对齐的部分
                    acc = self._compute_accuracy_gsm8k_batch(aligned_preds, labels[:min_len])
                
                results[f"baseline_{model_name}"] = acc
                print(f"  {model_name}: {acc:.4f}")
                
            except FileNotFoundError as e:
                print(f"  {model_name}: Data not found - {e}")
            except Exception as e:
                print(f"  {model_name}: Error - {e}")
                import traceback
                traceback.print_exc()
        
        return results

    def run_evaluation_suite(self, test_data, test_embeddings):
        """运行完整评估：单模型 + 集成策略"""
        questions = test_data['questions']
        labels = test_data['labels']
        
        # 1. 评估所有单模型
        single_model_results = self.evaluate_single_models(test_data, test_embeddings)
        
        # 2. 评估集成策略
        print("\n=== Evaluating Ensemble Strategies ===")
        np.random.seed(EVAL_CONFIG["seed"])
        
        detailed_logs = []
        ensemble_metrics = {
            "ensemble_threshold": 0, 
            "baseline_threshold_random": 0,
            "ensemble_sampling": 0, 
            "baseline_sampling_random": 0,
            "ensemble_weighted": 0, 
            "baseline_weighted_uniform": 0
        }
        
        total = len(questions)
        
        for i in tqdm(range(total), desc=f"Evaluating {self.task_type.upper()}"):
            true_label = labels[i]
            
            # 获取该问题的embedding并计算gate分数
            embedding = test_embeddings[i]
            gate_scores_map = self.get_gate_scores(embedding)
            gate_scores_vec = np.array([gate_scores_map[m] for m in self.train_models])
            
            # Strategy 1: Threshold
            threshold = EVAL_CONFIG.get("threshold", 0.6)
            selected_thr = [m for m, s in gate_scores_map.items() if s > threshold]
            if not selected_thr:
                max_model = max(gate_scores_map, key=gate_scores_map.get)
                selected_thr = [max_model]
            
            ans_thr = self._predict_ensemble(i, selected_thr, test_data)
            ensemble_metrics["ensemble_threshold"] += (
                1 if self._check_correct(ans_thr, true_label) else 0
            )
            
            k_thr = len(selected_thr)
            rand_thr = list(np.random.choice(self.train_models, k_thr, replace=False))
            ans_thr_rand = self._predict_ensemble(i, rand_thr, test_data)
            ensemble_metrics["baseline_threshold_random"] += (
                1 if self._check_correct(ans_thr_rand, true_label) else 0
            )
            
            # Strategy 2: Sampling
            selected_samp = [m for m in self.train_models 
                           if np.random.random() < gate_scores_map[m]]
            if not selected_samp:
                max_model = max(gate_scores_map, key=gate_scores_map.get)
                selected_samp = [max_model]
            
            ans_samp = self._predict_ensemble(i, selected_samp, test_data)
            ensemble_metrics["ensemble_sampling"] += (
                1 if self._check_correct(ans_samp, true_label) else 0
            )
            
            k_samp = len(selected_samp)
            rand_samp = list(np.random.choice(self.train_models, k_samp, replace=False))
            ans_samp_rand = self._predict_ensemble(i, rand_samp, test_data)
            ensemble_metrics["baseline_sampling_random"] += (
                1 if self._check_correct(ans_samp_rand, true_label) else 0
            )
            
            # Strategy 3: Weighted
            exp_scores = np.exp(gate_scores_vec)
            weights = exp_scores / np.sum(exp_scores)
            
            ans_weighted = self._predict_weighted(i, weights, test_data)
            ensemble_metrics["ensemble_weighted"] += (
                1 if self._check_correct(ans_weighted, true_label) else 0
            )
            
            uniform_weights = np.ones_like(weights) / len(weights)
            ans_weighted_uni = self._predict_weighted(i, uniform_weights, test_data)
            ensemble_metrics["baseline_weighted_uniform"] += (
                1 if self._check_correct(ans_weighted_uni, true_label) else 0
            )
            
            # 记录日志
            detailed_logs.append({
                "id": i,
                "question": questions[i],
                "true_label": str(true_label),
                "gate_scores": gate_scores_map,
                "threshold": {
                    "selected": selected_thr,
                    "prediction": str(ans_thr),
                    "random_prediction": str(ans_thr_rand)
                },
                "sampling": {
                    "selected": selected_samp,
                    "prediction": str(ans_samp),
                    "random_prediction": str(ans_samp_rand)
                },
                "weighted": {
                    "prediction": str(ans_weighted),
                    "uniform_prediction": str(ans_weighted_uni)
                }
            })
        
        # 计算最终准确率
        ensemble_results = {k: v/total for k, v in ensemble_metrics.items()}
        
        # 合并所有结果
        final_results = {**single_model_results, **ensemble_results}
        
        # 保存详细日志
        os.makedirs(os.path.join(RESULT_DIR, self.task_type), exist_ok=True)
        log_path = os.path.join(RESULT_DIR, self.task_type, f"detailed_logs_{self.gate_type}.json")
        with open(log_path, 'w') as f:
            json.dump(detailed_logs, f, indent=2)
        
        return final_results

    def _predict_ensemble(self, idx, selected_models, test_data):
        """集成预测（硬投票）"""
        if self.task_type == "mmlu":
            total_logits = np.zeros(4)
            model_predictions = test_data['data']
            for m in selected_models:
                total_logits += np.exp(model_predictions[m][idx])
            return ['A', 'B', 'C', 'D'][np.argmax(total_logits)]
        else:
            # GSM8K: 多数投票
            raw_preds = test_data['raw_predictions']
            answers = []
            for m in selected_models:
                m_idx = self.train_models.index(m)
                model_runs = raw_preds[idx, m_idx, :]
                valid_answers = [a for a in model_runs if not np.isnan(a)]
                if valid_answers:
                    most_common = Counter(valid_answers).most_common(1)[0][0]
                    answers.append(most_common)
            
            if not answers:
                return np.nan
            return Counter(answers).most_common(1)[0][0]

    def _predict_weighted(self, idx, weights, test_data):
        """加权预测"""
        if self.task_type == "mmlu":
            total_conf = np.zeros(4)
            model_predictions = test_data['data']
            for m_idx, m_name in enumerate(self.train_models):
                w = weights[m_idx]
                total_conf += w * np.exp(model_predictions[m_name][idx])
            return ['A', 'B', 'C', 'D'][np.argmax(total_conf)]
        else:
            # GSM8K: 加权投票
            raw_preds = test_data['raw_predictions']
            answer_weights = {}
            
            for m_idx, m_name in enumerate(self.train_models):
                w = weights[m_idx]
                model_runs = raw_preds[idx, m_idx, :]
                
                valid_answers = [a for a in model_runs if not np.isnan(a)]
                answer_counts = Counter(valid_answers)
                
                for ans, count in answer_counts.items():
                    if ans not in answer_weights:
                        answer_weights[ans] = 0
                    answer_weights[ans] += w * (count / len(model_runs))
            
            if not answer_weights:
                return np.nan
            return max(answer_weights, key=answer_weights.get)

    def _check_correct(self, pred, label):
        """检查预测是否正确"""
        if self.task_type == "mmlu":
            return pred == label
        else:
            try:
                return abs(float(pred) - float(label)) < 1e-4
            except:
                return False

    def _compute_accuracy_mmlu(self, predictions, labels):
        """计算MMLU准确率"""
        correct = 0
        for pred, label in zip(predictions, labels):
            pred_class = ['A', 'B', 'C', 'D'][np.argmax(np.exp(pred))]
            if pred_class == label:
                correct += 1
        return correct / len(labels)

    def _compute_accuracy_gsm8k_batch(self, predictions, labels):
        """批量计算GSM8K准确率"""
        total_acc = 0
        for pred_runs, label in zip(predictions, labels):
            total_acc += compute_gsm8k_accuracy(pred_runs, label)
        return total_acc / len(labels)


def run_evaluation(task_type="mmlu", embedding_key="bert", gate_type="mlp"):
    """运行评估"""
    print(f"\n{'='*80}")
    print(f"Running Evaluation for {task_type.upper()}")
    print(f"Embedding: {embedding_key} (shared, frozen)")
    print(f"Gate Type: {gate_type}")
    print(f"{'='*80}\n")
    
    # 1. 获取共享的embedding manager并加载预计算的embedding
    emb_manager = get_embedding_manager(embedding_key, device='cuda')
    _, test_embeddings = emb_manager.precompute_embeddings(task_type, force_recompute=False)
    
    # 2. 加载测试数据
    split_dir = os.path.join(CUR_DIR, DATA_DIR, "splits")
    with open(os.path.join(split_dir, f"{task_type}_test.pkl"), "rb") as f:
        test_data = pkl.load(f)
    
    # 3. 运行评估
    evaluator = EnsembleEvaluator(task_type, embedding_key=embedding_key, gate_type=gate_type)
    results = evaluator.run_evaluation_suite(test_data, test_embeddings)
    
    # 4. 保存结果
    os.makedirs(os.path.join(RESULT_DIR, task_type), exist_ok=True)
    res_path = os.path.join(RESULT_DIR, task_type, f"final_results_{gate_type}.json")
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 5. 打印格式化结果
    print("\n" + "="*80)
    print(f"FINAL RESULTS - {task_type.upper()} ({gate_type.upper()} Gate)")
    print("="*80)
    
    print("\n--- Training Models (participated in ensemble) ---")
    for k, v in sorted(results.items()):
        if k.startswith("train_model_"):
            print(f"  {k.replace('train_model_', ''):<30}: {v:.4f}")
    
    print("\n--- Baseline Models (for comparison only) ---")
    for k, v in sorted(results.items()):
        if k.startswith("baseline_"):
            print(f"  {k.replace('baseline_', ''):<30}: {v:.4f}")
    
    print("\n--- Ensemble Strategies ---")
    strategies = ["threshold", "sampling", "weighted"]
    for strategy in strategies:
        ensemble_key = f"ensemble_{strategy}"
        baseline_key = [k for k in results.keys() if strategy in k and "baseline" in k][0]
        print(f"\n  {strategy.upper()}:")
        print(f"    Ensemble ({strategy})             : {results.get(ensemble_key, 0):.4f}")
        print(f"    Baseline (random)                 : {results.get(baseline_key, 0):.4f}")
    
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mmlu", choices=["mmlu", "gsm8k"])
    parser.add_argument("--embedding", default="bert")
    parser.add_argument("--gate_type", default="mlp", choices=["mlp", "attention", "resnet"])
    args = parser.parse_args()
    run_evaluation(args.task, args.embedding, args.gate_type)