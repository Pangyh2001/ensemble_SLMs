import argparse
from data_split import split_mmlu_data, prepare_gsm8k_data
from trainer import train_all_gates
from evaluator import run_evaluation
from config import EMBEDDING_MODELS, SELECTED_EMBEDDING, GATE_TYPE

def main():
    parser = argparse.ArgumentParser(description="Small Language Model Ensemble Framework")
    parser.add_argument("--mode", default="all", choices=["split", "train", "eval", "all"],
                        help="运行模式：split(划分数据), train(训练), eval(评估), all(全部)")
    parser.add_argument("--task", default="mmlu", choices=["mmlu", "gsm8k", "both"],
                        help="任务类型")
    parser.add_argument("--embedding", default=SELECTED_EMBEDDING, choices=EMBEDDING_MODELS.keys(),
                        help="选择 Embedding 模型: bert, e5-base, e5-large, gte-large")
    parser.add_argument("--gate_type", default=GATE_TYPE, choices=["mlp", "attention", "resnet"],
                        help="Gate网络架构: mlp(简单MLP), attention(多头注意力), resnet(深层残差网络)")
    
    args = parser.parse_args()
    tasks = ["mmlu", "gsm8k"] if args.task == "both" else [args.task]
    
    print("\n" + "="*80)
    print("Small Language Model Ensemble Framework")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Embedding: {args.embedding} (shared, frozen)")
    print(f"Gate Type: {args.gate_type}")
    print("="*80 + "\n")
    
    if args.mode in ["split", "all"]:
        print("\n" + "="*80)
        print("STEP 1: Data Splitting")
        print("="*80)
        for t in tasks:
            if t == "mmlu":
                print("\n[MMLU] Splitting data...")
                split_mmlu_data()
            else:
                print("\n[GSM8K] Preparing data (using original train/test split)...")
                prepare_gsm8k_data()
            
    if args.mode in ["train", "all"]:
        print("\n" + "="*80)
        print("STEP 2: Training Gate Networks")
        print("="*80)
        print(f"Using {args.embedding} embedding (shared across all gates, frozen)")
        print(f"Gate architecture: {args.gate_type}")
        for t in tasks:
            print(f"\n{'='*80}")
            print(f"Training {t.upper()} Gates")
            print(f"{'='*80}")
            train_all_gates(t, embedding_key=args.embedding, gate_type=args.gate_type)
            
    if args.mode in ["eval", "all"]:
        print("\n" + "="*80)
        print("STEP 3: Evaluation")
        print("="*80)
        for t in tasks:
            res = run_evaluation(t, embedding_key=args.embedding, gate_type=args.gate_type)
            
            print(f"\n{'='*80}")
            print(f"Summary for {t.upper()}")
            print(f"{'='*80}")
            
            # 简要总结
            print("\n[Training Models]")
            train_accs = {k.replace('train_model_', ''): v 
                         for k, v in res.items() if k.startswith('train_model_')}
            for model, acc in sorted(train_accs.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {model:<30}: {acc:.4f}")
            
            print("\n[Ensemble Results]")
            ensemble_keys = ['ensemble_threshold', 'ensemble_sampling', 'ensemble_weighted']
            for key in ensemble_keys:
                if key in res:
                    method = key.replace('ensemble_', '')
                    print(f"  {method.capitalize():<30}: {res[key]:.4f}")
    
    print("\n" + "="*80)
    print("All tasks completed!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()