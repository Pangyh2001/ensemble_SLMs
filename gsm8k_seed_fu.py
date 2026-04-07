# run_gsm8k_experiment.py
#!/usr/bin/env python3
"""
运行GSM8K多seed多阈值实验的主脚本
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_experiment():
    """运行完整的GSM8K实验"""
    
    # 创建日志目录
    log_dir = "gsm8k_experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    print(f"Starting GSM8K Multi-Seed Multi-Threshold Experiment")
    print(f"Log file: {log_file}")
    print("-" * 80)
    
    # 运行实验
    try:
        # 使用subprocess运行，这样可以将输出同时保存到文件和显示在终端
        with open(log_file, 'w') as log_f:
            # 写入启动信息
            log_f.write(f"GSM8K Experiment started at {datetime.now()}\n")
            log_f.write("=" * 80 + "\n")
            
            # 运行测试脚本
            result = subprocess.run(
                [sys.executable, "test_gsm8k_multiseed.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # 写入输出
            log_f.write(result.stdout)
            
            # 同时打印到终端
            print(result.stdout)
            
            if result.returncode != 0:
                print(f"\nERROR: Experiment failed with return code {result.returncode}")
                log_f.write(f"\nERROR: Experiment failed with return code {result.returncode}\n")
            else:
                print(f"\n✓ Experiment completed successfully!")
                log_f.write(f"\n✓ Experiment completed successfully at {datetime.now()}\n")
    
    except Exception as e:
        print(f"Exception occurred: {e}")
        with open(log_file, 'a') as log_f:
            log_f.write(f"\nException: {e}\n")
    
    print("-" * 80)
    print(f"Experiment completed. Check {log_file} for details.")
    
    # 检查结果文件
    if os.path.exists("gsm8k_results.csv"):
        print(f"\nResults saved to: gsm8k_results.csv")
        
        # 显示CSV的头部
        import pandas as pd
        try:
            df = pd.read_csv("gsm8k_results.csv")
            print("\nResults Preview:")
            print(df.head())
        except:
            print("Could not preview results.")
    else:
        print("\nWARNING: Results file not created!")

if __name__ == "__main__":
    run_experiment()