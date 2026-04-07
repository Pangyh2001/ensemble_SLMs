#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from pprint import pprint

PKL_PATH = "/data2/pyh/ensembleLLM/Claude_code/data_generation/easy/gsm8k/deepseek-llm-7b-chat/test/run_1318_outputs.pkl"
NPY_PATH = "/data2/pyh/ensembleLLM/Claude_code/data_generation/easy/gsm8k/deepseek-llm-7b-chat/test/run_1318_predictions.npy"

# 1. 读取文件
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

pred = np.load(NPY_PATH)

print("===== GSM8K run_1318 数据统计 =====")
print("data 类型:", type(data))
print("prediction 形状:", pred.shape)

# 2. 如果是 list，打印长度
if isinstance(data, list):
    print("样本数量:", len(data))
else:
    print("data 不是 list，无法直接认为是一条条样本")

print("\n===== 第 1 条样本的原始结构 =====")
first = data[0]
print("data[0] 类型:", type(first))

# 如果还是 list/tuple，再把里面每个元素的类型打出来
if isinstance(first, (list, tuple)):
    print("data[0] 长度:", len(first))
    for i, elem in enumerate(first):
        print(f"  data[0][{i}] 类型: {type(elem)}")
    print("\ndata[0] 内容预览（pprint，截断显示）:")
    pprint(first, depth=3, width=120)
else:
    # 不是 list/tuple，就直接 pprint
    pprint(first, depth=3, width=120)

print("\n===== 第 1 条样本对应的 prediction 向量 =====")
print(pred[0])
