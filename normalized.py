# -*- coding: UTF-8 -*-
"""
 Author  : Jo
 User    : Timtic
 IDE     : PyCharm
 File    : normalized.py
 Data    : 2024/9/3 上午9:23
 Project : new_idea
Project description:
    用于原始特征预处理
"""
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from dataset import load_data, Data

file_path = r"C:\Users\Timtic\Desktop\next_states.csv"
df = pd.read_csv(file_path)

# print(type(df))
scaler = MinMaxScaler()
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
features = scaler.fit_transform(df)
features = features / features.sum(1, keepdims=True)
print()
