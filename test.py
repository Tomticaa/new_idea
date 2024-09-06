from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 示例一维数据
data = np.array([100, 8, 50, 88]).reshape(-1, 1)  # 将一维数组转换为二维数组，每行一个样本

# 创建MinMaxScaler对象，设置特征范围为0到1
scaler = MinMaxScaler(feature_range=(0, 1))

# 拟合数据并转换
scaled_data = scaler.fit_transform(data)
print(data)
print("原始数据:\n", data.flatten())  # 使用flatten将数据重新展平为一维数组显示
print("归一化后的数据:\n", scaled_data.flatten())
