# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：dataset.py
Date    ：2024/7/18 下午3:17 
Project ：new_model 
Project Description：
    重写 GraphSage-dataset 函数，要求使用原始数据集并不使用类定义方便后续调用；
TODO： 要求返回连续稠密的特征矩阵使得更利于dqn模型训练先添加扰动再进行归一化
TODO： 把 Katz 计算集成到数据处理函数中




"""
import torch
import numpy as np
import pickle as pkl
import os.path as osp
from collections import namedtuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler

Data = namedtuple('Data', ['adj', 'features', 'labels', 'idx_train', 'idx_val', 'idx_test', 'Katz_core'])


def encode_onehot(labels):  # 将cora.content最后一列字符串数组输入，返回numpy形式onehot
    classes = set(labels)  # 集合包含七类元素
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  # 创建类别到 one-hot 向量的映射
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)  # 将标签列表转换为 one-hot 编码数组
    return labels_onehot


def Katz_centrality(adj, katz_alpha=0.05):  # 计算 Katz 中心性
    adjacency_matrix = np.zeros((len(adj), len(adj)), dtype=int)  # 还原邻接矩阵
    for node, neighbors in adj.items():
        for neighbor in neighbors:
            adjacency_matrix[node, neighbor] = 1
    eigenvalues, _ = np.linalg.eig(adjacency_matrix)
    lamda = np.max(eigenvalues)
    print("max_eigenvalues : ", lamda)
    assert katz_alpha < 1 / lamda  # 保证收敛条件 ：衰减因子小于  1 / 最大特征值
    beta = 1.0  # 基本中心性
    n = adjacency_matrix.shape[0]
    I = np.eye(n)
    e = np.ones(n)
    score = np.linalg.solve(I - katz_alpha * adjacency_matrix, beta * e)
    np.set_printoptions(suppress=True)
    return score


def Feature_preprocessing(node_feature, noise_level=0.01):  # 原始特征进行数据预处理
    node_feature = torch.from_numpy(node_feature)
    node_feature = node_feature + noise_level * torch.randn_like(node_feature)  # 添加高斯扰动
    node_feature = node_feature.numpy()

    # scaler = MinMaxScaler(feature_range=(-1, 1))  # 节点值缩放  TODO: 导致分类准确率急剧降低
    # node_feature = scaler.fit_transform(node_feature)
    node_feature = node_feature / node_feature.sum(1, keepdims=True)  # 归一化
    return node_feature


def random_mask(num_node, train_ratio=0.2, val_ratio=0.2, test_ratio=0.3):  # 用于创建随机掩码进行数据集护划分
    indices = np.arange(num_node)  # 值为 2708 的数组：[0 1 2 3 4 5 6 .... 2707]
    np.random.seed(64)
    np.random.shuffle(indices)  # 打乱数组中元素的顺序
    train_end = int(num_node * train_ratio)
    val_end = int(num_node * val_ratio)
    test_end = int(num_node * test_ratio)
    train_indices = indices[:train_end]  # 取乱序前140个[2 0 5 3 9...]
    val_indices = indices[train_end:train_end + val_end]
    test_indices = indices[train_end + val_end:train_end + val_end + test_end]

    return train_indices, val_indices, test_indices  # 返回划分过后的集合，每个数组中的元素为节点的序号


def create_adjacency_list(edges):
    adjacency_list = {}
    for source, target in edges:
        if source not in adjacency_list:
            adjacency_list[source] = []
        if target not in adjacency_list:
            adjacency_list[target] = []
        adjacency_list[source].append(target)
        # 如果是无向图，需要添加以下行：
        adjacency_list[target].append(source)
    return adjacency_list


def process_data(path="./cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))  # 使用np方法将数据读取为字符串形式,将其命名为：索引_特征——标签
    features = idx_features_labels[:, 1:-1]
    features = np.array(features).astype(np.float32)  # 将特征矩阵读取为float32的numpy
    labels = encode_onehot(idx_features_labels[:, -1])  # 使用onehot函数对最后一列标签进行编码处理
    features = Feature_preprocessing(features)  # TODO:将特征矩阵转化成连续稠密的矩阵
    # 构建图的邻接矩阵
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 读取文章id
    idx_map = {j: i for i, j in enumerate(
        idx)}  # 创建索引映射字典：{31336: 0, 1061127: 1, 1106406: 2, 13195: 3, 37879: 4, 1126012: 5, 1107140: 6, 1102850: 7, 31349: 8, 1106418: 9}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 读取引用关系
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
        edges_unordered.shape)  # 将引用关系中的论文id通过字典映射到(0, 2707)
    # 创建邻接列表
    adj_list = create_adjacency_list(edges)
    labels = np.where(labels)[1]

    # 数据集随机划分
    idx_train, idx_val, idx_test = random_mask(num_node=features.shape[0], train_ratio=0.05, val_ratio=0.05, test_ratio=0.36)
    Katz_core = Katz_centrality(adj_list, katz_alpha=0.05)
    # Katz_core = Katz_centrality(adj_list, katz_alpha=0.05).reshape(-1, 1)
    # scaler = MinMaxScaler(feature_range=(0, 1))  # 平滑处理
    # Katz_core = scaler.fit_transform(Katz_core).flatten()
    return Data(adj=adj_list, features=features, labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test, Katz_core=Katz_core)


def load_data(path="./cora/cora_proceed/"):  # 装装载数据
    """

    :rtype: object
    """
    save_file = osp.join(path, "Data_cora_for_GraphSage.pkl")
    if osp.exists(save_file):
        # print("Using Cached file: {}".format(save_file))
        Data = pkl.load(open(save_file, "rb"))  # 加载文件
    else:
        Data = process_data()
        with open(save_file, "wb") as f:
            pkl.dump(Data, f)
        print("Cached file: {}".format(save_file))

    return Data.adj, Data.features, Data.labels, Data.idx_train, Data.idx_val, Data.idx_test, Data.Katz_core


if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test, katz_core = load_data()
    print("Adjacency's len: ", type(adj))
    print("Node's feature shape: ", type(features))
    print("Node's label shape: ", type(labels))
    print("Number of training nodes: ", len(idx_train))
    print("Number of validation nodes: ", len(idx_val))
    print("Number of test nodes: ", len(idx_test))
    print(len(katz_core))
    print()
