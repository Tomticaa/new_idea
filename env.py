# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：Sage_env.py
Date    ：2024/8/14 下午2:28
Project ：new_model
Project Description：
    使用 GraphSage 主网络设计强化学习环境，实现聚合邻居数量的动态选取。
    TODO： 环境：线性层中加入 Batch Normalization 层,以及加入batch—size进行加速训练
    解决办法 1，直接减少线性层为 1433 -> 7 并进行参数初始化
    解决办法 2，层数不变，使用正则化方法，加入BatchNormal方法。

    TODO: 每次训练之后，初始化一次环境参数是否有必要？？？为了训练更合适的智能体决策，我觉得有必要
    # TODO: 奖励应该能够区分不同动作的优劣；同时考虑短期和长期的回报
    # TODO: 合理设定终止条件，使得智能体在探索有效状态空间时可以得到足够的经验，同时防止过早或过晚的终止。
    # TODO: 检查 q 网络是否编写失误
    # TODO: 重大发现：原始状态节点原始特征过稀疏 ，状态转移之后都为连续向量，导致动作选择单一，数据预处理或者状态转移归一化；

    # TODO: 设定的奖励目标为： 使用更少的 采样数量达到更好的准确率效果

    # TODO： 更改状态转移测： 将经过聚合后的状态与训练集中其他节点状态进行特征比较，选择差异最小节点进行状态转移
"""
from collections import namedtuple

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

from dataset import load_data, Data

random.seed(64)
np.random.seed(64)
torch.manual_seed(64)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Trans_feature = namedtuple('Trans_feature', ['src_index', 'next_state'])  # 定义转移状态


class SageGCN(nn.Module):  # 定义图卷积层
    def __init__(self, activation=torch.relu):  # TODO: 可尝试使用多种激活函数
        super(SageGCN, self).__init__()
        self.activation = activation  # 中间层添加激活函数

    @staticmethod
    def NeighborAggregator(action, neighbor_feature):
        # TODO: 仅返回邻居特征的均值，也可以加入注意力机制为不同邻居增加权重系数
        t = 0
        aggr_neighbor = torch.zeros(len(action), neighbor_feature.size(1)).to(DEVICE)  # 初始化新张量，用于存储聚合后的结果
        for i, num in enumerate(action):  # 枚举每个采样数
            sample_num = num + 1  # 将动作 0~9 -> 1~10
            selected_tensors = neighbor_feature[t:t + sample_num]
            aggr_neighbor[i] = selected_tensors.mean(dim=0)  # 可以尝试多种聚合方法
            t += sample_num
        return aggr_neighbor

    def forward(self, action, src_node_features, neighbor_node_features):  # action = actions[hop] 为 0~9
        neighbor_hidden = self.NeighborAggregator(action, neighbor_node_features)  # 聚合邻居特征
        hidden = (src_node_features + neighbor_hidden) / 2  # TODO: 是否应该取平均
        if self.activation:  # 在中间层表示允许激活
            return self.activation(hidden)
        else:
            return hidden


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):  # hidden_dim = 128
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # 添加几层图卷积
        self.adj, self.features, _, _, _, _, self.core = load_data()
        self.gcn = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn.append(SageGCN())  # TODO： 舍弃 sigmoid 接入 relu
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, output_dim))  # 叠加多层线性层并使用relu函数进行连接
        self.initialize_weights()  # 参数初始化

    def initialize_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, actions, src_index, trans_features):  # 0~9  TODO：模型计算太慢
        self.features = trans_features  # 将转移后的特征进行赋值
        hidden = self.multi_hop_sampling(actions, src_index)  # 直接返回的tensor(已放在cuda上)
        for l in range(self.num_layers):  # 循环遍历每层，使用对象层对应的的sage卷积处理节点特征
            next_hidden = []  # 列表，用于收集当前层处理后的节点特征。
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                h = gcn(actions[hop], src_node_features, hidden[hop + 1])
                next_hidden.append(h)  # 在列表末尾添加
            hidden = next_hidden  # 更新列表，准备进行下一图层处理
        output = self.fc(hidden[0])
        return output, hidden[0]

    def compute_probabilities(self, sid):  # 根据中心性计算采样概率
        probabilities = []
        for neighbor_node in self.adj[sid]:  # 节点的每个邻居的Katz分数
            probabilities.append(self.core[neighbor_node])
        total = sum(probabilities)
        normlized_probabilities = [prob / total for prob in probabilities]  # 归一化
        return normlized_probabilities

    def sampling(self, src_nodes, action):  # 这里 sample_num 内部元素值为0~9，对应采样动作 1~10
        result = []
        for i, sid in enumerate(src_nodes):
            sample_num = action[i] + 1
            neighbor_nodes = self.adj.get(sid, [])
            if len(neighbor_nodes) > 0:
                probability = self.compute_probabilities(sid)
                # res = np.random.choice(self.adj[sid], size=(sample_num,), replace=True, p=probability)  # size：需要采样的数量
                res = np.random.choice(self.adj[sid], size=sample_num)  # TODO：使用随机采样进行调试
            else:
                res = []
            result.extend(res)
        return np.array(result)  # 返回采样结果的数组

    def multi_hop_sampling(self, actions, src_nodes):  # 集成多重采样，采样结果特征提取，actions则为一个多重列表
        sampling_result = [src_nodes]
        for k, action in enumerate(actions):
            hop_k_result = self.sampling(sampling_result[k], action)  # 返回对应层的采样结果数组
            sampling_result.append(hop_k_result)
        sampling_result_x = [torch.from_numpy(self.features[idx]).float().to(DEVICE) for idx in sampling_result]
        return sampling_result_x  # 提取采样结果的对应特征并返回特征列表


class Sage_env:
    def __init__(self, hid_dim, output_dim, num_layers, max_sample_num, lr, weight_decay, NUM_BATCH_PER_EPOCH,
                 policy=""):
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.max_sample_num = max_sample_num
        self.lr = lr
        self.weight_decay = weight_decay
        self.policy = policy
        self.processing_data()  # 进行数据处理
        self.trans_features = self.features  # 转移特征初始化为原始特征
        self.batch_size = len(self.idx_train)  # 为训练节点数量作为批次训练数量，同时也为 dqn 每次采样经验的数量
        self.past_performance = [0.0]  # 该两个变量用于计算前面的平均准确率，用于定义奖励
        self.baseline_experience = 5  # 过去五个批次
        self.last_sample = [0] * len(self.idx_train)  # TODO:储存过去一个时间步该批次节点对应采样数量的均值；初始化为采样的最少节点数量
        # --------------------------------------------------------------------------------------
        self.NUM_BATCH_PER_EPOCH = NUM_BATCH_PER_EPOCH  # 每轮训练执行多少个批次  TODO：没有用到
        self.buffers = [[] * len(self.idx_train)]  # 用于存储每一批次所有时间步对应节点采样数量的均值

        # 定义模型
        self.model = GraphSage(self.features.shape[1], hid_dim, output_dim, num_layers).to(
            DEVICE)  # 拟设定num_layers=2为图卷积层数
        self.optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def processing_data(self):
        print("processing dataset............")
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, _ = load_data()
        self.init_states = self.features
        print("The data has been loaded! ")

    def reset(self):  # TODO：是否加入随机性打乱初始状态顺序还是对节点特征随机加入扰动？
        self.reset_parameters()  # TODO: 是否应该重置模型参数，不重置环境验证集准确率更高，但是会为dqn学习到更好的拟合效果么
        # scr_index = np.random.choice(self.idx_train, int(len(self.idx_train)*0.1), replace=False)
        scr_index = self.idx_train
        states = self.init_states[scr_index]  # TODO：考虑在状态初始化过程中随机加入高斯扰动用于模拟状态初始化的轻微波动
        self.trans_features[scr_index] = self.init_states[scr_index]  # 将转移特征初始化
        self.optimizer.zero_grad()
        return scr_index, states

    def reset_parameters(self):  # 模型参数的重置
        self.model.initialize_weights()  # 仅仅清除线性层参数

    def step(self, actions, scr_index):  # action：0~9 -> 采样数量 1~10
        done = False
        print('----------------------------------------------------------------')
        self.reset_parameters()
        for _ in range(20):
            loss, accuracy, pred_rights = self.train(actions, scr_index)
            print("loss: ", loss, 'accuracy: ', accuracy.item())
        # 在训练集上的分类损失达到 0.01 一下则设置为终止状态
        # if loss < 0.01:  # plan3
        #     done = True
        # dones = [done] * len(actions[0])
        sample_num = actions[0]  # 获取原始节点采样数量 # TODO:设计为与上一时间步的对应节点采样数进行对比
        dones = [True if i == 1 else False for i in pred_rights]  # TODO: 终止标志也应该合理计入
        next_states, trans_index = self.state_transfer(actions, scr_index)  # 使用执行动作更新参数得到的网络得到下一阶段状态
        val_acc = self.eval()  # 仅计算验证集准确率，用于选择合适的agent，所以验证集合内结点的选择应该具有代表测试集节点的能力
        baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
        self.past_performance.append(val_acc)
        train_acc_performance = [0.6 if i == 1 else 0 for i in pred_rights]

        # 基础采样准确率为 一次agent训练的所有时间步长内节点多次采样的均值，仅在一此训练之后进行更新
        sample_num_performance = [0.06 * (x - y - 1) for x, y in zip(self.last_sample, sample_num)]
        val_acc_performance = val_acc - baseline
        # 奖励设置优先以 训练集节点节点是否预测成功为首，若预测成功，则计算应逐步减少节点聚合数量？？
        # TODO：设置多尺度奖励： r =  0.6 * (测试集分类准确率) + 0.2 * (当前批次节点聚合邻居数目相较于上一批次节点数目增量的负数) + 0.2 * (验证集相较于过去十个批次平均提升准确率是否上升)： 结合节点计算效率与准确性
        rewards = [x + y + val_acc_performance for x, y in zip(train_acc_performance, sample_num_performance)]
        self.last_sample = sample_num  # 更新当前采样数量，应该一次训练一更新
        r = np.sum(np.array(rewards))  # 计算该轮次总计奖励
        return (next_states, trans_index), rewards, dones, (val_acc, r)  # 返回平均准确率与平均奖励

    def state_transfer(self, actions, src_index):
        _, next_states = self.model(actions, src_index, self.trans_features)
        next_states = self.Feature_preprocessing(next_states)  # 特征规范化处理模块
        self.trans_features[src_index] = next_states
        return next_states, src_index

    def GNN_train(self, actions, scr_index):  # 在训练过程中使用随机采样，将过程的转移状态扩充到整个训练集和内
        loss, accuracy, _ = self.train(actions, scr_index)
        return loss, accuracy

    @staticmethod
    def Feature_preprocessing(node_feature):  # 对转移特征进行规范化处理：输入为 tensor 输出为 numpy
        # TODO： 是否计入节点特征的随机扰动，是否添加归一化函数
        node_feature = node_feature.detach().cpu().numpy()
        scaler = MinMaxScaler(feature_range=(0, 1))  # 节点值缩放  T
        node_feature = scaler.fit_transform(node_feature)
        return node_feature

    def train(self, actions, scr_index):  # 仅执行一次训练；
        self.model.train()
        logits, _ = self.model(actions, scr_index, self.trans_features)
        labels = torch.LongTensor(self.labels[scr_index]).to(DEVICE)
        loss = self.criterion(logits, labels)
        self.optimizer.zero_grad()
        preds = logits.max(1)[1]
        pred_rights = (preds == labels).int()
        accuracy = torch.eq(preds, labels).float().mean()
        loss.backward()  # 反向传播计算较为耗时
        self.optimizer.step()
        return loss.item(), accuracy, pred_rights

    def eval(self):  # 在验证集上对模型执行评估
        self.model.eval()
        with torch.no_grad():
            val_index = self.idx_val
            val_states = self.init_states[val_index]
            val_actions = self.policy.predict_action_sequences(val_index, val_states, self)
            logits, _ = self.model(val_actions, val_index, self.trans_features)
            preds = logits.argmax(dim=1)
            labels = torch.LongTensor(self.labels[val_index]).to(DEVICE)
            accuracy = torch.eq(preds, labels).float().mean()
        return accuracy.cpu().numpy()

    def test(self):
        self.model.eval()
        with torch.no_grad():  # 完全禁用梯度计算，提高模型推理速度
            test_index = self.idx_test
            test_states = self.init_states[test_index]
            test_actions = self.policy.predict_action_sequences(test_index, test_states, self)  # 将自己作为环境输入进函数
            logits, _ = self.model(test_actions, test_index, self.trans_features)
            preds = logits.argmax(dim=1)
            labels = torch.LongTensor(self.labels[test_index]).to(DEVICE)
            reward = (preds == labels).int()
            accuarcy = torch.eq(preds, labels).float().mean().item()
        return reward.detach().cpu().numpy(), accuarcy
