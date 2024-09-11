# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：agent.py
Date    ：2024/9/2 下午5:12 
Project ：new_idea 
Project Description：
        定义 DQN 智能体：定义动作值函数，作用：将节点特征作为节点状态，输出动作为int；
        TODO: 1，平衡原始特征与聚合之后特征的差异，抚平在 q 网络进行预测的过程中输入原始特征与聚合后的节点的特征后得到 q 值分布的差异；
        TODO：2， 设置多尺度奖励，避免单一尺度奖励使简单达成目标使 q 网络不能学习到更具体的策略，纠正仅输出单一动作的缺点；
"""
# -*- coding: UTF-8 -*-

import torch
import random
import numpy as np
import torch.nn as nn
from copy import deepcopy
from collections import namedtuple

random.seed(64)
np.random.seed(64)
torch.manual_seed(64)
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])  # 定义经验
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Memory_RelayBuffer:  # 定义经验回放池
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):  # 添加经验
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)  # 弹出队头经验维持缓冲池大小
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, batch_size):  # 采样
        if len(self.memory) < batch_size:
            samples = random.sample(self.memory, len(self.memory))

        else:
            samples = random.sample(self.memory, batch_size)
        return map(np.array, zip(*samples))  # 返回numpy形式的批量经验

    def size(self):  # 返回记忆缓存长度
        return len(self.memory)


class Qnet(nn.Module):  # 定义 Q 网络
    def __init__(self, input_dim, output_dim, mlp_layers):
        super(Qnet, self).__init__()
        self.state_dim = input_dim
        self.action_dim = output_dim
        layer_dims = [self.state_dim] + mlp_layers

        fc = [nn.Flatten()]
        for i in range(len(layer_dims) - 1):  # 重复叠加多层感知机:[1433 + 256, 128, 64]
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            fc.append(nn.LeakyReLU(negative_slope=0.01))  # 叠加过程中插入激活函数
        fc.append(nn.Linear(layer_dims[-1], self.action_dim, bias=True))  # 输出为 10 维度
        self.fc_layers = nn.Sequential(*fc)  # 序列打包
        self._init_weights()  # 参数初始化

    def _init_weights(self):
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states):  # 前向传播连接网络
        q_value = self.fc_layers(states)
        return q_value


class DQN:
    def __init__(self, state_dim, max_sample_num, mlp_layers, lr, device):
        self.device = device
        self.qnet = Qnet(input_dim=state_dim, output_dim=max_sample_num, mlp_layers=mlp_layers).to(self.device)
        self.qnet.eval()  # TODO：是否存在Dropout 或 Batch Normalization,应该加入
        self.alpha = 0.5  # TODO：仅用在最近邻机制
        for p in self.qnet.parameters():  # 使用于 Sigmoid 或 Tanh 激活函数的参数初始化
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self.mse_loss = nn.MSELoss(reduction='mean')  # 定义均方误差损失函数
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)

    def predict_nograd(self, states):  # 输入批量状态，预测对应 q 值
        with torch.no_grad():
            states = torch.from_numpy(states).float().to(self.device)
            q_values = self.qnet(states).cpu().numpy()
        return q_values  # 将 tensor 转化为数组返回

    def update(self, states, actions, target_q_values, episode):
        self.optimizer.zero_grad()
        self.qnet.train()
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(self.device)  # 转化为列向量
        # TD误差: target_q_values = r + (1 - done) * alpha * max_q(s' , a')
        # TODO：如果 done 总为 true 则 target_q_values = rewards，则要求拟合：q_values = rewards
        target_q_values = torch.tensor(target_q_values, dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.qnet(states).gather(1, actions)  # 得到了每个状态批量中选定相应动作后对应的Q值
        # target_q_values = (self.alpha ** episode) * q_values + (1 - self.alpha ** episode) * target_q_values
        Q_loss = self.mse_loss(q_values, target_q_values)
        Q_loss.backward()
        self.optimizer.step()
        Q_loss = Q_loss.item()  # 将 tensor 转化成标量
        self.qnet.eval()  # TODO：是否存在Dropout 或 Batch Normalization？
        return Q_loss


class QAgent:
    def __init__(self, replay_memory_size,
                 update_target_estimator_every,
                 discount_factor,
                 epsilon_decay_steps,
                 lr,
                 batch_size,
                 Sage_num_layers,  # 卷积层叠加层数(也为邻居采样的最大层数)
                 max_sample_num,
                 mlp_layers,
                 state_dim):
        self.replay_memory_size = replay_memory_size
        self.update_target_estimator_every = update_target_estimator_every  # 多久更新一次目标网络
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(1., 0., epsilon_decay_steps)  # 贪婪策略
        # 衰减步数应该为训练步骤的50%~80%，例如执行200个epoch每次执行20step_time，总计步骤200*20=4000步，衰减步长应该为 2000~3200左右
        self.state_dim = state_dim  # 1433
        self.batch_size = batch_size  # 每次在缓冲池中采样经验的数量  140 为训练节点数量
        self.max_sample_num = max_sample_num  # int: 10 :代表每个节点的邻居采样的范围为 0~9
        self.Sage_num_layers = Sage_num_layers  # int: 3 :卷积层叠加层数(也为邻居采样的最大层数),用于创建多重采样列表
        self.train_t = 0  # 训练次数计数器，用于控制更新目标网络的频率
        self.episode = 0  # 作用：用于计算当前 q 值与目标q值的加权和  TODO：与 alpha一样没用到
        self.total_t = 0  # 在 save 经验时自增，计算随时间递减的 ε 值，实现了动作随机探索
        self.device = DEVICE

        self.sample_q_net = DQN(state_dim, max_sample_num, mlp_layers, lr, self.device)  # 定义q网络，目标网络以及经验回放池
        self.sample_target_net = DQN(state_dim, max_sample_num, mlp_layers, lr, self.device)
        self.memory = Memory_RelayBuffer(memory_size=replay_memory_size)

    def learn(self, env, total_time_steps):  # 输入环境，在时间 t 内执行一次完整的强化学习过程，其中关于环境的各种定义在 Sage_env中获取
        src_index, states = env.reset()  # TODO： 初始化状态是否应该重置环境卷积模型参数
        # 执行一次智能体训练
        Cumulative_rewards = 0
        for _ in range(total_time_steps):  # 训练时间步  TODO:添加进度条并显示奖励
            actions = self.predict_action_sequences(src_index, states, env)  # 根据当前状态在q网络中形成最佳动作序列(已添加随机性)
            next_states, rewards, dones, (val_acc, r) = env.step(actions, src_index)  # 执行动作为多重列表
            transition = zip(states, actions[0], rewards, next_states, dones)  # 仅仅将第一层针对源节点采取的行动加入经验进行训练
            for ts in transition:
                self.feed(ts)
            states = next_states  # 进行时间步 次的状态转移
            # print("reward :{}".format(r))
            Cumulative_rewards += r
        loss = self.train()
        return loss, rewards, (val_acc, Cumulative_rewards)  # Cumulative_rewards:执行一个批次动作得到的全部奖励

    def feed(self, ts):  # 将经验分别添加到缓冲池当中 TODO；这里有优化空间
        (state, action, reward, next_state, done) = tuple(ts)  # 解包
        self.memory.save(state, action, reward, next_state, done)

    def eval_step(self, states):  # 批量动作作为输入，计算 q 值，直接返回该层次预测动作
        q_values = self.sample_q_net.predict_nograd(states)  # 返回的 q 值为数组形式
        best_actions = np.argmax(q_values, axis=1)
        return best_actions  # 返回值为 np 数组 ，代表该批次中该层节点采样的邻居个数

    def predict_action_sequences(self, index, states, env):
        actions_list = [[] for _ in range(self.Sage_num_layers)]
        new_states = states
        # epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]  # 设置随机探索率衰减到0
        epsilon = 0  # TODO：用于排查问题
        self.total_t += 1  # 衰减
        for actions_sub_list in actions_list:
            actions = self.eval_step(new_states)  # 返回一个批次的预测动作
            # best_actions = [act if np.random.rand() > epsilon else np.random.randint(0, 10) for act in actions]
            # TODO: 方便调试
            best_actions = [5 if np.random.rand() > epsilon else np.random.randint(0, 10) for act in actions]
            actions_sub_list.extend(best_actions)
            sample_result = env.model.sampling(index, best_actions)  # 针对索引列表内节点使用预测出来的最佳采样数量策略进行采样
            new_states = env.init_states[sample_result]  # TODO: 为什么每次将采样结果输入的新状态后得到的新动作都是一样的值？？因为新状态变为稀疏矩阵了！！！！！！！！！！
            index = sample_result  # 传递采样结果索引到下一层  TODO: 原始预测动作皆为相同值，因为原始特征相较于聚合后加入的参数的特征存在明显差异
        return actions_list  # 改造返回最佳动作的多重列表

    def train(self):  # 执行训练
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        best_actions_next = self.eval_step(next_states)  # 根据状态计算q值并批量预测动作
        q_values_next_target = self.sample_target_net.predict_nograd(next_states)  # 输出为没有梯度的数组，没法进行反传播
        # 根据公式计算目标q值
        max_q_values = q_values_next_target[np.arange(len(best_actions_next)), best_actions_next]  # 选择最好动作对应的q值
        #  T_q = r + (1 - done) * alpha * max_q
        target_q_values = rewards + np.invert(dones).astype(np.float64) * self.discount_factor * max_q_values  # TD目标  TODO：是否是done发生错误？？
        loss = self.sample_q_net.update(states, actions, target_q_values, self.episode)  # 训练q网络得到损失
        if self.train_t % self.update_target_estimator_every == 0:  # 训练10次一更新
            self.sample_target_net = deepcopy(self.sample_q_net)
        self.train_t += 1
        return loss
