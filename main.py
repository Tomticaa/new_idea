# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：main.py
Date    ：2024/9/2 下午5:12 
Project ：new_idea 
Project Description：
    
"""
# -*- coSage_envding: UTF-8 -*-
import time
import argparse
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from dataset import Data
from env import Sage_env
from agent import QAgent

parser = argparse.ArgumentParser(description='AutoSage')
# env definition=-
parser.add_argument('--env_hid_dim', type=int, default=128)
parser.add_argument('--env_out_dim', type=int, default=7)
parser.add_argument('--Sage_num_layers', type=int, default=2)  # 卷积层叠加层数(也为邻居采样的最大层数)
parser.add_argument('--Sage_batch_size', type=int, default=135)  # 每批次训练节点数以及经验池采样数量
parser.add_argument('--NUM_BATCH_PER_EPOCH', type=int, default=20)  # 每轮训练执行多少个batch
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
# agent definition
parser.add_argument('--replay_memory_size', type=int, default=5000)  # 经验回放内存的总大小
parser.add_argument('--update_target_estimator_every', type=int, default=10)  # 定义了目标网络的更新频率，默认每一步更新一次。
parser.add_argument('--discount_factor', type=float, default=0.9)  # 折扣因子，用于计算未来奖励的现值。
parser.add_argument('--max_sample_num', type=int, default=10)  # 最多选取十个数量的邻居选取动作
parser.add_argument('--mlp_layers', type=list, default=[256, 128, 64])  # 定义qnet中MLP的每层神经元数量
parser.add_argument('--max_episodes', type=int, default=1000)  # 总周期数
parser.add_argument('--max_timesteps', type=int, default=20)  # 每个周期填充 30 批次经验(30*135)

parser.add_argument('--epochs', type=int, default=100)  # GNN训练轮次
args = parser.parse_args()


def make_graph(epochs, train_accs, test_accs):  # 创建折线图
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, test_accs, label='Testing Accuracy')
    plt.legend()
    plt.title('Training and Testing Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.grid(True)
    plt.savefig(r'assets/performance.png')
    plt.show()


def main(K=0):  # 这里的 K 应该传入数据集处理函数实现 K 折交叉验证
    start = time.time()
    env = Sage_env(hid_dim=args.env_hid_dim,
                   output_dim=args.env_out_dim,
                   num_layers=args.Sage_num_layers,
                   max_sample_num=args.max_sample_num,
                   NUM_BATCH_PER_EPOCH=args.NUM_BATCH_PER_EPOCH,
                   lr=args.lr,
                   weight_decay=args.weight_decay,
                   policy="")  # 环境初始化
    agent = QAgent(replay_memory_size=args.replay_memory_size,
                   update_target_estimator_every=args.update_target_estimator_every,
                   discount_factor=args.discount_factor,
                   epsilon_decay_steps=int((args.max_episodes * args.max_timesteps)*0.3),  # 衰减步长应该为整体步数的0.3
                   lr=args.lr,
                   batch_size=env.batch_size,  # 140 / 135
                   Sage_num_layers=args.Sage_num_layers,
                   max_sample_num=args.max_sample_num,
                   mlp_layers=args.mlp_layers,
                   state_dim=env.features.shape[1])
    env.policy = agent  # 初始化策略
    end = time.time()
    print(f"init time: {end - start}")
    print("Training RL agent......")
    start = time.time()
    tag = 0
    last_val = 0.0
    return_list = []
    print("训练 {} 次 agent，每次装填 {} * {} 条经验；".format(args.max_episodes, args.max_timesteps, args.Sage_batch_size))
    for episode in range(args.max_episodes):  # 训练智能体的轮次
        loss, _, (val_acc, Cumulative_rewards) = agent.learn(env, args.max_timesteps)
        if val_acc > last_val:
            best_policy = deepcopy(agent)
            last_val = val_acc
            tag = episode
        print('Episode:', episode, "Val_Acc:", val_acc, "rewards:", Cumulative_rewards, 'DQN_Loss:', loss)  # 奖励，损失还是设计的不合适
    end = time.time()
    print(f"agent training time: {end - start}")
    print("Training GNNs with learned RL agent")

    new_env = Sage_env(hid_dim=args.env_hid_dim,
                       output_dim=args.env_out_dim,
                       num_layers=args.Sage_num_layers,
                       max_sample_num=args.max_sample_num,
                       NUM_BATCH_PER_EPOCH=args.NUM_BATCH_PER_EPOCH,
                       lr=args.lr,
                       weight_decay=args.weight_decay,
                       policy="")  # 环境初始化
    new_env.policy = best_policy
    start = time.time()
    index, states = new_env.reset()  # 重置环境状态
    train_accs = []
    test_accs = []
    epochs = np.arange(args.epochs)
    print("The episode: {} strategy guides GNN training".format(tag))
    for i_episode in range(args.epochs):  # 使用训练好的最佳策略指导GNN计算
        actions = new_env.policy.predict_action_sequences_new(index, states, new_env)  # TODO: 应该在训练阶段消耗完所有随机步长，在指导GNN计算时仅能使用DQN网络选取
        t = time.time()
        loss, train_accuracy = new_env.step(actions, index, dqn_train_tag=False)  # 仅仅执行一次训练，不计算其他参数
        # loss, train_accuracy = new_env.train(actions, index)  # 仅仅执行一次训练，不计算其他参数
        _, test_acc = new_env.test()
        train_accs.append(train_accuracy)
        test_accs.append(test_acc)
        print(" The {}th time: {:03f} train_loss:{}  train_acc: {:03f} test_acc：{:03f}".format(i_episode, time.time() - t, loss,  train_accuracy, test_acc))

    end = time.time()
    print(f"GNN training time: {end - start}")
    return max(test_accs), epochs, train_accs, test_accs


if __name__ == '__main__':
    # K = 10  # 期待实现 K 折交叉验证
    # for item in range(K):
    #     print(f"Start{item} fold")
    #     test_acc = main(K=item)
    #     print("Test Accuracy:", test_acc)
    max_test, epoch, train, test = main()
    make_graph(epoch, train, test)
    print("Test best Accuracy:", max_test)
