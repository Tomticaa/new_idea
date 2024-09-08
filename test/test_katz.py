# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：test.py
Date    ：2024/9/4 下午7:39 
Project ：new_idea 
Project Description：
    
"""
import numpy as np
from matplotlib import pyplot as plt


def make_graph(nums_node, with_katz, without_katz):  # 创建折线图
    plt.plot(nums_node, with_katz, label='with_katz Accuracy')
    plt.plot(nums_node, without_katz, label='without_katz Accuracy')
    plt.legend()
    plt.title('with_katz and without_katz Accuracy Over different_nums_node')
    plt.xlabel('nums_node')
    plt.ylabel('Accuracy')
    # plt.yticks(np.arange(0, 1.1, 0.05))
    plt.xticks(np.arange(1, len(nums_node)+1, 1))
    plt.grid(True)
    plt.savefig(r'katz/with_katz.png')
    plt.show()


a = [0.6724846363067627, 0.6827515363693237, 0.7043121457099915, 0.7166324853897095, 0.7402464151382446, 0.7371663451194763, 0.7156057953834534, 0.7238193154335022, 0.7330595850944519, 0.697125256061554]
b = [0.6817248463630676, 0.7073922157287598, 0.7197125554084778, 0.7320328950881958, 0.7351129651069641, 0.7238193154335022, 0.7289528250694275, 0.7258726954460144, 0.7320328950881958, 0.7145791053771973]
nums_node = np.arange(1, len(a) + 1)
make_graph(nums_node, b, a)
