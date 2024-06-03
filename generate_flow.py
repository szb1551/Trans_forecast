import random

import numpy as np
import math
import pickle
import networkx as nx
from data_utils import change_to_csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use('TkAgg')

# 使用例子
min_value = 1  # 假设的最小值
max_value = 10  # 假设的最大值

# 假设的邻接矩阵（需要您提供具体的矩阵）
# 这里以一个4x4矩阵为例，仅为了说明算法
adjacency_matrix = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
], dtype=float)


def generate_transition_probabilities(adjacency_matrix, output_nodes):
    # 指定入度为0的点和出度为0的点，对应的默认车流量（需要您提供这些数据）
    # output_nodes = [11, 13, 30, 41, 93, 149]  # 入度为0的节点列表

    # row_sums = np.sum(adjacency_matrix, axis=1)
    # output_nodes = np.where(row_sums == 0)[0]

    transition_probability_matrix = np.zeros_like(adjacency_matrix).astype(float)
    # np.sin((hours - 6) / 24 * 2 * np.pi) * (peak_traffic - base_traffic) / 2
    for i, row in enumerate(adjacency_matrix):
        # 找到当前路口可以转移到的路口
        transfer_indices = np.nonzero(row)[-1]
        # print(f"row {i}: {transfer_indices}")

        nonzerolen = len(transfer_indices)
        # if nonzerolen > 0:
        #     # 为这些路口随机分配概率
        #     rand_probs = np.random.rand(len(transfer_indices))
        #     # 归一化这些概率值，使它们的总和为1
        #     normalized_probs = rand_probs / rand_probs.sum()
        #     # 更新转移概率矩阵
        #     for j, prob in zip(transfer_indices, normalized_probs):
        #         if j in output_nodes:
        #             pass
        #         transition_probability_matrix[i,j] = prob
        temp = np.random.dirichlet(np.ones(nonzerolen), size=1)[0]
        transition_probability_matrix[i, transfer_indices] = temp
        # print(f"{transition_probability_matrix[i,:]}")

    return transition_probability_matrix


def map_to_probability_nonlinear(value, min_value, max_value):
    # 根据你的需求调整a的值，a > 1时函数增长较慢
    a = 2.5

    if value < min_value:
        return 0.2
    elif value > max_value:
        return 1
    else:
        # 将value映射到[0, 1]
        normalized_value = (value - min_value) / (max_value - min_value)
        # 使用指数函数进行非线性映射，并缩放到[0.3, 1]的范围内
        return 0.2 + (math.pow(a, normalized_value) - 1) / (a - 1) * 0.8

def generate_probability_leave():
    hours = np.arange(24)

    # 初始化概率数组，所有时段起始都有一个基础概率
    probabilities = np.ones(24) * 0.2

    # 定义早晚高峰的中心时间和高斯分布的标准差
    morning_peak_hour = 12
    evening_peak_hour = 21
    std_dev = 1.0

    # 使用高斯分布增加早晚高峰时段的概率
    morning_peak = np.exp(-0.5 * ((hours - morning_peak_hour) / std_dev) ** 2) * 0.5
    evening_peak = np.exp(-0.5 * ((hours - evening_peak_hour) / std_dev) ** 2) * 0.5

    # 将高峰概率叠加到基础概率上
    probabilities += morning_peak
    probabilities += evening_peak

    return probabilities

def generate_probability_high():
    hours = np.arange(24)

    # 初始化概率数组，所有时段起始都有一个基础概率
    probabilities = np.ones(24) * 0.2

    # 定义早晚高峰的中心时间和高斯分布的标准差
    morning_peak_hour = 9
    evening_peak_hour = 19
    std_dev = 1.0

    # 使用高斯分布增加早晚高峰时段的概率
    morning_peak = np.exp(-0.5 * ((hours - morning_peak_hour) / std_dev) ** 2) * 0.5
    evening_peak = np.exp(-0.5 * ((hours - evening_peak_hour) / std_dev) ** 2) * 0.5

    # 将高峰概率叠加到基础概率上
    probabilities += morning_peak
    probabilities += evening_peak

    return probabilities

def init_tarffic():
    # with open('./GCN_Dalian_Graph.pkl', 'rb') as f:
    # G = pickle.load(f)
    # adj = nx.adjacency_matrix(G).todense()  # 返回图的邻接矩阵
    # 分配默认车流量，例如这里我们假设默认流量为10

    data = pd.read_excel('data/数据源/zuobiaodaoluxinxi.xlsx', sheet_name=0, usecols='B:EU', nrows=150)
    adj = data.to_numpy()

    default_flow = 200
    num_intersections = adj.shape[0]
    initial_traffic_flow = np.zeros(num_intersections)

    # 只有对于有入度或出度的路口，我们才分配默认流量
    for i in range(num_intersections):
        random_integer = random.randint(-100, 100)
        initial_traffic_flow[i] = default_flow + random_integer
    initial_traffic_flow[11] = 0
    return adj, initial_traffic_flow


# 路口车流量的迭代转移
def transfer_traffic(adjacency_matrix, current_traffic_flow, node_prob_matrix, output_nodes, input_nodes, min_vehicle=50, max_vehicle=80):
    # 计算每个路口的出流量
    transition_prob_matrix = generate_transition_probabilities(adjacency_matrix, output_nodes)

    # input_nodes = [0, 9, 11, 12, 118]  # 入度为0的节点列表
    # output_nodes = [13, 30, 41, 93, 149]  # 出度为0的节点列表
    outgoing_traffic = np.floor(current_traffic_flow * node_prob_matrix)
    # 计算每个路口的入流量
    all_flow = np.sum(outgoing_traffic)
    incoming_traffic = np.array(np.dot(outgoing_traffic, transition_prob_matrix)).squeeze()
    incoming_traffic_int = np.floor(incoming_traffic).astype(int)
    # 计算损失的车辆数（因取整）
    # lost_traffic = np.sum(outgoing_traffic) - np.sum(incoming_traffic_int) - np.sum(outgoing_traffic[input_nodes])
    lost_traffic = np.sum(outgoing_traffic) - np.sum(incoming_traffic_int) - np.sum(outgoing_traffic[output_nodes])
    # lost_traffic = np.sum(incoming_traffic) - np.sum(incoming_traffic_int)
    # 新交通流量为输入流量 - 输出流量 + 本路口原流量
    # 分配损失的车辆数以保持总车辆数量不变
    float_remainder = incoming_traffic - incoming_traffic_int
    while lost_traffic > 0:
        # 找到最大小数部分的索引
        index = np.argmax(float_remainder)
        # 将一个损失的车辆分配给这个索引对应的路口
        incoming_traffic_int[index] += 1
        float_remainder[index] -= 1
        lost_traffic -= 1
    # values_to_add = [random.randint(min_vehicle, max_vehicle) for node in input_nodes]
    values_to_add = [random.randint(min_vehicle, max_vehicle) if node != 11 else 0 for node in input_nodes]
    incoming_traffic_int[input_nodes] += values_to_add
    new_traffic_flow = current_traffic_flow + incoming_traffic_int - outgoing_traffic
    # new_traffic_flow = incoming_traffic_int
    # 确保没有负的交通流量
    new_traffic_flow[new_traffic_flow < 0] = 0
    return new_traffic_flow


def plot_nodes(time_slots, traffic_flows, traffic_node):
    plt.figure(figsize=(12, 10))
    num = 0
    color_all = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    for node in traffic_node:
        flows1 = traffic_flows[:, node].reshape(-1, 4).sum(axis=-1)
        plt.subplot(len(traffic_node), 1, num + 1)
        plt.plot(time_slots, flows1, marker='o', linestyle='-', color=color_all[num % len(color_all)])
        plt.title('Flow Against Time (Node {})'.format(node))
        plt.xlabel('Time Slot')
        plt.ylabel('Flow')
        plt.grid(True)
        num += 1
    plt.tight_layout()
    plt.show()


def run():
    adjacency_matrix, initial_traffic_flow = init_tarffic()
    num_iterations = 96  # 迭代次数
    # 初始化车流量，给所有有入度或出度的路口分配默认车流量，假设默认流量为10
    # 最大，最小生成车辆
    min_vehicle = 50
    max_vehicle = 100
    num_intersections = adjacency_matrix.shape[0]  # 路口的数量
    # 初始化车流量矩阵
    row_sums = np.sum(adjacency_matrix, axis=1)
    output_nodes = np.where(row_sums == 0)[0]
    col_sums = np.sum(adjacency_matrix, axis=0)
    input_nodes = np.where(col_sums == 0)[0]

    traffic_flows = np.zeros((num_iterations, num_intersections)).astype(float)

    degree_over_one = np.where(row_sums > 1)[0]
    print(f"only one child node, {degree_over_one}")

    # 第0次迭代，只有初始车流量
    traffic_flows[0] = initial_traffic_flow
    transition_prob_matrix = generate_transition_probabilities(adjacency_matrix, output_nodes)

    print(transition_prob_matrix)

    # 生成24小时的离开车流量
    # node_pros = np.zeros(24, num_intersections)
    day_probabilities = generate_probability_high()
    day_leave = generate_probability_leave()
    node_pros = np.array([day_leave for _ in range(num_intersections)]).T
    vehicle_hours_min = np.round(min_vehicle + (min_vehicle*2-min_vehicle)*day_probabilities)
    vehicle_hours_max = np.round(max_vehicle + (max_vehicle*2-max_vehicle)*day_probabilities)
    # for i in range(24):
    #     node_temp = np.zeros(num_intersections)
    #     for j in range(len(node_pros)):
    #         value = random.uniform(min_value, max_value)
    #         probability = map_to_probability_nonlinear(value, min_value, max_value)
    #         node_temp[j] = probability
    #     node_pros[i] = node_temp

    # 开始迭代过程，从第1次迭代开始直到第96次
    for iteration in tqdm(range(1, num_iterations)):
        index = math.floor(iteration/4%24)
        traffic_flows[iteration] = transfer_traffic(adjacency_matrix, traffic_flows[iteration - 1], node_pros[index],
                                                    output_nodes, input_nodes, vehicle_hours_min[index], vehicle_hours_max[index])

    # 输出所有迭代后的每个路口的车流量
    print("所有迭代后的每个路口车流量:")
    print(traffic_flows)
    print("初始车辆{}，迭代后的车辆{}".format(initial_traffic_flow.sum(), traffic_flows[-1].sum()))
    # change_to_csv(traffic_flows)

    time_slots_hour = np.arange(0, int(num_iterations / 4))  # 按小时的采样
    plot_nodes(time_slots_hour, traffic_flows, [118, 93])
    plot_nodes(time_slots_hour, traffic_flows, [12, 149])
    plot_nodes(time_slots_hour, traffic_flows, [0, 13])
    plot_nodes(time_slots_hour, traffic_flows, [9, 30])
    plot_nodes(time_slots_hour, traffic_flows, [9, 41])


if __name__ == '__main__':
    run()
