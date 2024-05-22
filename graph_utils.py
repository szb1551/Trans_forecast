# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import scipy.sparse as sp
from datetime import timedelta
from math import sin, cos, sqrt, atan2, radians
from ProcessData import process_time
from data_handler import get_dataset
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
import re

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')
"""
将数据data处理为图结构
"""


def euclidian_dist(node_x, node_y):  # 边的权重
    lat1 = node_x['lat']
    long1 = node_x['long']
    lat2 = node_y['lat']
    long2 = node_y['long']
    return np.sqrt((lat1 - lat2) ** 2 + (long1 - long2) ** 2)


def distance_in_meters(node_x, node_y):  # 返回两点的半正弦距离
    R = 6373.0

    lat1 = radians(abs(node_x['lat']))
    lon1 = radians(abs(node_x['long']))
    lat2 = radians(abs(node_y['lat']))
    lon2 = radians(abs(node_y['long']))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def normalize_adj(adj, symmetric=True, add_self_loops=True):  # 初始化邻接矩阵吗，得到要处理的矩阵
    if add_self_loops:
        adj = adj + sp.diags(np.ones(adj.shape[0]) - adj.diagonal())  # adj + I

    if symmetric:  # 是否要对称
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()  # 返回压缩行格式
    else:
        d = sp.diags(np.float_power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()

    return a_norm


def get_graph(data):  # 得到图结构
    G = nx.Graph()

    for station in data['ID'].unique():
        G.add_node(station)
        G.nodes[station]['ID'] = data[data['ID'] == station]['ID'].iloc[0]
        G.nodes[station]['lat'] = data[data['ID'] == station]['Latitude'].iloc[0]
        G.nodes[station]['long'] = data[data['ID'] == station]['Longitude'].iloc[0]
        G.nodes[station]['pos'] = (G.nodes[station]['long'], G.nodes[station]['lat'])

    for node_x in G.nodes:
        for node_y in G.nodes:
            dist = distance_in_meters(G.nodes[node_x], G.nodes[node_y])
            if (dist > 2.5):
                continue
            G.add_edge(node_x, node_y)
            G[node_x][node_y]['weight'] = np.exp(-dist)

    adj = nx.adjacency_matrix(G)  # 返回图的邻接矩阵
    nx.write_gpickle(G, 'train_data/GCN_Graph.gpickle', protocol=3)
    return G, normalize_adj(adj).todense()  # todense稠密表示


def setup_GCN(data, forecast_horizon, num_lags, time=False):  # 得到图，邻接矩阵，训练测试集
    # G = get_graph(data)
    G = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    # 提取图的邻接矩阵
    adj = nx.adjacency_matrix(G)
    # 总天数的数量，这地方可以尝试修改特化一下
    number_of_hours = int((data.index.max() - data.index.min()).total_seconds() // (3600 * 24))
    timeseries_ = np.zeros([len(G.nodes()), number_of_hours + 1])  # 节点数与天数，记录当前节点当天的能量和
    start_time = data.index.min()
    time_list = []
    num = re.compile(r"\d+")
    for i in range(0, number_of_hours + 1):
        timewindow_start = start_time + timedelta(days=i)
        num_time = int(''.join(num.findall(str(timewindow_start))[:3]))
        time_list.append(num_time)
        current = data[(data.index == timewindow_start)]

        for k, node in enumerate(G.nodes()):
            tmp = current[G.nodes[node]['ID'] == current['ID']]
            timeseries_[k, i] = np.sum(tmp['Energy'])

    np.save('train_data/G_timeseries_map.npy', timeseries_)
    max_ = np.max(timeseries_, axis=1)[:, None]
    np.save('train_data/G_max_.npy', max_)
    normalized = timeseries_ / max_

    # timeseries_ones = np.ones(timeseries_.shape)
    # timeseries_ones = timeseries_+timeseries_ones
    # timeseries_logs = np.log(timeseries_ones)
    # np.save('train_data/G_timeseries_log_map.npy', timeseries_logs)
    # max_logs = np.max(timeseries_logs, axis=1)[:,None]
    # np.save('train_data/G_max_logs.npy', max_logs)

    # normalized = timeseries_ / np.max(timeseries_, axis=1)[:, None]

    NUM_LAGS = num_lags
    STEPS_AHEAD = forecast_horizon
    n_nodes = len(G.nodes())

    matrix_lags = np.zeros(
        (timeseries_.shape[-1] - (NUM_LAGS + STEPS_AHEAD), timeseries_.shape[0], NUM_LAGS + STEPS_AHEAD))
    i_train = matrix_lags.shape[0] - STEPS_AHEAD
    i_test = matrix_lags.shape[0]

    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = normalized[:, i:i + NUM_LAGS + STEPS_AHEAD]

    # ---------------- Train/test split
    train_dataset = matrix_lags[:i_train, :, :]  # [batch, 47, 37]
    test_dataset = matrix_lags[i_train:, :, :]  # [batch, 47, 37]

    # np.save('train_data/gcn_train_dataset.npy', train_dataset)
    # np.save('train_data/gcn_test_dataset.npy', test_dataset)
    print('GCN_train_dataset:', train_dataset.shape)
    print('GCN_test_dataset:', test_dataset.shape)
    if time:
        # timelist = np.load('train_data/time_list.npy')
        time_list = np.array(time_list)
        X_train_time, X_test_time = process_time(time_list, train_length=num_lags, forcast_window=forecast_horizon)
        return train_dataset, test_dataset, X_train_time, X_test_time
    return adj, train_dataset, test_dataset


# 得到Dalian数据结构的GCN图拓扑结构
def get_graph_Dalian(data, data_map, time=False, figure=True):
    G = nx.Graph()
    temp = 0
    for station in data['路口名称'].unique():
        G.add_node(station)
        G.nodes[station]['ID'] = temp
        G.nodes[station]['name'] = data[data['路口名称'] == station]['路口名称'].iloc[0]
        G.nodes[station]['lat'] = data[data['路口名称'] == station]['纬度'].iloc[0]
        G.nodes[station]['long'] = data[data['路口名称'] == station]['经度'].iloc[0]
        G.nodes[station]['pos'] = (G.nodes[station]['long'], G.nodes[station]['lat'])
        temp = temp + 1

    for node_x in G.nodes:
        for node_y in G.nodes:
            print(node_x)
            dist = distance_in_meters(G.nodes[node_x], G.nodes[node_y])
            if data_map[G.nodes[node_x]['ID'], G.nodes[node_y]['ID']] == 0:
                continue
            G.add_edge(node_x, node_y)
            G[node_x][node_y]['weight'] = np.exp(-dist)

    adj = nx.adjacency_matrix(G)  # 返回图的邻接矩阵
    # nx.write_gpickle(G, 'train_data/GCN_Dalian_Graph.gpickle', protocol=3)
    with open('train_data/GCN_Dalian_Graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    if figure:
        nx.draw(G, with_labels=True)
        # 提取pos属性作为坐标信息
        plt.show()
        pos = nx.get_node_attributes(G, 'pos')
        # 使用指定的坐标信息绘制图
        id_labels = nx.get_node_attributes(G, 'ID')
        nx.draw(G, pos, labels=id_labels, with_labels=True, node_color='skyblue')
        plt.show()
    return G, normalize_adj(adj).todense()  # todense稠密表示


def test_loadG(G_path='train_data/GCN_Dalian_Graph.pkl'):
    with open(G_path, 'rb') as f:
        G = pickle.load(f)
    print(G.edges(data=True))


if __name__ == '__main__':
    # data = get_dataset('Palo Alto')
    # get_graph(data)
    # setup_GCN(data, forecast_horizon=7, num_lags=30)
    data, data_map = get_dataset('Dalian')
    get_graph_Dalian(data, data_map)
    # test_loadG()
