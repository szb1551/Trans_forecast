# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import scipy.sparse as sp
from datetime import timedelta
from math import sin, cos, sqrt, atan2, radians, floor
from ProcessData import process_time
from data_handler import get_dataset
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
import re
import xml.etree.ElementTree as ET

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')
"""
将数据data处理为图结构
"""
Ele_hyd = 2
Electric_period = 0.4
Hydrogen_period = 0.1
Electric_power = 94.3  # （kwh）
Hydrogen_power = 122.7  # （L）
demand_pro = 0.2  # 充电需求概率
once_elec = 0.7  # 一次充的电量
once_hyd = 0.9  # 一次加氢的比例，氢和油的时间差不多


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
    G = nx.DiGraph()
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
            # G[node_x][node_y]['weight'] = dist
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


def get_graph_Dalian_Deal(data, data_map, forecast_horizon, num_lags, time=False):
    G, adj = get_graph_Dalian(data, data_map)
    # 总天数的数量，这地方可以尝试修改特化一下
    number_of_hours = int((data.index.max() - data.index.min()).total_seconds() // 3600)
    timeseries_ = np.zeros([len(G.nodes()), number_of_hours + 1, Ele_hyd])  # 节点数与天数，记录当前节点当天的能量和
    start_time = data.index.min()
    time_list = []  # 用来加载时间
    num = re.compile(r"\d+")
    for i in range(0, number_of_hours + 1):
        timewindow_start = start_time + timedelta(hours=i)
        # num_time = int(''.join(num.findall(str(timewindow_start))[:3]))
        # time_list.append(num_time)
        current = data[(data.index == timewindow_start)]

        for k, node in enumerate(G.nodes()):
            tmp = current[G.nodes[node]['ID'] == current['ID']]
            timeseries_[k, i, 0] = np.sum(tmp['traffic flow']) * Electric_period * Electric_power
            timeseries_[k, i, 1] = np.sum(tmp['traffic flow']) * Hydrogen_period * Hydrogen_power

    max_ = np.max(timeseries_, axis=-1)[:, None]
    # np.save('train_data/G_max_.npy', max_)
    normalized = timeseries_ / max_

    NUM_LAGS = num_lags

    n_nodes = len(G.nodes())

    matrix_lags = np.zeros(
        (timeseries_.shape[1] - (NUM_LAGS + forecast_horizon), timeseries_.shape[0], NUM_LAGS + forecast_horizon,
         Ele_hyd))
    STEPS_AHEAD = math.floor(matrix_lags.shape[0] * 0.2)
    i_train = matrix_lags.shape[0] - STEPS_AHEAD
    i_test = matrix_lags.shape[0]

    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = normalized[:, i:i + NUM_LAGS + forecast_horizon]

    # ---------------- Train/test split
    train_dataset = matrix_lags[:i_train]  # [batch, 47, 37]
    test_dataset = matrix_lags[i_train:]  # [batch, 47, 37]

    print('GCN_train_dataset:', train_dataset.shape)  # [batch, nodes, time_series, Ele and hye]
    print('GCN_test_dataset:', test_dataset.shape)
    if time:
        # timelist = np.load('train_data/time_list.npy')
        time_list = np.array(time_list)
        X_train_time, X_test_time = process_time(time_list, train_length=num_lags, forcast_window=forecast_horizon)
        return train_dataset, test_dataset, X_train_time, X_test_time
    return adj, train_dataset, test_dataset


# 按概率生成一个介于[-0.1, 0.1]的随机数，
# 概率越接近1，数值越接近0.1
def generate_random_number(probability_close_to_one):
    if not (0 <= probability_close_to_one <= 1):
        raise ValueError("Probability must be between 0 and 1.")

    # Generate a random float in the range [0, 1)
    rand_val = np.random.rand()

    # If the random value is less than the specified probability,
    # we lean towards 0.1; otherwise, we lean towards -0.1.
    if rand_val < probability_close_to_one:
        # Generate random float in [0, 0.1]
        return np.random.uniform(0, 0.1)
    else:
        # Generate random float in [-0.1, 0]
        return np.random.uniform(-0.1, 0)


def get_graph_Dalian_xml(data_xml, sim_xml, time=False):
    tree = ET.parse(data_xml)
    sim = ET.parse(sim_xml)
    root = tree.getroot()
    sim_root = sim.getroot()
    timeseries_ = np.zeros([150, 2])  # 【节点， 车流量，长度】
    max_length = 0
    edges = {}
    for edge in root.findall('.//edge'):
        edges[edge.get('id')] = float(edge.find('lane').get('length'))
        max_length = max(max_length, float(edge.find('lane').get('length')))
    routes = sim_root.findall('./route')
    map = dict()
    print(routes)
    for id in edges.keys():
        for route in routes:
            if route.get('edges') == id:
                map[id] = route.get('id')
    for id, value in map.items():
        print(id, value)
        begin = int(id.split('to')[0])
        end = int(id.split('to')[-1])

        pro = edges[id] / max_length
        rand = generate_random_number(pro)
        tmp_elec = len(sim_root.findall("./vehicle[@route='{}']".format(value))) * (
                    demand_pro + rand) * Electric_power * Electric_period * (once_elec + rand)
        tmp_hyd = len(sim_root.findall("./vehicle[@route='{}']".format(value))) * (
                    demand_pro + rand) * Hydrogen_power * Hydrogen_period * (once_hyd + rand)
        timeseries_[begin - 1][0] = timeseries_[begin - 1][0] + tmp_elec  # 统计所有的车流量的电负荷
        timeseries_[end - 1][0] = timeseries_[end - 1][0] + tmp_elec
        timeseries_[begin - 1][1] = timeseries_[begin - 1][1] + tmp_hyd  # 统计车流量的氢负荷
        timeseries_[end - 1][1] = timeseries_[end - 1][1] + tmp_hyd

    print(timeseries_)  # 返回路口节点的车流量和路口的路径长度？
    return timeseries_


# 处理所有的xml的交通流数据
def get_graph_Dalian_xml_days(data_xml='TrafficSim/out_day.xml', net_xml='TrafficSim/network.net.xml', time=False):
    tree = ET.parse(data_xml)  # 打开交通流的检测其xml
    net = ET.parse(net_xml)  # 打开network的xml文件
    root = tree.getroot()
    net_root = net.getroot()
    with open('train_data/GCN_Dalian_Graph.pkl', 'rb') as f:
        G = pickle.load(f)
    adj = nx.adjacency_matrix(G)  # 返回图的邻接矩阵
    max_length = 0
    edges = {}
    for edge in net_root.findall('.//edge'):
        edges[edge.get('id')] = float(edge.find('lane').get('length'))
        max_length = max(max_length, float(edge.find('lane').get('length')))
    detectors = root.findall('./interval')
    nodes = net.findall('./junction')
    last_dectector_time = detectors[-1].get('end')
    time_step = 900.00
    all_time_tags = int(float(last_dectector_time) / time_step)
    print(all_time_tags)
    timeseries_ = np.zeros([len(nodes), all_time_tags, 2])  # 【节点， 车流量的所有时间戳，电氢负荷】
    start_time = 0.00
    time_list = []
    for id, value in enumerate(detectors):
        dector_id = value.get('id')
        begin = int(dector_id.split('_')[1].split('to')[0])
        end = int(dector_id.split('_')[1].split('to')[-1])
        edge_id = dector_id.split('_')[1]
        pro = edges[edge_id] / max_length
        rand = generate_random_number(pro)
        begin_time = float(value.get('begin'))
        index = floor(begin_time / time_step)
        vec_num = int(value.get('nVehContrib'))
        tmp_elec = vec_num * (
                demand_pro + rand) * Electric_power * Electric_period * (once_elec + rand)
        tmp_hyd = vec_num * (
                demand_pro + rand) * Hydrogen_power * Hydrogen_period * (once_hyd + rand)
        timeseries_[begin - 1][index][0] = timeseries_[begin - 1][index][0] + tmp_elec  # 统计所有的车流量的电负荷
        timeseries_[end - 1][index][0] = timeseries_[end - 1][index][0] + tmp_elec
        timeseries_[begin - 1][index][1] = timeseries_[begin - 1][index][0] + tmp_hyd  # 统计车流量的氢负荷
        timeseries_[end - 1][index][1] = timeseries_[end - 1][index][1] + tmp_hyd
        if len(time_list)==0 or index*time_step>time_list[-1]:
            time_list.append(index*time_step)
    print(timeseries_.shape)

    if time:
        return normalize_adj(adj).todense(), timeseries_, np.array(time_list)
    return normalize_adj(adj).todense(), timeseries_


def test_loadG(G_path='train_data/GCN_Dalian_Graph.pkl'):
    with open(G_path, 'rb') as f:
        G = pickle.load(f)
    # print(G.edges(data=True))
    edges_with_things = G.edges(data=True)
    for nodex, nodey, weight in edges_with_things:
        print(nodex, nodey, weight['weight'])

if __name__ == '__main__':
    # setup_GCN(data, forecast_horizon=7, num_lags=30)
    data, data_map = get_dataset('Dalian')
    get_graph_Dalian(data, data_map)
    # test_loadG()
    # get_graph_Dalian_xml("network.net.xml", "vehicle_routes.rou.xml")
    # get_graph_Dalian_xml_days()
