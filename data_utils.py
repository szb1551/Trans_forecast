import re

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import networkx as nx
from tqdm import tqdm

"""
    将数据处理为5*5的格子，得到栅格地图
    Second
"""


def find_gridcell(row, x_list, y_list):  # 依据经纬度进行表格划分
    for i, x_cell in enumerate(x_list):
        if row['Longitude'] <= x_cell:
            x_grid_cell = i - 1  # [-1,4]   实际上不可能比最小的要小，所以为[0,4]
            break

    for i, y_cell in enumerate(y_list):
        if row['Latitude'] <= y_cell:
            y_grid_cell = i - 1
            break

    return x_grid_cell, y_grid_cell


def get_grid_cells(data, nx=5, ny=5):  # 对格子进行划分
    x_ = np.linspace(min(data['Longitude']) - 0.00005, max(data['Longitude']) + 0.00005, nx + 1)
    y_ = np.linspace(min(data['Latitude']) - 0.00005, max(data['Latitude']) + 0.00005, ny + 1)
    # print(x_)
    # print(y_)
    data['x_cell'] = -1
    data['y_cell'] = -1
    for index, row in data.iterrows():
        data.at[index, 'x_cell'], data.at[index, 'y_cell'] = find_gridcell(row, x_, y_)

    return data


def get_raster_map(data, normalized=False, verbose=False, nx=5, ny=5):  # 得到栅格地图
    number_of_hours = int((data.index.max() - data.index.min()).total_seconds() // (3600 * 24))  # 总共的天数
    raster_map = np.zeros([number_of_hours + 1, nx, ny])
    start_time = data.index.min()
    # print(data.index.max())  # 2020-12-31
    # print(data.index.min())  # 2011-7-29
    time_list = []
    num = re.compile(r"\d+")
    for i in range(0, number_of_hours + 1):
        timewindow_start = start_time + timedelta(days=i)  # 当前开始时间
        num_time = int(''.join(num.findall(str(timewindow_start))[:3]))
        time_list.append(num_time)
        current = data[(data.index == timewindow_start)]

        for x in range(0, nx):
            for y in range(0, ny):
                no_chargers = len(current[(current.x_cell == x) & (current.y_cell == y)]['ID'].unique())  # 充电站的数量
                if no_chargers == 0:
                    continue
                raster_map[i, x, y] = np.sum(
                    current[(current.x_cell == x) & (current.y_cell == y)]['Energy'])  # 栅格图的每个单位的能量之和

        if (verbose) and ((i % 50) == 0):
            print("Done with {} out of {}".format(i, number_of_hours))

    if normalized:  # 是否归一化
        max_ = np.max(raster_map, axis=0) + 0.01  # [5,5]
        # print(raster_map)
        # print(max_)
        # np.save('max_.npy', max_)
        raster_map = raster_map / max_  # [3444,5,5]
        # print(raster_map)
    # print(raster_map.shape)  # [3444, 5, 5]
    return raster_map, np.array(time_list)


def change_to_csv(data, name='predict'):
    import pandas as pd
    import numpy as np

    # 将numpy数组转换为pandas的DataFrame
    df = pd.DataFrame(data)

    # 将DataFrame保存为CSV文件
    df.to_csv('{}_data.csv'.format(name), index=False)  # 设置index=False来避免在csv文件中添加索引列


def generate_demand_related_random_number(distance):
    # 设定基础概率，确保所有情况下随机数都在[-0.2, 0.2]范围内
    base = np.random.uniform(-0.1, 0.2)

    # 根据距离调整倾斜程度，这里的调整系数可以根据实际需求调整
    # 距离越大，bias_factor 越小，随机数越倾向于取0.2的值
    bias_factor = 1 - np.tanh(distance)  # 这里使用tanh函数作为一个例子

    # 调整基础随机数，使其更倾向于0.2
    biased_random = base * bias_factor + 0.2 * (1 - bias_factor)

    return biased_random


# 交通流转电氢负荷
def flow_to_ele_hyd(flow_path="predict_data.csv"):
    data = pd.read_csv(flow_path)
    flow_thing = data.to_numpy()
    # 交通流转电氢负荷相应超参数
    Ele_hyd = 2
    Electric_period = 0.4
    Hydrogen_period = 0.1
    Electric_power = 94.3  # （kwh）
    Hydrogen_power = 122.7  # （L）
    demand_pro = 0.2  # 充电需求概率
    once_elec = 0.7  # 一次充的电量
    once_hyd = 0.9  # 一次加氢的比例，氢和油的时间差不多

    with open('train_data/GCN_Dalian_Graph.pkl', 'rb') as f:
        G = pickle.load(f)
    adj = nx.adjacency_matrix(G).todense()  # 返回图的邻接矩阵

    elec_nodes = np.zeros([flow_thing.shape[0], flow_thing.shape[1]])  # [time, nodes]
    hyd_nodes = np.zeros([flow_thing.shape[0], flow_thing.shape[1]])
    for i in tqdm(range(flow_thing.shape[1])):

        demand_elec_node = np.zeros(flow_thing.shape[0])
        demand_hyd_node = np.zeros(flow_thing.shape[0])
        for j in range(flow_thing.shape[0]):
            rand = generate_demand_related_random_number(adj[i].sum(axis=-1))
            tmp_elec = (demand_pro + rand) * Electric_power * Electric_period * (once_elec + rand)
            tmp_hyd = (demand_pro + rand) * Hydrogen_power * Hydrogen_period * (once_hyd + rand)
            demand_elec_node[j] = tmp_elec
            demand_hyd_node[j] = tmp_hyd

        elec_nodes[:, i] = flow_thing[:, i] * demand_elec_node
        hyd_nodes[:, i] = flow_thing[:, i] * demand_hyd_node
    change_to_csv(elec_nodes, name='elec')
    change_to_csv(hyd_nodes, name='hyd')


# 读取电氢csv数据，并转为numpy矩阵,数据类型为[T,Node,2]
# 转为[Node, T, 2]
# 先elec 再 hyd
def read_csv2numpy(elec_path, hyd_path):
    data = pd.read_csv(elec_path)
    elec_demand = data.to_numpy()
    data = pd.read_csv(hyd_path)
    hyd_demand = data.to_numpy()
    elec_with_hyd = np.dstack((elec_demand, hyd_demand))
    print(elec_with_hyd.shape)
    elec_with_hyd = np.transpose(elec_with_hyd, (1,0,2))
    return elec_with_hyd.astype(np.float32)

# 获取时序信息表，采样时间为每15分钟。
def get_time_list(elec_path):
    data = pd.read_csv(elec_path)
    elec_demand = data.to_numpy()
    all_time = len(elec_demand)
    time_list = np.arange(0, all_time)
    return time_list

if __name__ == "__main__":
    # flow_to_ele_hyd()
    read_csv2numpy(elec_path="data/数据源/elec_data.csv", hyd_path="data/数据源/hyd_data.csv")
