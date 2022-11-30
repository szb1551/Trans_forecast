import re

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


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