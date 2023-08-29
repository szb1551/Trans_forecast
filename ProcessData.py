from data_utils import *
import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
# from joblib import dump
from data_handler import get_dataset
import re


def split_batch(X_list, y_list, batch_size):  # 对数据进行分batch操作
    X_list = X_list.numpy()  # [3400, 30, 1]
    y_list = y_list.numpy()  # [3400, 7, 1]
    idx_list = np.arange(0, len(X_list), batch_size)  # [0, 40, 80...]
    batch_indexs = []

    for idx in idx_list:
        batch_indexs.append(np.arange(idx, min(idx + batch_size, len(X_list))))
    batches = []
    for batch_index in batch_indexs:
        batch_train = torch.tensor([X_list[index] for index in batch_index])  # [40, 30, 1]
        batch_test = torch.tensor([y_list[index] for index in batch_index])  # [40, 7,  1]
        batches.append([batch_train, batch_test])  # [[40, 30, 1], [40, 7, 1]]
    return batches


def process_data(csv_name, training_length, forecast_window, batch_size=None, nx=5, ny=5):  # 得到初始预处理数据
    raster_map, _ = get_raster_map(get_dataset(csv_name))  # 得到栅格地图 第一个维度为时间跨度 # [3444, 5, 5]   #2020-12-31
    raster_map = raster_map.sum(axis=-1).sum(axis=-1)
    max_ = np.max(raster_map, axis=0)
    np.save('max_.npy', max_)
    normalized = raster_map / max_
    print('normalized.shape:', normalized.shape)
    # normalized = raster_map
    normalized[np.isnan(normalized)] = 0
    # normalized[normalized == 0] = np.random.normal(np.zeros_like(normalized[normalized == 0]), 0.01)  # 无效值赋初值
    # matrix_lags = [3407,37,5,5]   # 2020-11-24
    matrix_lags = np.zeros(
        (normalized.shape[0] - (training_length + forecast_window), training_length + forecast_window))

    print('matrix_lags.shape:', matrix_lags.shape)  # [3407. 37]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = normalized[i:i + training_length + forecast_window]  # 批次 + num_lags+prediction_horizont天数

    # ---------------- Train/test split
    X_train = np.zeros((i_train, training_length))
    y_train = np.zeros((i_train, forecast_window))
    X_test = np.zeros((i_test - i_train, training_length))
    y_test = np.zeros((i_test - i_train, forecast_window))

    X_train = matrix_lags[:i_train, :training_length]  # [3400, 30]
    y_train = matrix_lags[:i_train, training_length:]  # [3400, 7]
    X_test = matrix_lags[i_train:i_test, :training_length]  # [7, 30]
    y_test = matrix_lags[i_train:i_test, training_length:]  # [7,7]

    X_train = torch.tensor(X_train).unsqueeze(-1)
    y_train = torch.tensor(y_train).unsqueeze(-1)
    X_test = torch.tensor(X_test).unsqueeze(-1)
    y_test = torch.tensor(y_test).unsqueeze(-1)

    print('X_train.shape:', X_train.shape)  # [3400,30,1]
    print('y_train.shape:', y_train.shape)  # [3400,7,1]
    print('X_test.shape:', X_test.shape)  # [7,30,1]
    print('y_test.shape:', y_test.shape)  # [7,7,1]
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # X_train = torch.tensor(scaler.transform(X_train)).unsqueeze(-1)  # [3400,30,1]
    # y_train = torch.tensor(scaler.transform(y_train).unsqueeze(-1))  # [3400,8,1]
    # X_test = torch.tensor(scaler.transform(X_test).unsqueeze(-1)) #[7,30,1]
    # y_test = torch.tensor(scaler.transform(y_test).unsqueeze(-1))

    # dump(scaler, "scaler_item.joblib")
    if batch_size:
        batches_train = split_batch(X_train, y_train, batch_size=batch_size)
        batches_test = split_batch(X_test, y_test, batch_size=batch_size)

        return batches_train, batches_test
    else:
        return X_train, y_train, X_test, y_test


def process_data2(batch_size):  # 存储后进行直接加载，省的每次都要处理数据
    # load directly after the save, saving the time for the processing the data.
    scaler = MinMaxScaler()
    X_train = torch.tensor(np.load('X_train.npy'))
    y_train = torch.tensor(np.load('y_train.npy'))
    X_test = torch.tensor(np.load('X_test.npy'))
    y_test = torch.tensor(np.load('y_test.npy'))

    # X_train = torch.tensor(scaler.fit_transform(X_train)).unsqueeze(-1)
    # y_train = torch.tensor(scaler.fit_transform(y_train)).unsqueeze(-1)  # [3400,8,1]
    # X_test = torch.tensor(scaler.fit_transform(X_test)).unsqueeze(-1)
    # y_test = torch.tensor(scaler.fit_transform(y_test)).unsqueeze(-1)

    # dump(scaler, "scaler_item.joblib")
    batches_train = split_batch(X_train, y_train, batch_size=batch_size)
    batches_test = split_batch(X_test, y_test, batch_size=batch_size)
    return batches_train, batches_test


def process_data3(csv_name, training_length, forecast_window, time=False):  # 得到初始预处理数据
    raster_map, time_list = get_raster_map(get_dataset(csv_name))  # 得到栅格地图 第一个维度为时间跨度 # [3444, 5, 5]   #2020-12-31
    print(type(time_list))
    raster_map = raster_map.sum(axis=-1).sum(axis=-1)
    max_ = np.max(raster_map, axis=0)
    np.save('max_.npy', max_)
    # normalized = raster_map
    normalized = raster_map / max_
    print('normalized.shape:', normalized.shape)
    normalized[np.isnan(normalized)] = 0
    normalized[normalized == 0] = np.random.normal(np.zeros_like(normalized[normalized == 0]), 0.001)  # 无效值赋初值
    # matrix_lags = [3407,37,5,5]   # 2020-11-24
    matrix_lags = np.zeros(
        (normalized.shape[0] - (training_length + forecast_window), training_length + forecast_window))

    print('matrix_lags.shape:', matrix_lags.shape)  # [3407. 37]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = normalized[i:i + training_length + forecast_window]  # 批次 + num_lags+prediction_horizont天数

    # ---------------- Train/test split
    train_dataset = np.zeros((i_train, training_length + forecast_window))  # [3400, 37]
    test_dataset = np.zeros((i_test - i_train, forecast_window + training_length))  # [7, 37]

    train_dataset = matrix_lags[:i_train, :]
    test_dataset = matrix_lags[i_train:, :]

    # np.save('train_dataset.npy', train_dataset)
    # np.save('test_dataset.npy', test_dataset)
    print('train_dataset:', train_dataset.shape)
    print('test_dataset:', test_dataset.shape)
    if time:
        X_train_time, X_test_time = process_time(time_list, training_length, forecast_window)
        return train_dataset, test_dataset, X_train_time, X_test_time
    return train_dataset, test_dataset


def process_data4(csv_name, training_length, forecast_window):  # 得到初始预处理数据
    raster_map, time_list = get_raster_map(get_dataset(csv_name))  # 得到栅格地图 第一个维度为时间跨度 # [3444, 5, 5]   #2020-12-31
    raster_map = raster_map.sum(axis=-1).sum(axis=-1)
    max_ = np.max(raster_map, axis=0)
    np.save('max_.npy', max_)
    # normalized = raster_map
    normalized = raster_map / max_
    print('normalized.shape:', normalized.shape)
    normalized[np.isnan(normalized)] = 0
    # normalized[normalized == 0] = np.random.normal(np.zeros_like(normalized[normalized == 0]), 0.01)  # 无效值赋初值
    # matrix_lags = [all_batch,37,5,5]   # 2020-11-24
    matrix_lags = np.zeros(
        (normalized.shape[0] - (training_length + 1), training_length + 1))

    print('matrix_lags.shape:', matrix_lags.shape)  # [all_batch. 31]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = normalized[i:i + training_length + 1]  # 批次 + num_lags+prediction_horizont天数

    # ---------------- Train/test split
    train_dataset = np.zeros((i_train, training_length + 1))  # [3400, 31]
    test_dataset = np.zeros((i_test - i_train, 1 + training_length))  # [7, 31]

    train_dataset = matrix_lags[:i_train, :]
    test_dataset = matrix_lags[i_train:, :]

    np.save('train_dataset.npy', train_dataset)
    np.save('test_dataset.npy', test_dataset)
    print('train_dataset:', train_dataset.shape)
    print('test_dataset:', test_dataset.shape)
    return train_dataset, test_dataset


def process_time(time_list, train_length, forcast_window,
                 matrix=False):  # 得到栅格地图 第一个维度为时间跨度 # [3444, 5, 5]   #2020-12-31
    print('time_list:', time_list.shape)
    all_time_lags = np.zeros(
        (time_list.shape[0] - (train_length + forcast_window), train_length + forcast_window), dtype=int)
    print('all_time_lags.shape:', all_time_lags.shape)  # [3407. 37]
    for i in range(all_time_lags.shape[0]):
        all_time_lags[i] = time_list[i: i + train_length + forcast_window]  # 批次 + num_lags+prediction_horizont天数
    if matrix:
        return all_time_lags
    # np.save('train_data_matrix/time_matrix_lags.npy', all_time_lags)
    # np.save('train_data_matrix/time_long_matrix_lags.npy', all_time_lags)
    i_train = all_time_lags.shape[0] - forcast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = all_time_lags.shape[0]

    X_train_time = all_time_lags[:i_train, :]  # [3400, 37]
    X_test_time = all_time_lags[i_train:i_test, :]  # [7, 37]
    return X_train_time, X_test_time


class process_all_time:
    def __init__(self, time_list, train_length, forecast_window):
        self.time_list = time_list
        self.train_length = train_length
        self.forecast_window = forecast_window

    def get_matrix_time(self):
        all_time_lags = np.zeros(
            (self.time_list.shape[0] - (self.train_length + self.forecast_window),
             self.train_length + self.forecast_window), dtype=int)
        for i in range(all_time_lags.shape[0]):
            all_time_lags[i] = self.time_list[i: i + self.train_length + self.forecast_window]
        return all_time_lags

    def get_process_time(self):
        print('time_list:', self.time_list.shape)
        all_time_lags = np.zeros(
            (self.time_list.shape[0] - (self.train_length + self.forecast_window),
             self.train_length + self.forecast_window), dtype=int)
        print('all_time_lags.shape:', all_time_lags.shape)  # [3407. 37]
        for i in range(all_time_lags.shape[0]):
            all_time_lags[i] = self.time_list[
                               i: i + self.train_length + self.forecast_window]  # 批次 + num_lags+prediction_horizont天数
        i_train = all_time_lags.shape[
                      0] - self.forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
        i_test = all_time_lags.shape[0]

        X_train_time = all_time_lags[:i_train, :]  # [3400, 37]
        X_test_time = all_time_lags[i_train:i_test, :]  # [7, 37]
        return X_train_time, X_test_time
