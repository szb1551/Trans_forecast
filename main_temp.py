from graph_utils import get_graph_Dalian_xml_days, normalize_adj
import numpy as np
from ProcessData import process_time
import math
from DataLoader import SensorDataset_GCN
from torch.utils.data import DataLoader
from models.model_utils import select_model, get_model_args
import torch
import os
from collections import deque
import datetime, time
import matplotlib.pyplot as plt
from models.utils import Rp_num_den, plot_Dalian_gcn_models, mape, rmse_diff
import pickle
import networkx as nx
from data_utils import read_csv2numpy, get_time_list
from tqdm import tqdm
from torch.utils.data import random_split
from models.train_models import Train_all_models

criterion = torch.nn.MSELoss()
lr = 0.0001
epochs = 10000
train_epoch_loss = []
Rp_best = 10
idx_example = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_num = 2
filenames = deque(maxlen=model_num)


# 模型的保存,只保存一定数量的模型
def save_model(model, e, train_length, Rp, file_name="train_process", model_name="GCN_Dalian"):
    save_name = file_name + '/model{}_epoch{}_length{}_Rp_{:.4f}'.format(model_name, e, train_length, Rp)
    torch.save(model, save_name + '.pkl')
    if len(filenames) >= model_num:
        os.remove(filenames[0] + '.pkl')
    filenames.append(save_name)


def save_loss(args):
    plt.figure(figsize=(10, 10))
    plt.plot(train_epoch_loss)
    plt.legend(['Train Loss'], fontsize=25)
    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("MSE Loss", fontsize=25)
    plt.grid()
    plt.title('{}_loss_length{}'.format(args['name'], args['train_length']))
    plt.savefig('train_result/{}_{}_loss_length{}.jpg'.format(args['name'], args['lr'], args['train_length']))
    np.savetxt('train_result/{}_{}_loss_length{}.txt'.format(args['name'], args['lr'], args['train_length']),
               train_epoch_loss, fmt='%6f')


def train_gcn_epoch(model, train_dl, optimizer):
    model.train()
    train_loss = 0
    n = 0
    for step, (x, src, tar, times) in tqdm(enumerate(train_dl)):
        # [batch, time]
        tar_in = tar[:, :, :-1]
        tar_out = tar[:, :, 1:]  # [batch nodes, time ,...]
        optimizer.zero_grad()
        output = model(x.to(device).float(), src.to(device).float(), tar_in.to(device))
        # output [batch, forcast_window, 1]
        loss = criterion(output.squeeze(-1), tar_out.to(device))  # not missing data
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n


def test_GCN_Transform(model, test_dl, forcast_window=7):
    # Return Rp, RMSE　MAPE
    with torch.no_grad():
        predictions = []
        observations = []

        model.eval()
        for step, (x, src, tar, times) in enumerate(test_dl):
            tar_in = tar[:, :, :1]
            tar_out = tar[:, :, 1:]
            next_input = tar_in
            all_predictions = []

            for i in range(forcast_window):
                output = model(x.to(device), src.to(device), next_input.to(device))  # [batch, Node, time, e_H]
                if all_predictions == []:
                    all_predictions = output[:, :, -1].unsqueeze(-2).detach()
                else:
                    all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-2).detach()), dim=-2)
                next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-2).detach()), dim=-2)
            for p, o in zip(all_predictions.cpu().numpy().sum(axis=1),
                            tar_out.numpy().sum(axis=1)):  # not missing data[batch, time ,e_H]
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        num2 = 0
        den2 = 0
        num_n = 0
        diff = 0
        mape_add = 0
        diff2 = 0
        mape_add2 = 0
        for y_preds, y_trues in zip(predictions, observations):
            if len(y_preds.shape) < 3:
                y_preds = y_preds[:, np.newaxis, :]
                y_trues = y_trues[:, np.newaxis, :]
            num_i_1, den_i_1 = Rp_num_den(y_preds[:, :, 0], y_trues[:, :, 0], .5)
            num_i_2, den_i_2 = Rp_num_den(y_preds[:, :, 1], y_trues[:, :, 1], .5)
            num += num_i_1
            den += den_i_1
            num2 += num_i_2
            den2 += den_i_2
            diff_i, n_i = rmse_diff(y_preds[:, :, 0], y_trues[:, :, 0])
            diff += diff_i
            mape_i, _ = mape(y_preds[:, :, 0], y_trues[:, :, 0])
            mape_add += mape_i
            num_n += n_i
            diff_i2, _ = rmse_diff(y_preds[:, :, 1], y_trues[:, :, 1])
            diff2 += diff_i2
            mape_i2, _ = mape(y_preds[:, :, 1], y_trues[:, :, 1])
            mape_add2 += mape_i2
        Rp1 = (2 * num) / den
        Rp2 = (2 * num2) / den2
        RMSE1 = np.sqrt(diff / num_n)
        MAPE1 = mape_add / num_n
        RMSE2 = np.sqrt(diff2 / num_n)
        MAPE2 = mape_add2 / num_n
    return Rp1, RMSE1[0], MAPE1[0], Rp2, RMSE2[0], MAPE2[0]


def get_dataset(dataset, time_list, train_proportion=0.8, random=False, save=False):
    num_train = int(dataset.shape[0] * train_proportion)
    num_test = dataset.shape[0] - num_train
    if random:
        torch.manual_seed(0)  # 设置随机种子
        train, test = random_split(dataset, [num_train, num_test])
        print(train.indices)
        print(test.indices)
        train_dataset = dataset[train.indices]
        test_dataset = dataset[test.indices]
        X_train_time = time_list[train.indices]
        X_test_time = time_list[test.indices]
    else:
        train_dataset = dataset[:num_train]
        test_dataset = dataset[num_train:]
        X_train_time = time_list[:num_train]
        X_test_time = time_list[num_train:]
    if save:
        np.save('split_data/{}_long_train_dataset.npy'.format("Dalian"), train_dataset, allow_pickle=True)
        np.save('split_data/{}_long_test_dataset.npy'.format("Dalian"), test_dataset, allow_pickle=True)
        np.save('split_data/{}_long_train_timeset.npy'.format("Dalian"), X_train_time)
        np.save('split_data/{}_long_test_timeset.npy'.format("Dalian"), X_test_time)
    return train_dataset, test_dataset, X_train_time, X_test_time


def process_dalian_gcn_data(G_timeseries_map, time_list, training_length, forecast_window,
                            time=False, matrix=False, random=False):  # gcn 通过给定数组进行dataset分类
    # G_timeseries_map [150, 96, 2]
    matrix_lags = np.zeros(
        (G_timeseries_map.shape[1] - (training_length + forecast_window),
         G_timeseries_map.shape[0], training_length + forecast_window, G_timeseries_map.shape[-1]), dtype=np.float32)
    print('G_timeseries_map.shape:', G_timeseries_map.shape)
    print('matrix_lags.shape:', matrix_lags.shape)  # [B. Node, T, e_H]

    i_train = math.floor(matrix_lags.shape[0] * 0.8)  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0] - i_train
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = G_timeseries_map[:, i:i + training_length + forecast_window]  # 批次 +
        # num_lags+prediction_horizont天数
    if matrix:
        return matrix_lags, process_time(time_list, training_length, forecast_window, matrix=True)
    if random:
        train_dataset, test_dataset, X_train_time, X_test_time = get_dataset(matrix_lags,
                                                                             process_time(time_list, training_length,
                                                                                          forecast_window, matrix=True),
                                                                             random=random)
        print('train_dataset:', train_dataset.shape)
        print('test_dataset:', test_dataset.shape)
        return train_dataset, test_dataset, X_train_time, X_test_time

    train_dataset = matrix_lags[:i_train]  # [batch, Node, T, e_H]
    test_dataset = matrix_lags[i_train:]  # [batch, Node, T, e_H]
    print('train_dataset:', train_dataset.shape)
    print('test_dataset:', test_dataset.shape)
    if time:
        X_train_time, X_test_time = process_time(time_list, training_length, forecast_window)
        return train_dataset, test_dataset, X_train_time, X_test_time
    return train_dataset, test_dataset


def Train_Dalian_Gcn_Transform(Rp_best, data_xml, net_xml, train_length, forecast_window, normalize=True):
    A, G_timeseries_map, time_list = get_graph_Dalian_xml_days(data_xml, net_xml, time=True)
    A = torch.tensor(A, dtype=torch.float32).to(device)
    if normalize:
        max_ = np.amax(G_timeseries_map, axis=(0, 1))
        G_timeseries_map = G_timeseries_map / max_
        G_timeseries_map[G_timeseries_map == 0] = np.random.normal(
            np.zeros_like(G_timeseries_map[G_timeseries_map == 0]),
            0.001)
    train_dataset, test_dataset, X_train_time, X_test_time = \
        process_dalian_gcn_data(G_timeseries_map, time_list, training_length=train_length,
                                forecast_window=forecast_window,
                                time=True)

    train_data = SensorDataset_GCN(train_dataset, X_train_time, train_length, forecast_window)
    test_data = SensorDataset_GCN(test_dataset, X_test_time, train_length, forecast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, Nodes, Times]
    test_dl = DataLoader(test_data, batch_size=1)
    args = get_model_args(name='Transformer_gcn_Dalian', train_length=train_length, forcast_window=forecast_window,
                          X_train=train_dataset, adj_matrix=A)
    model = select_model('Transformer_gcn_Dalian', args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    files = os.listdir("train_process/")
    if len(files):  # 初始化文件夹里面的文件命
        for file in files:
            if (file.split('.')[-1] == 'pkl' and file.split('_')[1] != 'last'):
                filenames.append("train_process/" + file.replace('.pkl', ''))
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(args['epochs'])):
        start = time.time()
        train_loss = []

        l_t = train_gcn_epoch(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp1, RMSE1, MAPE1, Rp2, RMSE2, MAPE2 = test_GCN_Transform(model, test_dl, forecast_window)

        # if Rp_best > Rp1:
        #     Rp_best = Rp1
        if e + 1 % 10 == 0:
            save_model(model, e + 1, train_length, Rp1)
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print(
            "Epoch {}: Train loss: {:.6f} \t R_p1={:.3f}, RMSE1={:.3f}, MAPE1={:.3f}, R_p2={:.3f}, RMSE2={:.3f}, MAPE2={:.3f} \tcost_time={:.4f}s".format(
                e + 1,
                np.mean(train_loss), Rp1, RMSE1, MAPE1, Rp2, RMSE2, MAPE2, end - start))
    print('结束训练时间:', datetime.datetime.now())
    torch.save(model, 'train_process/model_last_length{}.pkl'.format(train_length))
    save_loss(args)
    plot_Dalian_gcn_models(model, train_dl, train_length, forecast_window)
    plot_Dalian_gcn_models(model, test_dl, train_length, forecast_window, test=True)

    # attn_layers = get_gcn_attn(model, test_dl.dataset[idx_example][0].unsqueeze(0),
    #                            test_dl.dataset[idx_example][1].unsqueeze(0),
    #                            test_dl.dataset[idx_example][2].unsqueeze(0))
    # show_gcn_attn(test_data, attn_layers)


def Test_Dalian_Gcn_Transform(data_xml, net_xml, train_length, forecast_window, normalize=True):
    A, G_timeseries_map, time_list = get_graph_Dalian_xml_days(data_xml, net_xml, time=True)
    A = torch.tensor(A, dtype=torch.float32).to(device)
    if normalize:
        max_ = np.amax(G_timeseries_map, axis=(0, 1))
        G_timeseries_map = G_timeseries_map / max_
        G_timeseries_map[G_timeseries_map == 0] = np.random.normal(
            np.zeros_like(G_timeseries_map[G_timeseries_map == 0]),
            0.001)
    train_dataset, test_dataset, X_train_time, X_test_time = \
        process_dalian_gcn_data(G_timeseries_map, time_list, training_length=train_length,
                                forecast_window=forecast_window,
                                time=True)

    train_data = SensorDataset_GCN(train_dataset, X_train_time, train_length, forecast_window)
    test_data = SensorDataset_GCN(test_dataset, X_test_time, train_length, forecast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, Nodes, Times]
    test_dl = DataLoader(test_data, batch_size=1, shuffle=True)

    model = torch.load("train_process/model_last_length30.pkl")

    plot_Dalian_gcn_models(model, train_dl, train_length, forecast_window, max_=max_)
    plot_Dalian_gcn_models(model, test_dl, train_length, forecast_window, test=True, max_=max_)


# 读取电氢负荷数据，并进行训练预测
def Train_Dalian_Gcn_Transform_data(Rp_best, elec_path, hyd_path, graph_path, train_length, forecast_window,
                                    normalize=True, load=False):
    G_timeseries_map = read_csv2numpy(elec_path, hyd_path)  # [N,all_T 2]
    time_list = get_time_list(elec_path)
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    A = nx.adjacency_matrix(G)  # 返回图的邻接矩阵
    A = torch.tensor(A, dtype=torch.float32).to(device)
    if normalize:
        max_ = np.amax(G_timeseries_map, axis=(0, 1))
        G_timeseries_map = G_timeseries_map / max_
        G_timeseries_map[G_timeseries_map == 0] = np.random.normal(
            np.zeros_like(G_timeseries_map[G_timeseries_map == 0]),
            0.001)
    train_dataset, test_dataset, X_train_time, X_test_time = \
        process_dalian_gcn_data(G_timeseries_map, time_list, training_length=train_length,
                                forecast_window=forecast_window,
                                time=True, random=True)  # [B, N ,T=trian+forcast, 2]

    train_data = SensorDataset_GCN(train_dataset, X_train_time, train_length, forecast_window)
    test_data = SensorDataset_GCN(test_dataset, X_test_time, train_length, forecast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, Nodes, Times]
    test_dl = DataLoader(test_data, batch_size=1)
    args = get_model_args(name='Transformer_gcn_Dalian', train_length=train_length, forcast_window=forecast_window,
                          X_train=train_dataset, adj_matrix=A)
    if load:
        print("模型加载成功")
        model = torch.load("train_process/model_last_length30.pkl")
    else:
        model = select_model('Transformer_gcn_Dalian', args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    files = os.listdir("train_process/")
    if len(files):  # 初始化文件夹里面的文件命
        for file in files:
            if file.split('.')[-1] == 'pkl' and file.split('_')[1] != 'last':
                filenames.append("train_process/" + file.replace('.pkl', ''))
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(args['epochs'])):
        start = time.time()
        train_loss = []

        l_t = train_gcn_epoch(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp1, RMSE1, MAPE1, Rp2, RMSE2, MAPE2 = test_GCN_Transform(model, test_dl, forecast_window)

        if Rp_best > Rp1:
            Rp_best = Rp1
            save_model(model, e + 1, train_length, Rp1)
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print(
            "Epoch {}: Train loss: {:.6f} \t R_p1={:.3f}, RMSE1={:.3f}, MAPE1={:.3f}, R_p2={:.3f}, RMSE2={:.3f}, MAPE2={:.3f} \tcost_time={:.4f}s".format(
                e + 1,
                np.mean(train_loss), Rp1, RMSE1, MAPE1, Rp2, RMSE2, MAPE2, end - start))
    print('结束训练时间:', datetime.datetime.now())
    torch.save(model, 'train_process/model_last_length{}.pkl'.format(train_length))
    save_loss(args)
    plot_Dalian_gcn_models(model, train_dl, train_length, forecast_window)
    plot_Dalian_gcn_models(model, test_dl, train_length, forecast_window, test=True)


def Train_Dalian_Test(Rp_best, elec_path, hyd_path, graph_path, train_length, forecast_window,
                      normalize=True, load=False, name="Transformer_gcn"):
    G_timeseries_map = read_csv2numpy(elec_path, hyd_path)  # [N,all_T 2]
    time_list = get_time_list(elec_path)
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    A = nx.adjacency_matrix(G)  # 返回图的邻接矩阵
    A = normalize_adj(A).todense()
    A = torch.tensor(A, dtype=torch.float32).to(device)
    if normalize:
        max_ = np.amax(G_timeseries_map, axis=(0, 1))
        G_timeseries_map = G_timeseries_map / max_
        G_timeseries_map[G_timeseries_map == 0] = np.random.normal(
            np.zeros_like(G_timeseries_map[G_timeseries_map == 0]),
            0.001)
    dataset, time_list = process_dalian_gcn_data(G_timeseries_map, time_list, training_length=train_length,
                                                 forecast_window=forecast_window,
                                                 time=True, random=True, matrix=True)

    dataset = dataset[:, :, :, -1]
    # train_data = SensorDataset_GCN(train_dataset, X_train_time, train_length, forecast_window)
    # test_data = SensorDataset_GCN(test_dataset, X_test_time, train_length, forecast_window)
    # train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, Nodes, Times]
    # test_dl = DataLoader(test_data, batch_size=1)
    # args = get_model_args(name='Transformer_gcn_Dalian', train_length=train_length, forcast_window=forecast_window,
    #                       X_train=train_dataset, adj_matrix=A)

    Train = Train_all_models(name=name, dataset=dataset, time_list=time_list,
                             train_length=train_length, forcast_window=forecast_window, A=A, max_=max_[-1], random=True,
                             load=True)
    Train.train(device)


if __name__ == "__main__":
    # data_xml = "TrafficSim/out_7days.xml"
    # net_xml = "TrafficSim/network.net.xml"
    elec_path = "data/数据源/elec_data.csv"
    hyd_path = "data/数据源/hyd_data.csv"
    graph_path = "train_data/GCN_Dalian_Graph.pkl"
    train_length = 30
    forecast_window = 5
    # Train_Dalian_Gcn_Transform_data(Rp_best, elec_path, hyd_path, graph_path, train_length, forecast_window, load=True)
    # Test_Dalian_Gcn_Transform(data_xml, net_xml, train_length, forecast_window*5)
    Train_Dalian_Test(Rp_best, elec_path, hyd_path, graph_path, train_length, forecast_window)
