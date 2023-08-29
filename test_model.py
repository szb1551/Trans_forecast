import numpy as np
import torch
import datetime
import networkx as nx
from DataLoader import SensorDataset4, SensorDataset_GCN, SensorDataset_baseline_conv
from graph_utils import normalize_adj
from main import plot_models, process_data, get_date_data_PALO, get_attn, show_attn
from main import plot_raster_models, process_raster_data, get_raster_attn, show_raster_attn
from main import process_gcn_data, plot_gcn_models, get_gcn_attn, show_gcn_attn, get_gcn_date_PALO
from torch.utils.data import DataLoader
from models.test_models import Test_all_models, Compare_test_models
from main import plot_all_models
from models.model_utils import get_model_args, select_model
import os
import matplotlib.pyplot as plt

idx_example = 0
train_length = 30
forcast_window = 7
print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class test_all_model:
#     #TODO, 越写越麻烦 md
#     def __init__(self, select_model):
#         self.model, self.model_name = self.choose_model(select_model)
#         self.train_map, self.time_list, self.max_ = self.choose_init()
#
#     def choose_init(self):
#         if self.model_name=='Transformer':
#             train_map = np.load('train_data/normalized.npy')
#             time_list = np.load('train_data/time_list.npy')
#             max_ = np.load('train_data/max_.npy')
#         elif self.model_name == 'Transformer2':
#             train_map = np.load('train_data/raster_map.npy')
#             time_list = np.load('train_data/time_list.npy')
#             max_ = np.load('train_data/max_raster.npy')
#         elif self.model_name == 'Transformer3':
#             train_map = np.load('train_data/G_timeseries_map.npy')
#             time_list = np.load('train_data/time_list.npy')
#             max_ = np.load('train_data/G_max_.npy')
#         elif self.model_name == 'conv_lstm':
#             train_map = np.load('train_data/normalized.npy')
#             time_list = np.load('train_data/time_list.npy')
#             max_ = np.load('train_data/max_.npy')
#         elif self.model_name == 'Transformer':
#             train_map = np.load('train_data/normalized.npy')
#             time_list = np.load('train_data/time_list.npy')
#             max_ = np.load('train_data/max_.npy')
#         else:
#             raise ValueError('Invalid name')
#
#         return train_map, time_list, max_
#
#     def select_name(self, select_model):
#         if select_model.split('/')[1] == 'Trans':
#             model_name = 'Transformer'
#             X_train=None
#             adj_matrix = None
#         elif select_model.split('/')[1] == 'cnn_Trans':
#             model_name = 'Transformer2'
#             X_train = torch.zeros((3,train_length+forcast_window,5,5))
#             adj_matrix = None
#         elif select_model.split('/')[1] == 'gcn_Trans':
#             model_name = 'Transformer3'
#             X_train = None
#             adj_matrix = torch.zeros((47,47))
#         elif select_model.split('/')[1] == 'cnn_lstm':
#             model_name = 'conv_lstm'
#             X_train = torch.zeros((3,train_length+forcast_window,5,5))
#             adj_matrix = None
#         elif select_model.split('/')[1] == 'gcn_lstm':
#             model_name = 'gcn_lstm'
#             X_train = None
#             adj_matrix = torch.zeros((47,47))
#         else:
#             raise ValueError('请输入正确的路径{}'.format(select_model))
#         return model_name, X_train, adj_matrix
#     def choose_model(self, select_model, train_length=30, forcast_window=7):
#         if select_model.split('.')[-1] =='h5':
#             model = torch.load(select_model)
#             model_name, _, _ = self.select_name(select_model)
#             return model, model_name
#         model_name, X_train, adj_matrix = self.select_name(select_model)
#         args = get_model_args(name=model_name, train_length=train_length, forcast_window=forcast_window,
#                               X_train=X_train, adj_matrix=adj_matrix)
#         model = select_model(name=model_name, args=args)
#         return model, model_name


def test_conv_lstm(model_path='results/baseline_cnn10000次/model/model_last_length30.pkl', select_date=False, plot=True,
                   max=False):
    print('开始测试时间', datetime.datetime.now())
    # model = Transformer(train_length, forcast_window).cuda()
    # print(model)
    args = get_model_args(name='conv_lstm', train_length=30, forcast_window=7, X_train=torch.zeros((3, 37, 5, 5)))
    model = select_model(name='conv_lstm', args=args).to(device)
    model.load_state_dict(torch.load(model_path))
    # model = torch.load('train_process/model_epoch86_length30_Rp_0.066.h5')
    raster_map = np.load('train_data/raster_map.npy')
    time_list = np.load('train_data/time_list.npy')
    max_ = np.load('train_data/max_raster.npy')
    raster_map = raster_map / max_
    raster_map[np.isnan(raster_map)] = 0
    if select_date:
        raster_map, time_list = get_date_data_PALO(raster_map, time_list, year_begin=2011, year_end=2017, month_end=8,
                                                   day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_raster_data(raster_map, time_list, train_length,
                                                                                 forcast_window, time=True)

    train_data = SensorDataset_baseline_conv(train_dataset, X_train_time, train_length=train_length,
                                             forecast_window=forcast_window)
    test_data = SensorDataset_baseline_conv(test_dataset, X_test_time, train_length=train_length,
                                            forecast_window=forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1)
    plot_train = plot_all_models(model, train_dl, args)
    plot_test = plot_all_models(model, test_dl, args)
    if max:
        plot_train.plot_baseline(plot=plot, max_=max_)
        plot_test.plot_baseline(test=True, plot=plot, max_=max_)
    else:
        plot_train.plot_baseline(plot=plot)
        plot_test.plot_baseline(test=True, plot=plot)
    print('结束测试时间', datetime.datetime.now())


def test_gcn_lstm(model_path='results/baseline_gcn10000次/model/model_last_length30.pkl', select_date=False, plot=True,
                  max=False):
    print('开始测试时间', datetime.datetime.now())
    Graph = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    G_timeseries_map = np.load('train_data/G_timeseries_map.npy')
    time_list = np.load('train_data/time_list.npy')
    max_ = np.load('train_data/G_max_.npy')
    G_timeseries_map = G_timeseries_map / max_
    A = nx.adjacency_matrix(Graph)
    A = normalize_adj(A).todense()
    A = torch.tensor(A, dtype=torch.float32).to(device)
    if select_date:
        G_timeseries_map, time_list = get_gcn_date_PALO(G_timeseries_map, time_list, year_begin=2011, year_end=2017,
                                                        month_end=8,
                                                        day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_gcn_data(G_timeseries_map, time_list, train_length,
                                                                              forcast_window, time=True)

    train_data = SensorDataset_GCN(train_dataset, X_train_time, train_length, forcast_window, baseline=True)
    test_data = SensorDataset_GCN(test_dataset, X_test_time, train_length, forcast_window, baseline=True)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, Nodes, Times]
    test_dl = DataLoader(test_data, batch_size=1)
    args = get_model_args(name='gcn_lstm', train_length=30, forcast_window=7, X_train=train_dataset, adj_matrix=A)
    model = select_model(name='gcn_lstm', args=args).to(device)
    model.load_state_dict(torch.load(model_path))
    plot_train = plot_all_models(model, train_dl, args)
    plot_test = plot_all_models(model, test_dl, args)
    if max:
        plot_train.plot_baseline(plot=plot, max_=max_)
        plot_test.plot_baseline(test=True, plot=plot, max_=max_)
    else:
        plot_train.plot_baseline(plot=plot)
        plot_test.plot_baseline(test=True, plot=plot)
    print('结束测试时间', datetime.datetime.now())


def test_day_model(select_model='results/30+7decoder10000次结果3/train_process/model_last_length30.h5',
                   select_date=False, attn=False, plot=True, max=False):  # 测试transformer+日的
    print('开始测试时间', datetime.datetime.now())
    # model = Transformer(train_length, forcast_window).cuda()
    # print(model)
    # test_model = torch.load('server_result/model_epoch6_length30_Rp_0.321.h5')
    model = torch.load(select_model).to(device)
    day_map = np.load('train_data/normalized.npy')
    time_list = np.load('train_data/time_list.npy')
    max_ = np.load('train_data/max_.npy')
    if select_date:
        day_map, time_list = get_date_data_PALO(day_map, time_list, year_begin=2011, year_end=2017, month_end=8,
                                                day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_data(day_map, time_list, train_length,
                                                                          forcast_window, time=True)
    train_data = SensorDataset4(train_dataset, X_train_time, train_length=train_length, forecast_window=forcast_window)
    test_data = SensorDataset4(test_dataset, X_test_time, train_length=train_length, forecast_window=forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1)
    if max:
        plot_models(model, train_dl, train_length, forcast_window, plot=plot, max_=max_)
        plot_models(model, test_dl, train_length, forcast_window, test=True, plot=plot, max_=max_)
    else:
        plot_models(model, train_dl, train_length, forcast_window, plot=plot)
        plot_models(model, test_dl, train_length, forcast_window, test=True, plot=plot)
    if attn:
        attn_layers = get_attn(model, test_data[idx_example][0].unsqueeze(0),
                               test_data[idx_example][1].unsqueeze(0),
                               test_data[idx_example][2].unsqueeze(0))
        show_attn(test_data, attn_layers)
    print('结束测试时间', datetime.datetime.now())


def test_raster_model(select_model='results/30+7cnn+decoder10000/model/model_last_length30.h5', select_date=False,
                      attn=False, plot=True, max=False):  # 测试transformer+cnn的
    print('开始测试时间', datetime.datetime.now())
    # model = Transformer(train_length, forcast_window).cuda()
    # print(model)
    model = torch.load(select_model)
    # model = torch.load('train_process/model_epoch86_length30_Rp_0.066.h5')
    raster_map = np.load('train_data/raster_map.npy')
    time_list = np.load('train_data/time_list.npy')
    max_ = np.load('train_data/max_raster.npy')
    raster_map = raster_map / max_
    raster_map[np.isnan(raster_map)] = 0
    if select_date:
        raster_map, time_list = get_date_data_PALO(raster_map, time_list, year_begin=2011, year_end=2017, month_end=8,
                                                   day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_raster_data(raster_map, time_list, train_length,
                                                                                 forcast_window, time=True)

    train_data = SensorDataset4(train_dataset, X_train_time, train_length=train_length, forecast_window=forcast_window)
    test_data = SensorDataset4(test_dataset, X_test_time, train_length=train_length, forecast_window=forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1)
    if max:
        plot_raster_models(model, train_dl, train_length, forcast_window, plot=plot, max_=max_)
        plot_raster_models(model, test_dl, train_length, forcast_window, test=True, plot=plot, max_=max_)
    else:
        plot_raster_models(model, train_dl, train_length, forcast_window, plot=plot)
        plot_raster_models(model, test_dl, train_length, forcast_window, test=True, plot=plot)
    if attn:
        attn_layers = get_raster_attn(model, test_data[idx_example][0].unsqueeze(0),
                                      test_data[idx_example][1].unsqueeze(0),
                                      test_data[idx_example][2].unsqueeze(0))
        show_raster_attn(test_data, attn_layers)
    print('结束测试时间', datetime.datetime.now())


def test_gcn_model(model_path='results/30+7gcn10000次_add结果4/model/model_last_length30.h5', select_date=False,
                   attn=False, plot=True, max=False):  # 测试transformer+gcn的
    print('开始测试时间', datetime.datetime.now())
    Graph = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    G_timeseries_map = np.load('train_data/G_timeseries_map.npy')
    time_list = np.load('train_data/time_list.npy')
    max_ = np.load('train_data/G_max_.npy')
    G_timeseries_map = G_timeseries_map / max_
    A = nx.adjacency_matrix(Graph)
    A = normalize_adj(A).todense()
    A = torch.tensor(A)
    print(G_timeseries_map.shape)
    if select_date:
        G_timeseries_map, time_list = get_gcn_date_PALO(G_timeseries_map, time_list, year_begin=2011, year_end=2017,
                                                        month_end=8,
                                                        day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_gcn_data(G_timeseries_map, time_list, train_length,
                                                                              forcast_window, time=True)
    train_data = SensorDataset_GCN(train_dataset, X_train_time, train_length, forcast_window)
    test_data = SensorDataset_GCN(test_dataset, X_test_time, train_length, forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, Nodes, Times]
    test_dl = DataLoader(test_data, batch_size=1)
    if path_backname(model_path):
        model = torch.load(model_path)
    else:
        args = get_model_args(name='Transformer_gcn', train_length=train_length, forcast_window=forcast_window,
                              X_train=train_dataset, adj_matrix=A)
        model = select_model(name='Transformer_gcn', args=args)
        model.load_state_dict(torch.load(model_path))
    if max:
        plot_gcn_models(model, train_dl, train_length, forcast_window, plot=plot, max_=max_)
        plot_gcn_models(model, test_dl, train_length, forcast_window, test=True, plot=plot, max_=max_)
    else:
        plot_gcn_models(model, train_dl, train_length, forcast_window, plot=plot)
        plot_gcn_models(model, test_dl, train_length, forcast_window, test=True, plot=plot)
    if attn:
        attn_layers = get_gcn_attn(model, test_data[idx_example][0].unsqueeze(0),
                                   test_data[idx_example][1].unsqueeze(0),
                                   test_data[idx_example][2].unsqueeze(0))
        show_gcn_attn(test_data, attn_layers)
    print('结束测试时间', datetime.datetime.now())


def get_test_data():  # 得到各个模型的真实测试数据，而不是加入误差的
    if not os.path.exists('test_data'):
        os.mkdir('test_data')
    G_timeseries_map = np.load('train_data/G_timeseries_map.npy')
    G_max_ = np.load('train_data/G_max_.npy')
    G_timeseries_map = G_timeseries_map / G_max_
    raster_map = np.load('train_data/raster_map.npy')
    max_raster = np.load('train_data/max_raster.npy')
    raster_map = raster_map / max_raster
    raster_map[np.isnan(raster_map)] = 0
    day_map = np.load('train_data/normalized.npy')
    time_list = np.load('train_data/time_list.npy')

    _, day_test_dataset, _, day_X_test_time = process_data(day_map, time_list, train_length,
                                                           forcast_window, time=True)
    _, raster_test_dataset, _, raster_X_test_time = process_raster_data(raster_map, time_list, train_length,
                                                                        forcast_window, time=True)
    _, gcn_test_dataset, _, gcn_X_test_time = process_gcn_data(G_timeseries_map, time_list, train_length,
                                                               forcast_window, time=True)

    np.save('test_data/day_test_dataset', day_test_dataset)
    np.save('test_data/day_X_test_time', day_X_test_time)
    np.save('test_data/raster_test_dataset', raster_test_dataset)
    np.save('test_data/raster_X_test_time', raster_X_test_time)
    np.save('test_data/gcn_test_dataset', gcn_test_dataset)
    np.save('test_data/gcn_X_test_time', gcn_X_test_time)


def compare_model(select_day_model='results/30+7decoder10000次结果3/train_process/model_last_length30.h5',
                  select_raster_model='results/30+7cnn+decoder10000/model/model_last_length30.h5',
                  select_gcn_model='results/30+7gcn10000次结果/model/model_last_length30.h5',
                  train_length=30, forcast_window=7, plot=True, max=False):
    day_model = torch.load(select_day_model)
    raster_model = torch.load(select_raster_model)
    gcn_model = torch.load(select_gcn_model)

    day_test_dataset = np.load('test_data/day_test_dataset.npy')
    day_max_ = np.load('train_data/max_.npy')
    raster_test_dataset = np.load('test_data/raster_test_dataset.npy')
    raster_max_ = np.load('train_data/max_raster.npy')
    gcn_test_dataset = np.load('test_data/gcn_test_dataset.npy')
    gcn_max_ = np.load('train_data/G_max_.npy')
    test_time_list = np.load('test_data/day_X_test_time.npy')

    day_test_data = SensorDataset4(day_test_dataset, test_time_list, train_length=train_length,
                                   forecast_window=forcast_window)
    day_test_dl = DataLoader(day_test_data, batch_size=1)
    raster_test_data = SensorDataset4(raster_test_dataset, test_time_list, train_length=train_length,
                                      forecast_window=forcast_window)
    raster_test_dl = DataLoader(raster_test_data, batch_size=1)
    gcn_test_data = SensorDataset_GCN(gcn_test_dataset, test_time_list, train_length=train_length,
                                      forecast_window=forcast_window)
    gcn_test_dl = DataLoader(gcn_test_data, batch_size=1)

    if max:
        plot_models(day_model, day_test_dl, train_length, forcast_window, test=True, plot=plot, max_=day_max_)
        plot_raster_models(raster_model, raster_test_dl, train_length, forcast_window, test=True, plot=plot,
                           max_=raster_max_)
        plot_gcn_models(gcn_model, gcn_test_dl, train_length, forcast_window, test=True, plot=plot, max_=gcn_max_)
    else:
        plot_models(day_model, day_test_dl, train_length, forcast_window, test=True, plot=plot)
        plot_raster_models(raster_model, raster_test_dl, train_length, forcast_window, test=True, plot=plot)
        plot_gcn_models(gcn_model, gcn_test_dl, train_length, forcast_window, test=True, plot=plot)


def path_backname(path):
    if path.split('.')[-1] == 'h5':
        return True
    elif path.split('.')[-1] == 'pkl':
        return False
    else:
        raise ValueError('错误的路径后缀名称{}'.format(path))


def compare_all_models(conv_lstm_path='results/baseline_cnn10000次/model/model_last_length30.pkl',
                       gcn_lstm_path='results/baseline_gcn10000次/model/model_last_length30.pkl',
                       day_path='results/30+7decoder10000次结果2/model_last_length30.h5',
                       raster_day_path='results/30+7cnn+decoder10000次/model/model_last_length30.h5',
                       gcn_day_path='results/split_results/gcn_Trans/30+7_10000次/model/model_last_length30.h5',
                       train_length=30, forcast_window=7, save_figure=False):
    day_test_dataset = np.load('test_data/day_test_dataset.npy')
    day_max_ = np.load('train_data/max_.npy')
    raster_test_dataset = np.load('test_data/raster_test_dataset.npy')
    raster_max_ = np.load('train_data/max_raster.npy')
    Graph = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    A = nx.adjacency_matrix(Graph)
    A = torch.Tensor(normalize_adj(A).todense())
    gcn_test_dataset = np.load('test_data/gcn_test_dataset.npy')
    gcn_max_ = np.load('train_data/G_max_.npy')
    test_time_list = np.load('test_data/day_X_test_time.npy')

    baseline1_args = get_model_args(name='conv_lstm', train_length=train_length,
                                    forcast_window=forcast_window, X_train=raster_test_dataset)
    baseline2_args = get_model_args(name='gcn_lstm', train_length=train_length, forcast_window=forcast_window,
                                    X_train=gcn_test_dataset, adj_matrix=A)
    day_args = get_model_args(name='Transformer', train_length=train_length, forcast_window=forcast_window,
                              X_train=day_test_dataset)
    raster_args = get_model_args(name='Transformer_cnn', train_length=train_length, forcast_window=forcast_window,
                                 X_train=raster_test_dataset)
    gcn_args = get_model_args(name='Transformer_gcn', train_length=train_length, forcast_window=forcast_window,
                              X_train=day_test_dataset, adj_matrix=A)

    if path_backname(conv_lstm_path):
        baseline1_model = torch.load(conv_lstm_path)
    else:
        baseline1_model = select_model(name='conv_lstm', args=baseline1_args)
        baseline1_model.load_state_dict(torch.load(conv_lstm_path))
    if path_backname(gcn_lstm_path):
        baseline2_model = torch.load(gcn_lstm_path)
    else:
        baseline2_model = select_model(name='gcn_lstm', args=baseline2_args)
        baseline2_model.load_state_dict(torch.load(gcn_lstm_path))
    if path_backname(day_path):
        day_model = torch.load(day_path)
    else:
        day_model = select_model(name='Transformer', args=day_args)
        day_model.load_state_dict(torch.load(day_path))
    if path_backname(raster_day_path):
        raster_model = torch.load(raster_day_path)
    else:
        raster_model = select_model(name='Transformer_cnn', args=raster_args)
        raster_model.load_state_dict(torch.load(raster_day_path))
    if path_backname(gcn_day_path):
        gcn_model = torch.load(gcn_day_path)
    else:
        gcn_model = select_model(name='Transformer_gcn', args=gcn_args)
        gcn_model.load_state_dict(torch.load(gcn_day_path))
    day_test_data = SensorDataset4(day_test_dataset, test_time_list, train_length=train_length,
                                   forecast_window=forcast_window)
    day_test_dl = DataLoader(day_test_data, batch_size=1)
    raster_test_data = SensorDataset4(raster_test_dataset, test_time_list, train_length=train_length,
                                      forecast_window=forcast_window)
    raster_test_dl = DataLoader(raster_test_data, batch_size=1)
    gcn_test_data = SensorDataset_GCN(gcn_test_dataset, test_time_list, train_length=train_length,
                                      forecast_window=forcast_window)
    gcn_test_dl = DataLoader(gcn_test_data, batch_size=1)

    baseline1_test = plot_all_models(baseline1_model, raster_test_dl, baseline1_args, n_plots=7)
    baseline2_test = plot_all_models(baseline2_model, gcn_test_dl, baseline2_args, n_plots=7)
    day_test = plot_all_models(day_model, day_test_dl, day_args, n_plots=7)
    raster_test = plot_all_models(raster_model, raster_test_dl, raster_args, n_plots=7)
    gcn_test = plot_all_models(gcn_model, gcn_test_dl, gcn_args, n_plots=7)

    outputs_baseline1, date_lists = baseline1_test.get_output_baseline(max_=raster_max_, time_=True)
    outputs_baseline2 = baseline2_test.get_output_baseline(max_=gcn_max_)
    outputs_day = day_test.get_output_Transformer(max_=day_max_)
    outputs_target = day_test.get_target(max_=day_max_)
    outputs_raster = raster_test.get_output_Transformer(max_=raster_max_)
    outputs_gcn = gcn_test.get_output_Transformer_gcn(max_=gcn_max_)

    for i in range(baseline1_test.n_plots):
        plt.figure(figsize=(20, 10))
        plt.title('7 Day Forecast')
        plt.xticks(np.arange(forcast_window), date_lists[i][train_length:])
        plt.xticks(rotation=90)
        plt.plot(date_lists[i][train_length:], outputs_target[i].squeeze(), 'k', linewidth=5)
        plt.plot(date_lists[i][train_length:], outputs_baseline1[i].sum(axis=-1).sum(axis=-1).squeeze(), 'b--',
                 linewidth=3)
        plt.plot(date_lists[i][train_length:], outputs_baseline2[i].sum(axis=1).squeeze(), 'c--', linewidth=3)
        plt.plot(date_lists[i][train_length:], outputs_day[i].squeeze(), 'g--', linewidth=3)
        plt.plot(date_lists[i][train_length:], outputs_raster[i].sum(axis=-1).sum(axis=-1).squeeze(), 'y--',
                 linewidth=3)
        plt.plot(date_lists[i][train_length:], outputs_gcn[i].sum(axis=1).squeeze(), 'r--', linewidth=3)
        plt.grid()
        plt.legend(["$[t_0,t_0+{7})_{true}$", "$[t_0,t_0+{7})_{CNN+LSTM}$", "$[t_0,t_0+{7})_{T-GCN}$",
                    "$[t_0,t_0+{7})_{Trans}$", "$[t_0,t_0+{7})_{CNN+Trans}$", "$[t_0,t_0+{7})_{GCN+Trans}$"])
        if save_figure:
            plt.savefig('results/compare_results/01/model_compare_7Days_{}.jpg'.format(i + 1))
        else:
            plt.show()


def Test_split(train_length=30, forcast_window=7, long=False):
    if long:
        dataset = np.load('train_data_matrix/raster_long_matrix_lags.npy')
        dataset1 = np.load('train_data_matrix/G_long_matrix_lags.npy')
        dataset2 = np.load('train_data_matrix/day_long_matrix_lags.npy')
        time_list = np.load('train_data_matrix/time_long_matrix_lags.npy')
    else:
        dataset = np.load('train_data_matrix/raster_matrix_lags.npy')
        dataset1 = np.load('train_data_matrix/G_matrix_lags.npy')
        dataset2 = np.load('train_data_matrix/day_matrix_lags.npy')
        time_list = np.load('train_data_matrix/time_matrix_lags.npy')
    Graph = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    A = nx.adjacency_matrix(Graph)
    A = normalize_adj(A).todense()
    A = torch.tensor(A, dtype=torch.float32).to(device)
    max_ = np.load('train_data/max_raster.npy')
    max_1 = np.load('train_data/G_max_.npy')
    max_2 = np.load('train_data/max_.npy')
    model = torch.load('results/split_results/cnn_lstm/{}+{}_10000次/model/model_last_length{}.h5'.format(train_length, forcast_window, train_length))
    Test = Test_all_models(name='conv_lstm', model=model, dataset=dataset, time_list=time_list,
                           train_length=train_length, forcast_window=forcast_window, A=A, max_=max_)
    Test.test_model(device, plot=False)


if __name__ == '__main__':
    # test_gcn_model(select_date=False, attn=True, plot=False, max=False)
    # test_raster_model(select_date=False, attn=True, plot=False, max=True)
    # test_day_model(select_date=False, attn=True, plot=False, max=True)
    # test_conv_lstm(select_date=False, plot=True, max=True)
    # test_gcn_lstm(select_date=True, plot=True, max=True)
    # compare_all_models()
    # a = Compare_test_models(5, long=True)
    # a.plot_models()
    # a.plot_station_models(station_num=9)
    # a.plot_grid_models(1, 1)
    Test_split(120, 30, True)
