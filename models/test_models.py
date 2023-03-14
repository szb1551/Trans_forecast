import networkx as nx

from .train_models import Train_all_models
from .plot import plot_all_models
from DataLoader import SensorDataset4, SensorDataset_GCN
from .model_utils import get_Compare_model_args, get_model_args, select_model, get_Compare_long_model_args
import torch
import torch.nn as nn
import numpy as np
from graph_utils import normalize_adj
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Test_all_models(Train_all_models):
    def __init__(self, name, model, dataset, time_list, train_length, forcast_window, max_=None, Rp_best=10, A=None):
        super(Test_all_models, self).__init__(name, dataset, time_list, train_length, forcast_window, max_, Rp_best, A)
        self.model = model
        self.args = self.get_args()


    def test_model(self, device, plot=False):
        if self.args['name'] == 'gcn':
            train_dl, test_dl = self.get_dl(SensorDataset_GCN)
            self.plot_model(self.model, train_dl, self.args, plot=plot)
            self.plot_model(self.model, test_dl, self.args, test=True, plot=plot)
            self.plot_attn(self.model, train_dl, self.args, device=device)
        elif self.args['name'] == 'raster':
            train_dl, test_dl = self.get_dl(SensorDataset_GCN)
            self.plot_model(self.model, train_dl, self.args, plot=plot)
            self.plot_model(self.model, test_dl, self.args, test=True, plot=plot)
            self.plot_attn(self.model, train_dl, self.args, device=device)
        elif self.args['name'] == 'day':
            train_dl, test_dl = self.get_dl(SensorDataset_GCN)
            self.plot_model(self.model, train_dl, self.args, plot=plot)
            self.plot_model(self.model, test_dl, self.args, test=True, plot=plot)
            self.plot_attn(self.model, train_dl, self.args, device=device)
        elif self.args['name'] == 'conv_lstm':
            train_dl, test_dl = self.get_dl(SensorDataset_GCN, baseline=True)
            self.plot_model(self.model, train_dl, self.args, plot=plot)
            self.plot_model(self.model, test_dl, self.args, test=True, plot=plot)
            self.plot_attn(self.model, train_dl, self.args, device=device)
        elif self.args['name'] == 'gcn_lstm':
            train_dl, test_dl = self.get_dl(SensorDataset_GCN, baseline=True)
            self.plot_model(self.model, train_dl, self.args, plot=plot)
            self.plot_model(self.model, test_dl, self.args, test=True, plot=plot)
            self.plot_attn(self.model, train_dl, self.args, device=device)


class Compare_test_models:
    def __init__(self, model_num, n_plots=7, long=False):
        if long:
            self.args = get_Compare_long_model_args()
        else:
            self.args = get_Compare_model_args()
        self.model_num = model_num
        self.model_list = []
        self.model_args_list = []
        self.outputs = []
        self.day_test_dataset = np.load(self.args['day_test_dataset'])
        self.raster_test_dataset = np.load(self.args['raster_test_dataset'])
        self.gcn_test_dataset = np.load(self.args['gcn_test_dataset'])
        self.A = self.initial_A()
        self.day_max_ = np.load(self.args['day_max_'])
        self.raster_max_ = np.load(self.args['raster_max_'])
        self.gcn_max_ = np.load(self.args['gcn_max_'])
        self.test_time_list = np.load(self.args['test_time_list'])
        self.n_plots = n_plots
        self.model_initial()

    def initial_A(self):
        Graph = nx.read_gpickle(self.args['Graph'])
        A = nx.adjacency_matrix(Graph)
        A = torch.Tensor(normalize_adj(A).todense())
        return A

    def model_initial(self):
        for i in range(self.model_num):
            model_name = self.args['model_initial_sequential'][i]
            model_args_name = self.args['model_args_name'][i]

            args = get_model_args(model_args_name, self.args['train_length'], self.args['forcast_window'],
                                  self.raster_test_dataset, self.A.cpu())
            # model = select_model(model_args_name, args)
            # model.load_state_dict(torch.load(self.args[model_name]))
            model = torch.load(self.args[model_name])
            self.model_list.append(model)
            self.model_args_list.append(args)

    def get_model_output(self):
        self.plot_initial()
        outputs_baseline1, self.date_lists = self.plot_test[0].get_output_baseline(max_=self.max_lists[0], time_=True, device=device)
        outputs_baseline2 = self.plot_test[1].get_output_baseline(max_=self.gcn_max_, device=device)
        outputs_day = self.plot_test[2].get_output_Transformer(max_=self.day_max_)
        outputs_target = self.plot_test[2].get_target(max_=self.day_max_)
        outputs_raster = self.plot_test[3].get_output_Transformer(max_=self.raster_max_)
        outputs_gcn = self.plot_test[4].get_output_Transformer_gcn(max_=self.gcn_max_)
        # outputs = [outputs_target[:self.model_num], torch.stack(outputs_baseline1[:self.model_num]).sum(axis=-1).sum(axis=-1),
        #            torch.stack(outputs_baseline2[:self.model_num]).sum(axis=2), outputs_day[:self.model_num],
        #            torch.stack(outputs_raster[:self.model_num]).sum(axis=-1).sum(axis=-1), torch.stack(outputs_gcn[:self.model_num]).sum(axis=2)]
        outputs = [outputs_target,
                   outputs_baseline1,
                   outputs_baseline2,
                   outputs_day,
                   outputs_raster,
                   outputs_gcn]
        return outputs

    # def compute_loss(self):
    #     outputs = self.get_model_output()
    #     criterion = torch.nn.MSELoss()
    #     target = outputs[0]
    #     for i in range(1, self.model_num+1):
    #         RMSE = torch.sqrt(criterion(outputs[i], ))


    def plot_initial(self):
        self.plot_test = []
        day_test_data = SensorDataset4(self.day_test_dataset, self.test_time_list,
                                       train_length=self.args['train_length'],
                                       forecast_window=self.args['forcast_window'])
        day_test_dl = DataLoader(day_test_data, batch_size=64)
        raster_test_data = SensorDataset4(self.raster_test_dataset, self.test_time_list,
                                          train_length=self.args['train_length'],
                                          forecast_window=self.args['forcast_window'])
        raster_test_dl = DataLoader(raster_test_data, batch_size=64)
        gcn_test_data = SensorDataset_GCN(self.gcn_test_dataset, self.test_time_list,
                                          train_length=self.args['train_length'],
                                          forecast_window=self.args['forcast_window'])
        gcn_test_dl = DataLoader(gcn_test_data, batch_size=64)
        dl_lists = [raster_test_dl, gcn_test_dl, day_test_dl, raster_test_dl, gcn_test_dl]
        self.max_lists = [self.raster_max_, self.gcn_max_, self.day_max_, self.raster_max_, self.gcn_max_]
        for i in range(self.model_num):
            model_test = plot_all_models(self.model_list[i], dl_lists[i], self.model_args_list[i], n_plots=self.n_plots)
            self.plot_test.append(model_test)

    def plot_models(self, fontsize=20):
        outputs = self.get_model_output()
        colors = ['k', 'b--', 'c--', 'g--', 'y--', 'r--']
        linewidths = [5,3,3,3,3,3]
        print(outputs[0][0].shape)
        for i in range(self.n_plots):
            plt.figure(figsize=(20, 10))
            plt.title('{} Day Forecast'.format(self.args['forcast_window']), fontsize=30)
            plt.xticks(np.arange(self.args['forcast_window']), self.date_lists[i][self.args['train_length']:])
            plt.xticks(rotation=45, fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            # for j in range(self.model_num+1):
            #     plt.plot(self.date_lists[i][self.args['train_length']:], outputs[j][i].squeeze(), colors=colors[j], linewidth=linewidths[j])
            plt.plot(self.date_lists[i][self.args['train_length']:], outputs[0][i][i].squeeze(), colors[0],
                     linewidth=linewidths[0])
            plt.plot(self.date_lists[i][self.args['train_length']:], outputs[1][i][i].sum(axis=-1).sum(axis=-1).squeeze(), colors[1],
                     linewidth=linewidths[1])
            plt.plot(self.date_lists[i][self.args['train_length']:], outputs[2][i][i].sum(axis=0).squeeze(), colors[2],
                     linewidth=linewidths[2])
            plt.plot(self.date_lists[i][self.args['train_length']:], outputs[3][i][i].squeeze(), colors[3],
                     linewidth=linewidths[3])
            plt.plot(self.date_lists[i][self.args['train_length']:], outputs[4][i][i].sum(axis=-1).sum(axis=-1).squeeze(), colors[4],
                     linewidth=linewidths[4])
            plt.plot(self.date_lists[i][self.args['train_length']:], outputs[5][i][i].sum(axis=0).squeeze(), colors[5],
                     linewidth=linewidths[5])
            plt.grid()
            plt.legend(["$[t_0,t_0+{7})_{true}$", "$[t_0,t_0+{7})_{CNN+LSTM}$", "$[t_0,t_0+{7})_{T-GCN}$",
                        "$[t_0,t_0+{7})_{Trans}$", "$[t_0,t_0+{7})_{CNN+Trans}$", "$[t_0,t_0+{7})_{GCN+Trans}$"], fontsize=fontsize)
            plt.subplots_adjust(wspace=0, hspace=0, top=0.94, bottom=0.15)
            if self.args['save_figure']:
                plt.savefig('results/compare_results/03/model_compare_7Days_{}.jpg'.format(i + 1))
            else:
                plt.show()
