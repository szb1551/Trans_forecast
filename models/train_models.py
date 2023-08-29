from torch.utils.data import random_split, DataLoader
from DataLoader import SensorDataset4, SensorDataset_GCN, SensorDataset_baseline_conv, SensorDataset_GCN_diff
from models.model_utils import get_model_args, select_model
import torch
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from models.plot import plot_all_models
from collections import deque


def Dp(y_pred, y_true, q):  # softmax更新
    return max([q * (y_pred - y_true), (1 - q) * (y_true - y_pred)])


def Rp_num_den(y_preds, y_trues, q):  # RP_loss [7,]
    numerator = np.sum([Dp(y_pred, y_true, q) for y_pred, y_true in zip(y_preds, y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator, denominator


def diff_squre(y_pred, y_true):
    return (y_pred - y_true) ** 2


def rmse_diff(predictions, targets):
    differences_squared = np.sum(diff_squre(y_pred, y_true) for y_pred, y_true in zip(predictions, targets))
    num = len(predictions)
    return differences_squared, num


def mape_ins(y_pred, y_true):
    return abs((y_pred - y_true) / y_true)


def mape(predictions, targets):
    mape_sum = np.sum(mape_ins(y_pred, y_true) for y_pred, y_true in zip(predictions, targets))
    num = len(predictions)
    return mape_sum, num


class Train_all_models:
    def __init__(self, name, dataset, time_list, train_length, forcast_window, max_=None, random=False, Rp_best=10, A=None):
        self.name = name
        self.dataset = dataset
        self.Rp_best = Rp_best
        self.time_list = time_list
        self.model_num = 2
        self.filenames = deque(maxlen=self.model_num)
        self.train_epoch_loss = []
        self.train_length = train_length
        self.forcast_window = forcast_window
        self.criterion = torch.nn.MSELoss()
        self.max_ = max_
        self.A = A
        self.random = random

    def save_dataset(self, train_dataset, test_dataset, save):
        if save:
            np.save('split_data/{}_long_train_dataset.npy'.format(self.name), train_dataset, allow_pickle=True)
            np.save('split_data/{}_long_test_dataset.npy'.format(self.name), test_dataset, allow_pickle=True)

    def save_timeset(self, X_train_time, X_test_time, save):
        if save:
            np.save('split_data/{}_long_train_timeset.npy'.format(self.name), X_train_time)
            np.save('split_data/{}_long_test_timeset.npy'.format(self.name), X_test_time)

    def get_dataset(self, train_proportion=0.8, random=False, save=False):
        num_train = int(self.dataset.shape[0] * train_proportion)
        num_test = self.dataset.shape[0] - num_train
        if random:
            torch.manual_seed(0)  # 设置随机种子
            train, test = random_split(self.dataset, [num_train, num_test])
            print(train.indices)
            print(test.indices)
            train_dataset = self.dataset[train.indices]
            test_dataset = self.dataset[test.indices]
            X_train_time = self.time_list[train.indices]
            X_test_time = self.time_list[test.indices]
        else:
            train_dataset = self.dataset[:num_train]
            test_dataset = self.dataset[num_train:]
            X_train_time = self.time_list[:num_train]
            X_test_time = self.time_list[num_train:]
        self.save_dataset(train_dataset, test_dataset, save)
        self.save_timeset(X_train_time, X_test_time, save)
        return train_dataset, test_dataset, X_train_time, X_test_time

    def get_dl(self, Sensor, baseline=False, random=False):
        train_dataset, test_dataset, X_train_time, X_test_time = self.get_dataset(random=random)
        train_data = Sensor(train_dataset, X_train_time, self.train_length, self.forcast_window, baseline)
        test_data = Sensor(test_dataset, X_test_time, self.train_length, self.forcast_window, baseline)
        train_dl = DataLoader(train_data, batch_size=64, shuffle=True)  # [batch_size, Nodes, Times]
        test_dl = DataLoader(test_data, batch_size=64)
        return train_dl, test_dl

    def initial_files(self, dir_name='train_process/'):
        files = os.listdir(dir_name)
        if len(files):  # 初始化文件夹里面的文件命
            for file in files:
                if file.split('.')[-1] == 'h5' and file.split('_')[1] != 'last':
                    self.filenames.append(dir_name + file.replace('.h5', ''))

    def train_gcn_epoch(self, model, train_dl, optimizer, device, baseline=False):
        model.train()
        train_loss = 0
        n = 0
        for step, (x, src, tar, times) in enumerate(train_dl):
            # [batch, time]
            tar_in = tar[:, :, :-1]
            tar_out = tar[:, :, 1:]
            optimizer.zero_grad()
            output = model(x.to(device).float(), src.to(device).float(), tar_in.to(device))
            # output [batch, forcast_window, 1]
            loss = self.criterion(output.squeeze(-1), tar_out.to(device))  # not missing data
            loss.backward()
            optimizer.step()
            train_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]
        return train_loss / n

    def train_gcn_diff_epoch(self, model, train_dl, optimizer, device, *args):
        model.train()
        train_loss = 0
        n = 0
        for step, (x, src, tar, times, _, _) in enumerate(train_dl):
            # [batch, time]
            tar_in = tar[:, :, :-1]
            tar_out = tar[:, :, 1:]
            optimizer.zero_grad()
            output = model(x.to(device).float(), src.to(device).float(), tar_in.to(device))
            # output [batch, forcast_window, 1]
            loss = self.criterion(output.squeeze(-1), tar_out.to(device))  # not missing data
            loss.backward()
            optimizer.step()
            train_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]
        return train_loss / n

    def train_epoch(self, model, train_dl, optimizer, device, baseline):
        model.train()
        train_loss = 0
        n = 0
        for step, (x, src, tar, times) in enumerate(train_dl):
            # [batch, time]
            tar_in = tar[:, :-1]
            tar_out = tar[:, 1:]
            optimizer.zero_grad()
            if baseline:
                output = model(src.to(device))

                loss = self.criterion(output, tar.to(device))
            else:
                output = model(x.to(device).float(), src.to(device).float(), tar_in.to(device))
                # output [batch, forcast_window, 1]
                loss = self.criterion(output.squeeze(-1), tar_out.to(device))  # not missing data
            loss.backward()
            optimizer.step()
            train_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]
        return train_loss / n

    def choose_train_func(self, args):
        if args['name'] == 'gcn' or args['name'] == 'day_station':
            return self.train_gcn_epoch
        elif args['name'] == 'gcn_diff':
            return self.train_gcn_diff_epoch
        elif args['name'] == 'raster' or args['name'] == 'day':
            return self.train_epoch
        elif args['name'] == 'conv_lstm' or args['name'] == 'gcn_lstm' or args['name']=='conv_lstm2':
            return self.train_epoch
        else:
            raise ValueError('错误的模型名称{}'.format(args['name']))

    def choose_test_func(self, args):
        if args['name'] == 'gcn' or args['name'] == 'day_station':
            return self.test_GCN_Transform
        elif args['name'] == 'gcn_diff':
            return self.test_GCN_diff_Transform
        elif args['name'] == 'raster':
            return self.test_cnn_Transform
        elif args['name'] == 'day':
            return self.test_epoch
        elif args['name'] == 'conv_lstm' or args['name']=='conv_lstm2':
            return self.test_baseline_epoch
        elif args['name'] == 'gcn_lstm':
            return self.test_gcn_baseline_epoch
        else:
            raise ValueError('错误的模型名称{}'.format(args['name']))

    def train_step(self, model, args, train_dl, test_dl, device, baseline=False):
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        train_func = self.choose_train_func(args)
        test_func = self.choose_test_func(args)
        for e, epoch in enumerate(range(args['epochs'])):
            start = time.time()
            train_loss = []

            l_t = train_func(model, train_dl, optimizer, device, baseline)
            train_loss.append(l_t)

            Rp, RMSE, MAPE = test_func(model, test_dl, device)

            if self.Rp_best > Rp:
                self.Rp_best = Rp
                self.save_model(model, e + 1, self.train_length, Rp, model_name=args['name'])
            self.train_epoch_loss.append([np.mean(train_loss), Rp, RMSE, MAPE])
            end = time.time()
            print("Epoch {}: Train loss: {:.6f} \t R_p={:.3f}\t RMSE={:.3f}\t MAPE={:.3f}\t cost_time={:.4f}s"
                  .format(e + 1, np.mean(train_loss), Rp, RMSE, MAPE, end - start))

    def test_GCN_Transform(self, model, test_dl, device):
        with torch.no_grad():
            predictions = []
            observations = []
            model.eval()
            for step, (x, src, tar, times) in enumerate(test_dl):
                tar_in = tar[:, :, 0].unsqueeze(-1)
                tar_out = tar[:, :, 1:]
                next_input = tar_in
                all_predictions = []

                for i in range(self.forcast_window):
                    output = model(x.to(device), src.to(device), next_input.to(device))  # [batch, Node, time]
                    if all_predictions == []:
                        all_predictions = output[:, :, -1].unsqueeze(-1).detach()
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                    next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                if self.max_ is not None:
                    for p, o in zip((all_predictions.cpu() * self.max_).numpy().sum(axis=1),
                                    (tar_out.numpy() * self.max_).sum(axis=1)):  # not missing data
                        predictions.append(p)
                        observations.append(o)
                else:
                    for p, o in zip(all_predictions.cpu().numpy().sum(axis=1),
                                    tar_out.numpy().sum(axis=1)):  # not missing data
                        predictions.append(p)
                        observations.append(o)

            num = 0
            den = 0
            num_n = 0
            diff = 0
            mape_add = 0
            for y_preds, y_trues in zip(predictions, observations):
                num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
                num += num_i
                den += den_i
                diff_i, n_i = rmse_diff(y_preds, y_trues)
                diff += diff_i
                mape_i, _ = mape(y_preds, y_trues)
                mape_add += mape_i
                num_n += n_i
            Rp = (2 * num) / den
            RMSE = np.sqrt(diff / num_n)
            MAPE = mape_add / num_n
        return Rp, RMSE, MAPE

    def get_predictions_log_observations(self, all_pre, tar_out):
        predictions = []
        observations = []
        if self.max_ is not None:
            all_pre = all_pre * self.max_
            all_pre = np.exp(all_pre) - np.ones(all_pre.shape)
            tar_out = tar_out * self.max_
            tar_out = np.exp(tar_out) - np.ones(tar_out.shape)
            for p, o in zip(all_pre.sum(axis=1),
                            tar_out.sum(axis=1)):  # not missing data
                predictions.append(p)
                observations.append(o)
        else:
            raise ValueError('此方法必须恢复数据,缺少max_')

        return predictions, observations

    def get_evaluation(self, predictions, observations):
        num = 0
        den = 0
        num_n = 0
        diff = 0
        mape_add = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
            diff_i, n_i = rmse_diff(y_preds, y_trues)
            diff += diff_i
            mape_i, _ = mape(y_preds, y_trues)
            mape_add += mape_i
            num_n += n_i
        Rp = (2 * num) / den
        RMSE = np.sqrt(diff / num_n)
        MAPE = mape_add / num_n

        return Rp, RMSE, MAPE

    def recover_data(self, diff_data, diff_begin, ax=3):
        if ax == 3:
            pad_diff_data = np.pad(diff_data, ((0, 0), (0, 0), (1, 0)), mode='constant')
            pad_diff_data[:, :, 0] = diff_begin[:, :, 0]
            recovered_data = np.cumsum(pad_diff_data, axis=-1)
        else:
            pad_diff_data = np.pad(diff_data, ((0, 0), (1, 0)), mode='constant')
            pad_diff_data[:, 0] = diff_begin[:, 0]
            recovered_data = np.cumsum(pad_diff_data, axis=-1)
        return recovered_data

    def test_GCN_diff_Transform(self, model, test_dl, device):
        with torch.no_grad():
            model.eval()
            for step, (x, src, tar, times, _, tar_begin) in enumerate(test_dl):
                tar_in = tar[:, :, 0].unsqueeze(-1)
                tar_out = tar[:, :, 1:]
                next_input = tar_in
                all_predictions = []

                for i in range(self.forcast_window):
                    output = model(x.to(device), src.to(device), next_input.to(device))  # [batch, Node, time]
                    if all_predictions == []:
                        all_predictions = output[:, :, -1].unsqueeze(-1).detach()
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                    next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                all_predictions = all_predictions.cpu().numpy()
                recovered_data = self.recover_data(all_predictions, tar_begin)
                recovered_tar_data = self.recover_data(tar_out, tar_begin)
                predictions, observations = self.get_predictions_log_observations(recovered_data, recovered_tar_data)
                Rp, RMSE, MAPE = self.get_evaluation(predictions, observations)
        return Rp, RMSE, MAPE

    def test_cnn_Transform(self, model, test_dl, device):
        with torch.no_grad():
            predictions = []
            observations = []

            model.eval()
            for step, (x, src, tar, times) in enumerate(test_dl):
                tar_in = tar[:, 0].unsqueeze(1)
                tar_out = tar[:, 1:]
                next_input = torch.tensor(tar_in)
                all_predictions = []

                for i in range(self.forcast_window):
                    output = model(x.to(device), src.to(device), next_input.to(device))  # [batch, time, 5, 5]
                    if all_predictions == []:
                        all_predictions = output[:, -1].unsqueeze(1).detach()
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, -1].unsqueeze(1).detach()), dim=1)
                    next_input = torch.cat((next_input.to(device), output[:, -1].unsqueeze(1).detach()), dim=1)
                if self.max_ is not None:
                    for p, o in zip((all_predictions.cpu().numpy() * self.max_).sum(axis=-1).sum(axis=-1),
                                    (tar_out.numpy() * self.max_).sum(axis=-1).sum(axis=-1)):  # not missing data
                        predictions.append(p)
                        observations.append(o)
                else:
                    for p, o in zip(all_predictions.cpu().numpy().sum(axis=-1).sum(axis=-1),
                                    tar_out.numpy().sum(axis=-1).sum(axis=-1)):  # not missing data
                        predictions.append(p)
                        observations.append(o)

            num = 0
            den = 0
            num_n = 0
            diff = 0
            mape_add = 0
            for y_preds, y_trues in zip(predictions, observations):
                num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
                num += num_i
                den += den_i
                diff_i, n_i = rmse_diff(y_preds, y_trues)
                diff += diff_i
                mape_i, _ = mape(y_preds, y_trues)
                mape_add += mape_i
                num_n += n_i
            Rp = (2 * num) / den
            RMSE = np.sqrt(diff / num_n)
            MAPE = mape_add / num_n
        return Rp, RMSE, MAPE

    def test_epoch(self, model, test_dl, device):
        with torch.no_grad():
            predictions = []
            observations = []

            model.eval()
            for step, (x, src, tar, times) in enumerate(test_dl):
                tar_in = src[:, -1].unsqueeze(-1)
                tar_out = tar[:, 1:]
                next_input = torch.tensor(tar_in)
                all_predictions = []

                for i in range(self.forcast_window):
                    output = model(x.to(device), src.to(device), next_input.to(device))  # [batch, time]
                    if all_predictions == []:
                        all_predictions = output[:, -1].detach()
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, -1].detach()), dim=-1)
                    next_input = torch.cat((next_input.to(device), output[:, -1].detach()), dim=-1)
                if self.max_ is not None:
                    for p, o in zip(all_predictions.cpu().numpy() * self.max_,
                                    (tar_out.numpy() * self.max_).tolist()):  # not missing data
                        predictions.append(p)
                        observations.append(o)
                else:
                    for p, o in zip(all_predictions.cpu().numpy(), tar_out.numpy().tolist()):  # not missing data
                        predictions.append(p)
                        observations.append(o)

            num = 0
            den = 0
            num_n = 0
            diff = 0
            mape_add = 0
            for y_preds, y_trues in zip(predictions, observations):
                num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
                num += num_i
                den += den_i
                diff_i, n_i = rmse_diff(y_preds, y_trues)
                diff += diff_i
                mape_i, _ = mape(y_preds, y_trues)
                mape_add += mape_i
                num_n += n_i
            Rp = (2 * num) / den
            RMSE = np.sqrt(diff / num_n)
            MAPE = mape_add / num_n
        return Rp, RMSE, MAPE

    def test_baseline_epoch(self, model, test_dl, device):
        predictions = []
        observations = []
        with torch.no_grad():
            model.eval()
            for step, (x, src, tar, times) in enumerate(test_dl):
                output = model(src.to(device))  # [batch, time, 5, 5]
                if self.max_ is not None:
                    for p, o in zip((output.cpu() * self.max_).numpy().sum(axis=-1).sum(axis=-1),
                                    (tar.cpu() * self.max_).sum(axis=-1).sum(axis=-1).numpy()):
                        predictions.append(p)
                        observations.append(o)
                else:
                    for p, o in zip(output.sum(axis=-1).sum(axis=-1).cpu().numpy(),
                                    tar.sum(axis=-1).sum(axis=-1).numpy()):
                        predictions.append(p)
                        observations.append(o)

            num = 0
            den = 0
            num_n = 0
            diff = 0
            mape_add = 0
            for y_preds, y_trues in zip(predictions, observations):
                num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
                num += num_i
                den += den_i
                diff_i, n_i = rmse_diff(y_preds, y_trues)
                diff += diff_i
                mape_i, _ = mape(y_preds, y_trues)
                mape_add += mape_i
                num_n += n_i
            Rp = (2 * num) / den
            RMSE = np.sqrt(diff / num_n)
            MAPE = mape_add / num_n
        return Rp, RMSE, MAPE

    def test_gcn_baseline_epoch(self, model, test_dl, device):
        predictions = []
        observations = []
        with torch.no_grad():
            model.eval()
            for step, (x, src, tar, times) in enumerate(test_dl):
                output = model(src.to(device))  # [batch, nodes, time]
                if self.max_ is not None:
                    for p, o in zip((output.cpu() * self.max_).sum(axis=1).numpy(),
                                    (tar * self.max_).sum(axis=1).numpy()):
                        predictions.append(p)
                        observations.append(o)
                else:
                    for p, o in zip(output.sum(axis=1).cpu().numpy(), tar.sum(axis=1).numpy()):
                        predictions.append(p)
                        observations.append(o)

            num = 0
            den = 0
            num_n = 0
            diff = 0
            mape_add = 0
            for y_preds, y_trues in zip(predictions, observations):
                num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
                num += num_i
                den += den_i
                diff_i, n_i = rmse_diff(y_preds, y_trues)
                diff += diff_i
                mape_i, _ = mape(y_preds, y_trues)
                mape_add += mape_i
                num_n += n_i
            Rp = (2 * num) / den
            RMSE = np.sqrt(diff / num_n)
            MAPE = mape_add / num_n
        return Rp, RMSE, MAPE

    def save_model(self, model, e, train_length, Rp, model_name="GCN"):
        save_name = 'train_process/model{}_epoch{}_length{}_Rp_{:.4f}'.format(model_name, e, train_length, Rp)
        torch.save(model, save_name + '.h5')
        torch.save(model.state_dict(), save_name + '.pkl')
        if len(self.filenames) >= self.model_num:
            os.remove(self.filenames[0] + '.h5')
            os.remove(self.filenames[0] + '.pkl')
        self.filenames.append(save_name)

    def save_loss(self, args):
        self.train_epoch_loss = np.array(self.train_epoch_loss)
        plt.figure(figsize=(10, 10))
        plt.plot(self.train_epoch_loss[:, :1])
        plt.legend(['Train Loss'], fontsize=25)
        plt.xlabel("Epoch", fontsize=25)
        plt.ylabel("MSE Loss", fontsize=25)
        plt.grid()
        plt.title('{}_loss_length{}'.format(args['name'], args['train_length']))
        plt.savefig('train_result/{}_{}_loss_length{}.jpg'.format(args['name'], args['lr'], args['train_length']))
        np.savetxt('train_result/{}_{}_loss_length{}.txt'.format(args['name'], args['lr'], args['train_length']),
                   self.train_epoch_loss, fmt='%6f', header='loss and Rp and RMSE and MAPE', delimiter=', ')

    def plot_model(self, model, dl, args, test=False, plot=False):
        plot_pic = plot_all_models(model, dl, args)
        if args['name'] == 'gcn' or args['name'] == 'day_station':
            plot_pic.plot_gcn(max_=self.max_, test=test, plot=plot)
        elif args['name'] == 'gcn_diff':
            plot_pic.plot_gcn_diff(max_=self.max_, test=test, plot=plot)
        elif args['name'] == 'raster' or args['name'] == 'day':
            plot_pic.plot_Trans(max_=self.max_, test=test, plot=plot)
        elif args['name'] == 'conv_lstm' or args['name'] == 'gcn_lstm' or args['name']=='conv_lstm2':
            plot_pic.plot_baseline(max_=self.max_, test=test, plot=plot)
        else:
            raise ValueError('错误的模型名称{}'.format(args['name']))

    def plot_attn(self, model, dl, args, device, idx_example=0):
        if args['name'] == 'gcn' or args['name'] == 'gcn_diff':
            attn_layers = self.get_gcn_attn(model, args, dl.dataset[idx_example][0].unsqueeze(0),
                                            dl.dataset[idx_example][1].unsqueeze(0),
                                            dl.dataset[idx_example][2].unsqueeze(0), device)

        elif args['name'] == 'raster':
            attn_layers = self.get_raster_attn(model, args, dl.dataset[idx_example][0].unsqueeze(0),
                                               dl.dataset[idx_example][1].unsqueeze(0),
                                               dl.dataset[idx_example][2].unsqueeze(0), device)
        elif args['name'] == 'day':
            attn_layers = self.get_day_attn(model, args, dl.dataset[idx_example][0].unsqueeze(0),
                                            dl.dataset[idx_example][1].unsqueeze(0),
                                            dl.dataset[idx_example][2].unsqueeze(0), device)
        elif args['name'] == 'day_station':
            attn_layers = self.get_day_station_attn(model, args, dl.dataset[idx_example][0].unsqueeze(0),
                                                    dl.dataset[idx_example][1].unsqueeze(0),
                                                    dl.dataset[idx_example][2].unsqueeze(0), device)
        else:
            raise ValueError('错误的模型名称{}'.format(args['name']))
        self.show_attn(dl, args, attn_layers, idx_example)

    def get_raster_attn(self, model, args, x, src, tar, device):
        model.eval()
        with torch.no_grad():
            x_in = x[:, :args['train_length']]
            x_out = x[:, args['train_length'] - 1:args['train_length'] - 1 + args['forcast_window']]
            x_in = x_in.to(device)
            src = src.to(device)
            x_out = x_out.to(device)
            tar_in = tar[:, :-1].to(device)
            z_enc = model.input_embedding(src.unsqueeze(2))
            z_enc_embedding = model.flatten(z_enc).permute(1, 0, 2)  # (sequence len,Batch size,embedding_size)
            enc_positional_embeddings = model.positional_embedding(x_in.type(torch.long)).permute(1, 0, 2)
            input_embedding = z_enc_embedding + enc_positional_embeddings
            enc_out = model.transformer_encoder(input_embedding)
            z_dec = model.input_embedding(tar_in.unsqueeze(2))
            z_dec_embedding = model.flatten(z_dec).permute(1, 0, 2)
            dec_positional_embeddings = model.positional_embedding(x_out.type(torch.long)).permute(1, 0, 2)
            tar_embedding = z_dec_embedding + dec_positional_embeddings
            attn_layer_i = []
            for layer in model.transformer_decoder.layers:
                attn_layer_i.append(
                    layer.self_attn(tar_embedding, enc_out, enc_out)[-1].squeeze().cpu().detach().numpy())
                tar_embedding = layer.forward(tar_embedding, enc_out)

            return attn_layer_i

    def get_day_attn(self, model, args, x, src, tar, device):
        model.eval()
        with torch.no_grad():
            x_in = x[:, :args['train_length']]
            x_out = x[:, args['train_length'] - 1:args['train_length'] - 1 + args['forcast_window']]
            x_in = x_in.to(device)
            src = src.to(device)
            x_out = x_out.to(device)
            tar_in = tar[:, :-1].to(device)
            attention_masks = model._generate_square_subsequent_mask(tar_in.shape[1]).to(device)
            z_enc = torch.cat((src.unsqueeze(1), x_in.unsqueeze(1)), 1)
            z_enc_embedding = model.input_embedding(z_enc).permute(2, 0, 1)  # (sequence len,Batch size,embedding_size)
            enc_positional_embeddings = model.positional_embedding(x_in.type(torch.long)).permute(1, 0, 2)
            input_embedding = z_enc_embedding + enc_positional_embeddings
            enc_out = model.transformer_encoder(input_embedding)
            z_dec = torch.cat((tar_in.unsqueeze(1), x_out.unsqueeze(1)), 1)
            z_dec_embedding = model.input_embedding(z_dec).permute(2, 0, 1)
            dec_positional_embeddings = model.positional_embedding(x_out.type(torch.long)).permute(1, 0, 2)
            tar_embedding = z_dec_embedding + dec_positional_embeddings
            attn_layer_i = []
            for layer in model.transformer_decoder.layers:
                # print(len(layer.self_attn(enc_out,enc_out,tar_embedding,attn_mask=attention_masks)))
                # [2, batch, seq_len_dec, seq_len_enc]
                # [num_head?, batch, seq_len_dec, seq_len_enc]
                attn_layer_i.append(
                    layer.self_attn(tar_embedding, enc_out, enc_out)[-1].squeeze().cpu().detach().numpy())
                tar_embedding = layer.forward(tar_embedding, enc_out)

            return attn_layer_i

    def get_day_station_attn(self, model, args, x, src, tar, device):
        model.eval()
        with torch.no_grad():
            x_in = x[:, :args['train_length']]
            x_out = x[:, args['train_length'] - 1:args['train_length'] - 1 + args['forcast_window']]
            x_in = x_in.to(device)
            src = src.to(device)
            x_out = x_out.to(device)
            tar_in = tar[:, :, :-1].to(device)
            z_enc = torch.cat((src, x_in.unsqueeze(1)), 1)
            z_enc_embedding = model.input_embedding(z_enc).permute(2, 0, 1)  # (sequence len,Batch size,embedding_size)
            enc_positional_embeddings = model.positional_embedding(x_in.type(torch.long)).permute(1, 0, 2)
            input_embedding = z_enc_embedding + enc_positional_embeddings
            enc_out = model.transformer_encoder(input_embedding)
            z_dec = torch.cat((tar_in, x_out.unsqueeze(1)), 1)
            z_dec_embedding = model.input_embedding(z_dec).permute(2, 0, 1)
            dec_positional_embeddings = model.positional_embedding(x_out.type(torch.long)).permute(1, 0, 2)
            tar_embedding = z_dec_embedding + dec_positional_embeddings
            attn_layer_i = []
            for layer in model.transformer_decoder.layers:
                attn_layer_i.append(
                    layer.self_attn(tar_embedding, enc_out, enc_out)[-1].squeeze().cpu().detach().numpy())
                tar_embedding = layer.forward(tar_embedding, enc_out)

            return attn_layer_i

    def get_gcn_attn(self, model, args, x, src, tar, device):
        model.eval()
        with torch.no_grad():
            x_in = x[:, :args['train_length']]
            x_out = x[:, args['train_length'] - 1:args['train_length'] - 1 + args['forcast_window']]
            x_in = x_in.to(device)
            src = src.to(device)
            x_out = x_out.to(device)
            tar_in = tar[:, :, :-1].to(device)
            z_enc = model.gcn_enc(src.unsqueeze(-1))
            z_enc = z_enc.permute(2, 0, 1, 3).reshape(z_enc.shape[2], z_enc.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
            z_enc_embedding = model.fc_enc_gcn(z_enc)  # (sequence len,Batch size,embedding_size)
            enc_positional_embeddings = model.positional_embedding(x_in.type(torch.long)).permute(1, 0, 2)
            input_embedding = z_enc_embedding + enc_positional_embeddings
            enc_out = model.transformer_encoder(input_embedding)
            z_dec = model.gcn_dec(tar_in.unsqueeze(-1))
            z_dec = z_dec.permute(2, 0, 1, 3).reshape(z_dec.shape[2], z_dec.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
            z_dec_embedding = model.fc_dec_gcn(z_dec)
            dec_positional_embeddings = model.positional_embedding(x_out.type(torch.long)).permute(1, 0, 2)
            tar_embedding = z_dec_embedding + dec_positional_embeddings
            attn_layer_i = []
            for layer in model.transformer_decoder.layers:
                # print(len(layer.self_attn(enc_out,enc_out,tar_embedding,attn_mask=attention_masks)))
                # [2, batch, seq_len_dec, seq_len_enc]
                # [num_head?, batch, seq_len_dec, seq_len_enc]
                attn_layer_i.append(
                    layer.self_attn(tar_embedding, enc_out, enc_out)[-1].squeeze().cpu().detach().numpy())
                tar_embedding = layer.forward(tar_embedding, enc_out)

            return attn_layer_i

    def choose_attn_plot(self, dl, args, idx_example):
        if args['name'] == 'gcn' or args['name'] == 'day_station':
            dl_y = dl.dataset[idx_example][1].sum(axis=0).numpy()
            dl_z = dl.dataset[idx_example][2][:, 1:].sum(axis=0).numpy()
        elif args['name'] == 'raster':
            dl_y = dl.dataset[idx_example][1].sum(axis=-1).sum(axis=-1).numpy()
            dl_z = dl.dataset[idx_example][2][1:].sum(axis=-1).sum(axis=-1).numpy()
        elif args['name'] == 'day':
            dl_y = dl.dataset[idx_example][1].numpy()
            dl_z = dl.dataset[idx_example][2][1:].numpy()
        elif args['name'] == 'gcn_diff':
            dl_y = self.recover_data(dl.dataset[idx_example][1], dl.dataset[idx_example][4], ax=2).sum(axis=0)
            dl_z = self.recover_data(dl.dataset[idx_example][2][:,1:], dl.dataset[idx_example][5], ax=2).sum(axis=0)
        else:
            raise ValueError('错误的模型名称{}'.format(args['name']))
        return dl_y, dl_z

    def show_attn(self, dl, args, attn_layers, idx_example=0):
        plt.figure(figsize=(10, 5))
        dl_y, dl_z = self.choose_attn_plot(dl, args, idx_example)
        plt.plot(dl.dataset[idx_example][0].numpy()[:dl_y.shape[0]],
                 dl_y)
        plt.plot(dl.dataset[idx_example][0].numpy()[dl_y.shape[0]:dl_y.shape[0] + dl_z.shape[0]],
                 dl_z)
        plt.plot([dl_y.shape[0] + dl_z.shape[0] - 1, dl_y.shape[0] + dl_z.shape[0] - 1],
                 [0, 0.1],
                 'g--')  # not missing data
        plt.savefig('train_result/{}_attn1_length{}.jpg'.format(args['name'], dl_y.shape[0]))
        plt.figure(figsize=(10, 10))

        plt.matshow(attn_layers[0])
        plt.savefig('train_result/{}_attn2_length{}.jpg'.format(args['name'], dl_y.shape[0]))
        plt.figure()
        plt.plot(attn_layers[0][args['forcast_window'] - 1])  # not missing data
        plt.plot(attn_layers[1][args['forcast_window'] - 1])  # not missing data
        plt.plot(attn_layers[2][args['forcast_window'] - 1])
        plt.plot(attn_layers[3][args['forcast_window'] - 1])  # not missing data

        plt.legend(["attn score in layer 1", "attn score in layer 2", "attn score in layer 3", "attn score in layer 4"])
        plt.title("Attn for {}_t = 30+7".format(args['name']))  # not missing data
        plt.savefig('train_result/{}_attn3_length{}.jpg'.format(args['name'], args['train_length']))

    def get_args(self):
        args = get_model_args(self.name, train_length=self.train_length, forcast_window=self.forcast_window,
                              X_train=self.dataset, adj_matrix=self.A)
        return args

    def train_gcn(self, device):
        train_dl, test_dl = self.get_dl(SensorDataset_GCN, random=self.random)
        args = self.get_args()
        model = select_model(self.name, args).to(device)
        self.initial_files()
        self.train_step(model, args, train_dl, test_dl, device)
        torch.save(model, 'train_process/model_last_length{}.h5'.format(self.train_length))
        torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(self.train_length))
        self.save_loss(args)
        self.plot_model(model, train_dl, args)
        self.plot_model(model, test_dl, args, test=True)
        self.plot_attn(model, test_dl, args, device)

    def train_gcn_diff(self, device):
        train_dl, test_dl = self.get_dl(SensorDataset_GCN_diff, random=self.random)
        args = self.get_args()
        model = select_model(self.name, args).to(device)
        self.initial_files()
        self.train_step(model, args, train_dl, test_dl, device)
        torch.save(model, 'train_process/model_last_length{}.h5'.format(self.train_length))
        torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(self.train_length))
        self.save_loss(args)
        self.plot_model(model, train_dl, args)
        self.plot_model(model, test_dl, args, test=True)
        self.plot_attn(model, test_dl, args, device)

    def train_cnn(self, device):
        train_dl, test_dl = self.get_dl(SensorDataset4, random=self.random)
        args = self.get_args()
        model = select_model(self.name, args).to(device)
        self.initial_files()
        self.train_step(model, args, train_dl, test_dl, device)
        torch.save(model, 'train_process/model_last_length{}.h5'.format(self.train_length))
        torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(self.train_length))
        self.save_loss(args)
        self.plot_model(model, train_dl, args)
        self.plot_model(model, test_dl, args, test=True)
        self.plot_attn(model, test_dl, args, device)

    def train_day(self, device):
        train_dl, test_dl = self.get_dl(SensorDataset4, random=self.random)
        args = self.get_args()
        model = select_model(self.name, args).to(device)
        self.initial_files()
        self.train_step(model, args, train_dl, test_dl, device)
        torch.save(model, 'train_process/model_last_length{}.h5'.format(self.train_length))
        torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(self.train_length))
        self.save_loss(args)
        self.plot_model(model, train_dl, args)
        self.plot_model(model, test_dl, args, test=True)
        self.plot_attn(model, test_dl, args, device)

    def train_baseline_conv(self, device, baseline=True):
        train_dl, test_dl = self.get_dl(SensorDataset4, baseline=baseline, random=self.random)
        args = self.get_args()
        model = select_model(self.name, args).to(device)
        self.initial_files()
        self.train_step(model, args, train_dl, test_dl, device, baseline=baseline)
        torch.save(model, 'train_process/model_last_length{}.h5'.format(self.train_length))
        torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(self.train_length))
        self.save_loss(args)
        self.plot_model(model, train_dl, args)
        self.plot_model(model, test_dl, args, test=True)

    def train_baseline_gcn(self, device, baseline=True):
        train_dl, test_dl = self.get_dl(SensorDataset_GCN, baseline=baseline, random=self.random)
        args = self.get_args()
        model = select_model(self.name, args).to(device)
        self.initial_files()
        self.train_step(model, args, train_dl, test_dl, device, baseline=baseline)
        torch.save(model, 'train_process/model_last_length{}.h5'.format(self.train_length))
        torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(self.train_length))
        self.save_loss(args)
        self.plot_model(model, train_dl, args)
        self.plot_model(model, test_dl, args, test=True)

    def train(self, device):
        print('开始训练时间:', datetime.datetime.now())
        if self.name == 'Transformer_gcn' or self.name == 'Transformer_all':
            self.train_gcn(device)
        elif self.name == 'Transformer_gcn_diff':
            self.train_gcn_diff(device)
        elif self.name == 'Transformer_cnn':
            self.train_cnn(device)
        elif self.name == 'Transformer':
            self.train_day(device)
        elif self.name == 'conv_lstm':
            self.train_baseline_conv(device)
        elif self.name == 'conv_lstm2':
            self.train_baseline_conv(device)
        elif self.name == 'gcn_lstm':
            self.train_baseline_gcn(device)
        else:
            raise ValueError('错误的模型名称{}'.format(self.name))
        print('结束训练时间:', datetime.datetime.now())
