import time
import numpy as np
import torch
from DataLoader import SensorDataset4, SensorDataset_GCN, SensorDataset_baseline_conv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.model_utils import select_model, get_model_args
from ProcessData import process_time
import datetime
import os
from collections import deque
import networkx as nx
from graph_utils import normalize_adj
from models.plot import plot_all_models
from models.train_models import Train_all_models

train_length = 30
forcast_window = 7
csv_name = "Palo Alto"
criterion = torch.nn.MSELoss()
lr = 0.00001
epochs = 10000
train_epoch_loss = []
Rp_best = 10
idx_example = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_num = 2
filenames = deque(maxlen=model_num)


def Dp(y_pred, y_true, q):  # softmax更新
    return max([q * (y_pred - y_true), (1 - q) * (y_true - y_pred)])


def Rp_num_den(y_preds, y_trues, q):  # RP_loss
    numerator = np.sum([Dp(y_pred, y_true, q) for y_pred, y_true in zip(y_preds, y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator, denominator


def train_epoch(model, train_dl, optimizer, train_length=30):
    model.train()
    train_loss = 0
    n = 0
    for step, (x, src, tar, times) in enumerate(train_dl):
        # [batch, time]
        tar_in = tar[:, :-1]
        tar_out = tar[:, 1:]
        optimizer.zero_grad()
        output = model(x.to(device).float(), src.to(device).float(), tar_in.to(device))
        # output [batch, forcast_window, 1]
        loss = criterion(output.squeeze(-1), tar_out.to(device))  # not missing data
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n


def train_baseline(model, train_dl, optimizer):
    """
    训练baseline数据
    """
    model.train()
    train_loss = 0
    n = 0
    for step, (x, src, tar, times) in enumerate(train_dl):
        optimizer.zero_grad()
        output = model(src.to(device))

        loss = criterion(output, tar.to(device))
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]

    return train_loss / n


def train_gcn_epoch(model, train_dl, optimizer, train_length=30):
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
        loss = criterion(output.squeeze(-1), tar_out.to(device))  # not missing data
        loss.backward()
        optimizer.step()
        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n


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


def test_baseline(model, test_dl, forcast_window=7):
    predictions = []
    observations = []
    with torch.no_grad():
        model.eval()
        for step, (x, src, tar, times) in enumerate(test_dl):
            output = model(src.to(device))
            for p, o in zip(output.sum(axis=-1).sum(axis=-1).cpu().numpy(), tar.sum(axis=-1).sum(axis=-1).numpy()):
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den
    return Rp


def test_gcn_baseline(model, test_dl, forcast_window=7):
    predictions = []
    observations = []
    with torch.no_grad():
        model.eval()
        for step, (x, src, tar, times) in enumerate(test_dl):
            output = model(src.to(device))
            for p, o in zip(output.sum(axis=-1).cpu().numpy(), tar.sum(axis=-1).numpy()):
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den
    return Rp


def test_epoch(model, test_dl, train_length=30, forcast_window=7):
    with torch.no_grad():
        predictions = []
        observations = []

        model.eval()
        for step, (x, src, tar, times) in enumerate(test_dl):
            tar_in = src[:, -1].unsqueeze(-1)
            tar_out = tar[:, 1:]
            next_input = torch.tensor(tar_in)
            all_predictions = []

            for i in range(forcast_window):
                output = model(x.to(device), src.to(device), next_input.to(device))
                if all_predictions == []:
                    all_predictions = output[:, -1].detach()
                else:
                    all_predictions = torch.cat((all_predictions, output[:, -1].detach()), dim=-1)
                next_input = torch.cat((next_input.to(device), output[:, -1].detach()), dim=-1)
            for p, o in zip(all_predictions.cpu().numpy(), tar_out.numpy().tolist()):  # not missing data
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den
    return Rp


def test_raster_epoch(model, test_dl, train_length=30, forcast_window=7):
    with torch.no_grad():
        predictions = []
        observations = []

        model.eval()
        for step, (x, src, tar, times) in enumerate(test_dl):
            tar_in = tar[:, 0].unsqueeze(1)
            tar_out = tar[:, 1:]
            next_input = torch.tensor(tar_in)
            all_predictions = []

            for i in range(forcast_window):
                output = model(x.to(device), src.to(device), next_input.to(device))
                if all_predictions == []:
                    all_predictions = output[:, -1].unsqueeze(1).detach()
                else:
                    all_predictions = torch.cat((all_predictions, output[:, -1].unsqueeze(1).detach()), dim=1)
                next_input = torch.cat((next_input.to(device), output[:, -1].unsqueeze(1).detach()), dim=1)
            for p, o in zip(all_predictions.cpu().numpy().sum(axis=-1).sum(axis=-1),
                            tar_out.numpy().sum(axis=-1).sum(axis=-1)):  # not missing data
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den
    return Rp


def test_GCN_Transform(model, test_dl, train_length=30, forcast_window=7):
    with torch.no_grad():
        predictions = []
        observations = []

        model.eval()
        for step, (x, src, tar, times) in enumerate(test_dl):
            tar_in = tar[:, :, 0].unsqueeze(-1)
            tar_out = tar[:, :, 1:]
            next_input = tar_in
            all_predictions = []

            for i in range(forcast_window):
                output = model(x.to(device), src.to(device), next_input.to(device))  # [batch, Node, time]
                if all_predictions == []:
                    all_predictions = output[:, :, -1].unsqueeze(-1).detach()
                else:
                    all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
            for p, o in zip(all_predictions.cpu().numpy().sum(axis=1),
                            tar_out.numpy().sum(axis=1)):  # not missing data
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den
    return Rp


def change_to_date(date_num):  # 更改数字日期到真实日期
    date_num = date_num.cpu().numpy()
    date_list = []
    for date in date_num:
        date_list.append(str(date)[:4] + '-' + str(date)[4:6] + '-' + str(date)[6:8])
    return date_list


def plot_models(model, dl, train_length, forcast_window, test=False, n_plots=5, plot=False, max_=None):
    with torch.no_grad():
        model.eval()
        for step, (x, src, tar, times) in enumerate(dl):
            tar_in = src[:, -1].unsqueeze(-1)  # 取最后一个日子的输入当作transformer的起始输入
            tar_out = tar[:, 1:]
            next_input = tar_in
            all_predictions = []
            for i in range(forcast_window):
                output = model(x.to(device).float(), src.to(device), next_input.to(device).float())
                if all_predictions == []:
                    all_predictions = output[:, -1]
                else:
                    all_predictions = torch.cat((all_predictions, output[:, -1].detach()), dim=-1)
                next_input = torch.cat((next_input.to(device), output[:, -1]), dim=-1)
            # x[batch, seq_len]
            # y[batch, seq_len]
            if step >= n_plots:
                break

            with torch.no_grad():
                plt.figure(figsize=(10, 10))
                data_list = change_to_date(times[0])
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if max_:
                    plt.plot(x[0][:src.shape[1]].cpu().detach().squeeze(-1).numpy(),
                             src[0].cpu().detach().squeeze(-1).numpy() * max_, 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             all_predictions[0, :].cpu().detach().numpy() * max_,
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             tar_out[0] * max_, 'r--', linewidth=3)  # not missing data
                else:
                    plt.plot(x[0][:src.shape[1]].cpu().detach().squeeze(-1).numpy(),
                             src[0].cpu().detach().squeeze(-1).numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             all_predictions[0, :].cpu().detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             tar_out[0], 'r--', linewidth=3)  # not missing data
                plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0+{7})_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                if plot:
                    plt.show()
                else:
                    if test:
                        plt.title('test{}_length{}'.format(step + 1, train_length))
                        plt.savefig('train_result/test_val{}_length{}.jpg'.format(step + 1, train_length))
                    else:
                        plt.title('train{}_length{}'.format(step + 1, train_length))
                        plt.savefig('train_result/train_val{}_length{}.jpg'.format(step + 1, train_length))


def plot_raster_models(model, dl, train_length, forcast_window, test=False, n_plots=5, plot=False, max_=None):
    # 画出每个点的预测图片
    with torch.no_grad():
        model.eval()
        for step, (x, src, tar, times) in enumerate(dl):
            tar_in = src[:, -1].unsqueeze(1)  # 取最后一个日子的输入当作transformer的起始输入
            tar_out = tar[:, 1:]
            next_input = tar_in.to(device)
            all_predictions = []
            for i in range(forcast_window):
                output = model(x.to(device).float(), src.to(device), next_input.to(device).float())
                if all_predictions == []:
                    all_predictions = output[:, -1].unsqueeze(1)
                else:
                    all_predictions = torch.cat((all_predictions, output[:, -1].unsqueeze(1).detach()), dim=1)
                next_input = torch.cat((next_input.to(device), output[:, -1].unsqueeze(1)), dim=1)
            # x[batch, seq_len]
            # y[batch, seq_len]
            if step >= n_plots:
                break
            with torch.no_grad():
                # 总预测图
                plt.figure(figsize=(10, 10))
                data_list = change_to_date(times[0])
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if max_ is not None:
                    plt.plot(x[0][:src.shape[1]].cpu().detach().squeeze(-1).numpy(),
                             (src[0] * max_).sum(axis=-1).sum(axis=-1).cpu().detach().squeeze(-1).numpy(), 'g--',
                             linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             (all_predictions[0].cpu() * max_).sum(axis=-1).sum(axis=-1).detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             (tar_out[0] * max_).sum(axis=-1).sum(axis=-1), 'r--', linewidth=3)  # not missing data
                    plt.xlabel("x", fontsize=20)
                else:
                    plt.plot(x[0][:src.shape[1]].cpu().detach().squeeze(-1).numpy(),
                             src[0].sum(axis=-1).sum(axis=-1).cpu().detach().squeeze(-1).numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             all_predictions[0].sum(axis=-1).sum(axis=-1).cpu().detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             tar_out[0].sum(axis=-1).sum(axis=-1), 'r--', linewidth=3)  # not missing data
                    plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0)_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                if plot:
                    plt.show()
                else:
                    if test:
                        plt.title('raster_test{}_length{}'.format(step + 1, train_length))
                        plt.savefig('train_result/raster_test_val{}_length{}.jpg'.format(step + 1, train_length))
                    else:
                        plt.title('raster_train{}_length{}'.format(step + 1, train_length))
                        plt.savefig('train_result/raster_train_val{}_length{}.jpg'.format(step + 1, train_length))

                # 单点预测图
                plt.figure(figsize=(10, 10))
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if max_ is not None:
                    plt.plot(x[0][:src.shape[1]].cpu().detach().squeeze(-1).numpy(),
                             (src[0] * max_)[:, 0, 4].cpu().detach().squeeze(-1).numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             (all_predictions[0].cpu() * max_)[:, 0, 4].cpu().detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             (tar_out[0] * max_)[:, 0, 4], 'r--', linewidth=3)  # not missing data
                else:
                    plt.plot(x[0][:src.shape[1]].cpu().detach().squeeze(-1).numpy(),
                             src[0, :, 0, 4].cpu().detach().squeeze(-1).numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             all_predictions[0, :, 0, 4].cpu().detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             tar_out[0, :, 0, 4], 'r--', linewidth=3)  # not missing data
                plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0)_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                if plot:
                    plt.show()
                else:
                    if test:
                        plt.title('raster_test{}_length{}_point{}'.format(step + 1, train_length, '04'))
                        plt.savefig(
                            'train_result/raster_test_val{}_length{}_point{}.jpg'.format(step + 1, train_length, '04'))
                    else:
                        plt.title('raster_train{}_length{}_point{}'.format(step + 1, train_length, '04'))
                        plt.savefig(
                            'train_result/raster_train_val{}_length{}_point{}.jpg'.format(step + 1, train_length, '04'))


def plot_gcn_models(model, dl, train_length, forcast_window, test=False, n_plots=5, plot=False, max_=None):
    # 画出每个点的预测图片
    with torch.no_grad():
        model.eval()
        for step, (x, src, tar, times) in enumerate(dl):
            tar_in = src[:, :, -1].unsqueeze(-1)  # 取最后一个日子的输入当作transformer的起始输入
            tar_out = tar[:, :, 1:]
            next_input = tar_in.to(device)
            all_predictions = []
            for i in range(forcast_window):
                output = model(x.to(device).float(), src.to(device), next_input.to(device).float())
                if all_predictions == []:
                    all_predictions = output[:, :, -1].unsqueeze(-1)
                else:
                    all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-1)), dim=-1)
            # x[batch, seq_len]
            # y[batch, seq_len]
            if step >= n_plots:
                break
            with torch.no_grad():
                # 总预测图
                plt.figure(figsize=(10, 10))
                data_list = change_to_date(times[0])
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if max_ is not None:
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             (src[0] * max_).sum(axis=0).cpu().detach().numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             (all_predictions[0].cpu() * max_).sum(axis=0).detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             (tar_out[0] * max_).sum(axis=0), 'r--', linewidth=3)  # not missing data
                else:
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             src[0].sum(axis=0).cpu().detach().numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             all_predictions[0].sum(axis=0).cpu().detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             tar_out[0].sum(axis=0), 'r--', linewidth=3)  # not missing data
                plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0)_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                if plot:
                    plt.show()
                else:
                    if test:
                        plt.title('gcn_test{}_length{}'.format(step + 1, train_length))
                        plt.savefig('train_result/gcn_test_val{}_length{}.jpg'.format(step + 1, train_length))
                    else:
                        plt.title('gcn_train{}_length{}'.format(step + 1, train_length))
                        plt.savefig('train_result/gcn_train_val{}_length{}.jpg'.format(step + 1, train_length))

                # 单点预测图
                plt.figure(figsize=(10, 10))
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if max_ is not None:
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             (src[0] * max_)[4, :].cpu().detach().numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             (all_predictions[0].cpu() * max_)[4, :].detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             (tar_out[0] * max_)[4, :], 'r--', linewidth=3)  # not missing data
                else:
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             src[0, 4, :].cpu().detach().numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             all_predictions[0, 4, :].cpu().detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x[0][train_length:train_length + forcast_window],
                             tar_out[0, 4, :], 'r--', linewidth=3)  # not missing data
                plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0)_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                if plot:
                    plt.show()
                else:
                    if test:
                        plt.title('gcn_test{}_length{}_point{}'.format(step + 1, train_length, '04'))
                        plt.savefig(
                            'train_result/gcn_test_val{}_length{}_point{}.jpg'.format(step + 1, train_length, '04'))
                    else:
                        plt.title('gcn_train{}_length{}_point{}'.format(step + 1, train_length, '04'))
                        plt.savefig(
                            'train_result/gcn_train_val{}_length{}_point{}.jpg'.format(step + 1, train_length, '04'))


def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def get_attn(model, x, src, tar):
    model.eval()
    with torch.no_grad():
        x_in = x[:, :train_length]
        x_out = x[:, train_length - 1:train_length - 1 + forcast_window]
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
            attn_layer_i.append(layer.self_attn(tar_embedding, enc_out, enc_out)[-1].squeeze().cpu().detach().numpy())
            tar_embedding = layer.forward(tar_embedding, enc_out)

        return attn_layer_i


def get_raster_attn(model, x, src, tar):
    model.eval()
    with torch.no_grad():
        x_in = x[:, :train_length]
        x_out = x[:, train_length - 1:train_length - 1 + forcast_window]
        x_in = x_in.to(device)
        src = src.to(device)
        x_out = x_out.to(device)
        tar_in = tar[:, :-1].to(device)
        attention_masks = model._generate_square_subsequent_mask(tar_in.shape[1]).to(device)
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
            # print(len(layer.self_attn(enc_out,enc_out,tar_embedding,attn_mask=attention_masks)))
            # [2, batch, seq_len_dec, seq_len_enc]
            # [num_head?, batch, seq_len_dec, seq_len_enc]
            attn_layer_i.append(layer.self_attn(tar_embedding, enc_out, enc_out)[-1].squeeze().cpu().detach().numpy())
            tar_embedding = layer.forward(tar_embedding, enc_out)

        return attn_layer_i


def get_gcn_attn(model, x, src, tar):
    model.eval()
    with torch.no_grad():
        x_in = x[:, :train_length]
        x_out = x[:, train_length - 1:train_length - 1 + forcast_window]
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
            attn_layer_i.append(layer.self_attn(tar_embedding, enc_out, enc_out)[-1].squeeze().cpu().detach().numpy())
            tar_embedding = layer.forward(tar_embedding, enc_out)

        return attn_layer_i


def show_attn(data, attn_layers, idx_example=0):
    plt.figure(figsize=(10, 5))
    plt.plot(data[idx_example][0].numpy()[:train_length], data[idx_example][1].numpy())
    plt.plot(data[idx_example][0].numpy()[train_length:train_length + forcast_window],
             data[idx_example][2][1:].numpy())
    plt.plot([train_length + forcast_window - 1, train_length + forcast_window - 1], [0, 0.1],
             'g--')  # not missing data
    plt.savefig('train_result/attn1_length{}.jpg'.format(train_length))
    plt.figure(figsize=(10, 10))

    plt.matshow(attn_layers[0])
    plt.savefig('train_result/attn2_length{}.jpg'.format(train_length))
    plt.figure()
    plt.plot(attn_layers[0][forcast_window - 1])  # not missing data
    plt.plot(attn_layers[1][forcast_window - 1])  # not missing data
    plt.plot(attn_layers[2][forcast_window - 1])
    plt.plot(attn_layers[3][forcast_window - 1])  # not missing data

    # plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[0][119-10]) # missing data
    # plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[1][119-10]) # missing data
    # plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[2][119-10]) # missing data

    plt.legend(["attn score in layer 1", "attn score in layer 2", "attn score in layer 3", "attn score in layer 4"])
    plt.title("Attn for t = 30+7")  # not missing data
    plt.savefig('train_result/attn3_length{}.jpg'.format(train_length))


def show_raster_attn(data, attn_layers, idx_example=0):
    plt.figure(figsize=(10, 5))
    plt.plot(data[idx_example][0].numpy()[:train_length], data[idx_example][1].sum(axis=-1).sum(axis=-1).numpy())
    plt.plot(data[idx_example][0].numpy()[train_length:train_length + forcast_window],
             data[idx_example][2][1:].sum(axis=-1).sum(axis=-1).numpy())
    plt.plot([train_length + forcast_window - 1, train_length + forcast_window - 1], [0, 0.1],
             'g--')  # not missing data
    plt.savefig('train_result/raster_attn1_length{}.jpg'.format(train_length))
    plt.figure(figsize=(10, 10))

    plt.matshow(attn_layers[0])
    plt.savefig('train_result/raster_attn2_length{}.jpg'.format(train_length))
    plt.figure()
    plt.plot(attn_layers[0][forcast_window - 1])  # not missing data
    plt.plot(attn_layers[1][forcast_window - 1])  # not missing data
    plt.plot(attn_layers[2][forcast_window - 1])
    plt.plot(attn_layers[3][forcast_window - 1])  # not missing data

    plt.legend(["attn score in layer 1", "attn score in layer 2", "attn score in layer 3", "attn score in layer 4"])
    plt.title("Attn for t = 30+7")  # not missing data
    plt.savefig('train_result/raster_attn3_length{}.jpg'.format(train_length))


def show_gcn_attn(data, attn_layers, idx_example=0):
    plt.figure(figsize=(10, 5))
    plt.plot(data[idx_example][0].numpy()[:train_length], data[idx_example][1].sum(axis=0).numpy())
    plt.plot(data[idx_example][0].numpy()[train_length:train_length + forcast_window],
             data[idx_example][2][:, 1:].sum(axis=0).numpy())
    plt.plot([train_length + forcast_window - 1, train_length + forcast_window - 1], [0, 0.1],
             'g--')  # not missing data
    plt.savefig('train_result/gcn_attn1_length{}.jpg'.format(train_length))
    plt.figure(figsize=(10, 10))

    plt.matshow(attn_layers[0])
    plt.savefig('train_result/gcn_attn2_length{}.jpg'.format(train_length))
    plt.figure()
    plt.plot(attn_layers[0][forcast_window - 1])  # not missing data
    plt.plot(attn_layers[1][forcast_window - 1])  # not missing data
    plt.plot(attn_layers[2][forcast_window - 1])
    plt.plot(attn_layers[3][forcast_window - 1])  # not missing data

    plt.legend(["attn score in layer 1", "attn score in layer 2", "attn score in layer 3", "attn score in layer 4"])
    plt.title("Attn for t = 30+7")  # not missing data
    plt.savefig('train_result/gcn_attn3_length{}.jpg'.format(train_length))


def process_data(day_map, time_list, training_length, forecast_window, time=False, matrix=False):  # 通过给定数组进行dataset分类
    matrix_lags = np.zeros(
        (day_map.shape[0] - (training_length + forecast_window), training_length + forecast_window))

    print('matrix_lags.shape:', matrix_lags.shape)  # [3407. 37]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = day_map[i:i + training_length + forecast_window]  # 批次 + num_lags+prediction_horizont天数
    # np.save('train_data_matrix/day_matrix_lags.npy', matrix_lags)
    # np.save('train_data_matrix/day_long_matrix_lags.npy', matrix_lags)
    # ---------------- Train/test split
    if matrix:
        return matrix_lags, process_time(time_list, training_length, forecast_window, matrix=True)
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


def process_raster_data(raster_map, time_list, training_length, forecast_window, nx=5, ny=5,
                        time=False, matrix=False):  # raster 通过给定数组进行dataset分类
    matrix_lags = np.zeros(
        (raster_map.shape[0] - (training_length + forecast_window), training_length + forecast_window, nx, ny))

    print('matrix_lags.shape:', matrix_lags.shape)  # [3407. 37, 5, 5]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = raster_map[i:i + training_length + forecast_window, :,
                         :]  # 批次 + num_lags+prediction_horizont天数

    if matrix:
        return matrix_lags, process_time(time_list, training_length, forecast_window, matrix=True)
    # np.save('train_data_matrix/raster_matrix_lags.npy', matrix_lags)
    # np.save('train_data_matrix/raster_long_matrix_lags.npy', matrix_lags)
    # ---------------- Train/test split
    train_dataset = np.zeros((i_train, training_length + forecast_window, nx, ny))  # [3400, 37, 5, 5]
    test_dataset = np.zeros((i_test - i_train, forecast_window + training_length, nx, ny))  # [7, 37]

    for i in range(nx):
        for j in range(ny):
            train_dataset[:, :, i, j] = matrix_lags[:i_train, :, i, j]
            test_dataset[:, :, i, j] = matrix_lags[i_train:, :, i, j]

    # np.save('train_dataset.npy', train_dataset)
    # np.save('test_dataset.npy', test_dataset)
    print('train_dataset:', train_dataset.shape)
    print('test_dataset:', test_dataset.shape)
    if time:
        X_train_time, X_test_time = process_time(time_list, training_length, forecast_window)
        return train_dataset, test_dataset, X_train_time, X_test_time
    return train_dataset, test_dataset


def process_gcn_data(G_timeseries_map, time_list, training_length, forecast_window,
                     time=False, matrix=False):  # gcn 通过给定数组进行dataset分类
    matrix_lags = np.zeros(
        (G_timeseries_map.shape[-1] - (training_length + forecast_window),
         G_timeseries_map.shape[0], training_length + forecast_window))
    print('G_timeseries_map.shape:', G_timeseries_map.shape)
    print('matrix_lags.shape:', matrix_lags.shape)  # [3407. 47, 37]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = G_timeseries_map[:, i:i + training_length + forecast_window]  # 批次 +
        # num_lags+prediction_horizont天数
    # np.save('train_data_matrix/G_matrix_lags.npy', matrix_lags)
    # np.save('train_data_matrix/G_long_matrix_lags.npy', matrix_lags)
    if matrix:
        return matrix_lags, process_time(time_list, training_length, forecast_window, matrix=True)
    train_dataset = matrix_lags[:i_train, :, :]  # [batch, 47, 37]
    test_dataset = matrix_lags[i_train:, :, :]  # [batch, 47, 37]
    print('train_dataset:', train_dataset.shape)
    print('test_dataset:', test_dataset.shape)
    if time:
        X_train_time, X_test_time = process_time(time_list, training_length, forecast_window)
        return train_dataset, test_dataset, X_train_time, X_test_time
    return train_dataset, test_dataset


def process_gcn_log_data(G_timeseries_log_map, time_list, training_length, forecast_window,
                         time=False, matrix=False):  # gcn 通过给定数组进行dataset分类
    matrix_lags = np.zeros(
        (G_timeseries_log_map.shape[-1] - (training_length + forecast_window),
         G_timeseries_log_map.shape[0], training_length + forecast_window))
    print('G_timeseries_log_map.shape:', G_timeseries_log_map.shape)
    print('matrix_lags.shape:', matrix_lags.shape)  # [3407. 47, 37]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = G_timeseries_log_map[:, i:i + training_length + forecast_window]  # 批次 +
        # num_lags+prediction_horizont天数
    # np.save('train_data_matrix/G_matrix_log_lags.npy', matrix_lags)
    # np.save('train_data_matrix/G_long_matrix_log_lags.npy', matrix_lags)
    if matrix:
        return matrix_lags, process_time(time_list, training_length, forecast_window, matrix=True)
    train_dataset = matrix_lags[:i_train, :, :]  # [batch, 47, 37]
    test_dataset = matrix_lags[i_train:, :, :]  # [batch, 47, 37]
    print('train_dataset:', train_dataset.shape)
    print('test_dataset:', test_dataset.shape)
    if time:
        X_train_time, X_test_time = process_time(time_list, training_length, forecast_window)
        return train_dataset, test_dataset, X_train_time, X_test_time
    return train_dataset, test_dataset


def get_time(year, month, day):  # 将年月日转化为数字信息
    if len(str(month)) > 1:
        if len(str(day)) > 1:
            time = int(str(year) + str(month) + str(day))
        else:
            time = int(str(year) + str(month) + '0' + str(day))
    else:
        if len(str(day)) > 1:
            time = int(str(year) + '0' + str(month) + str(day))
        else:
            time = int(str(year) + '0' + str(month) + '0' + str(day))
    return time


def get_date_data_PALO(day_map, time_list, year_begin=2011, month_begin=7, day_begin=29, year_end=2020, month_end=12,
                       day_end=31):  # 获取指定的年月的数据
    begin_time = get_time(year_begin, month_begin, day_begin)
    end_time = get_time(year_end, month_end, day_end)
    begin_index = np.argwhere(time_list == begin_time)[0][0]  # 0
    end_index = np.argwhere(time_list == end_time)[0][0]  # 2195
    day_index_map = []
    for i in range(begin_index, end_index):
        day_index_map.append(day_map[i])
    return np.array(day_index_map), time_list[begin_index:end_index]


def get_gcn_date_PALO(day_map, time_list, year_begin=2011, month_begin=7, day_begin=29, year_end=2020, month_end=12,
                      day_end=31):
    begin_time = get_time(year_begin, month_begin, day_begin)
    end_time = get_time(year_end, month_end, day_end)
    begin_index = np.argwhere(time_list == begin_time)[0][0]  # 0
    end_index = np.argwhere(time_list == end_time)[0][0]  # 2195
    day_index_map = day_map[:, begin_index:end_index]
    return np.array(day_index_map), time_list[begin_index:end_index]


def train_day(Rp_best=10, normalize=True):
    day_map = np.load('train_data/normalized.npy')
    time_list = np.load('train_data/time_list.npy')
    # day_map, time_list = get_date_data_PALO(day_map, time_list, year_begin=2011, year_end=2017, month_end=8,
    #                                         day_end=1)  # 筛选日期
    day_map[np.isnan(day_map)] = 0
    day_map[day_map == 0] = np.random.normal(np.zeros_like(day_map[day_map == 0]), 0.001)
    train_dataset, test_dataset, X_train_time, X_test_time = process_data(day_map, time_list, train_length,
                                                                          forcast_window, time=True)
    train_data = SensorDataset4(train_dataset, X_train_time, train_length, forcast_window)
    test_data = SensorDataset4(test_dataset, X_test_time, train_length, forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, feature_size]
    test_dl = DataLoader(test_data, batch_size=1)
    args = get_model_args(name='Transformer', train_length=train_length,
                          forcast_window=forcast_window, X_train=train_dataset)
    model = select_model(name='Transformer', args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    files = os.listdir("train_process/")
    if len(files):  # 初始化文件夹里面的文件命
        for file in files:
            if (file.split('.')[-1] == 'h5' and file.split('_')[1] != 'last'):
                filenames.append("train_process/" + file.replace('.h5', ''))
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(args['epochs'])):
        start = time.time()
        train_loss = []

        l_t = train_epoch(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp = test_epoch(model, test_dl, train_length, forcast_window)

        if Rp_best > Rp:
            Rp_best = Rp
            save_model(model, e + 1, train_length, Rp, model_name='day')
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print("Epoch {}: Train loss: {:.6f} \t R_p={:.3f}\tcost_time={:.3f}s".format(e + 1,
                                                                                     np.mean(train_loss), Rp,
                                                                                     end - start))
    print('结束训练时间:', datetime.datetime.now())
    save_loss(args)
    torch.save(model, 'train_process/model_last_length{}.h5'.format(train_length))
    torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(train_length))
    plot_models(model, train_dl, train_length, forcast_window)
    plot_models(model, test_dl, train_length, forcast_window, test=True)

    attn_layers = get_attn(model, test_data[idx_example][0].unsqueeze(0), test_data[idx_example][1].unsqueeze(0),
                           test_data[idx_example][2].unsqueeze(0))
    show_attn(test_data, attn_layers)


def save_model(model, e, train_length, Rp, file_name="train_process/", model_name="GCN"):  # 模型的保存,只保存一定的模型
    save_name = 'train_process/model{}_epoch{}_length{}_Rp_{:.4f}'.format(model_name, e, train_length, Rp)
    torch.save(model, save_name + '.h5')
    torch.save(model.state_dict(), save_name + '.pkl')
    if len(filenames) >= model_num:
        os.remove(filenames[0] + '.h5')
        os.remove(filenames[0] + '.pkl')
    filenames.append(save_name)


def save_model2(model, e, train_length, Rp, file_name="train_process/", model_name="GCN"):  # 模型的保存,效率比较低
    torch.save(model, 'train_process/model{}_epoch{}_length{}_Rp_{:.4f}.h5'.format(model_name, e, train_length, Rp))
    files = os.listdir(file_name)  # 乱序读取
    files.sort(key=lambda x: float(x.split('_')[-1][:-3]), reverse=True)  # 改为从大到小读取
    if len(files) > model_num:
        os.remove(filenames[0])
    print(files)
    filenames.append(file_name + files[-1])


def Train_raster_day(Rp_best, normalize=True):
    raster_map = np.load('train_data/raster_map.npy')  # [3444,5,5]
    time_list = np.load('train_data/time_list.npy')  # [3444,]
    if normalize:
        max_ = np.max(raster_map, axis=0)
        # np.save('train_data/max_raster.npy', max_)
        raster_map = raster_map / max_
        raster_map[np.isnan(raster_map)] = 0
        raster_map[raster_map == 0] = np.random.normal(np.zeros_like(raster_map[raster_map == 0]), 0.001)
    # raster_map, time_list = get_date_data_PALO(raster_map, time_list, year_begin=2011, year_end=2017, month_end=8,
    #                                            day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_raster_data(raster_map, time_list, train_length,
                                                                                 forcast_window, time=True)
    train_data = SensorDataset4(train_dataset, X_train_time, train_length, forcast_window)
    test_data = SensorDataset4(test_dataset, X_test_time, train_length, forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, feature_size]
    test_dl = DataLoader(test_data, batch_size=1)
    args = get_model_args(name='Transformer_cnn', train_length=train_length,
                          forcast_window=forcast_window, X_train=train_dataset)
    model = select_model(name='Transformer_cnn', args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    files = os.listdir("train_process/")
    if len(files):  # 初始化文件夹里面的文件命
        for file in files:
            if (file.split('.')[-1] == 'h5' and file.split('_')[1] != 'last'):
                filenames.append("train_process/" + file.replace('.h5', ''))
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(args['epochs'])):
        start = time.time()
        train_loss = []

        l_t = train_epoch(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp = test_raster_epoch(model, test_dl, train_length, forcast_window)

        if Rp_best > Rp:
            Rp_best = Rp
            save_model(model, e + 1, train_length, Rp, model_name='raster')
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print("Epoch {}: Train loss: {:.6f} \t R_p={:.3f}\tcost_time={:.4f}s".format(e + 1,
                                                                                     np.mean(train_loss), Rp,
                                                                                     end - start))
    print('结束训练时间:', datetime.datetime.now())
    save_loss(args)
    torch.save(model, 'train_process/model_last_length{}.h5'.format(train_length))
    torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(train_length))
    plot_raster_models(model, train_dl, train_length, forcast_window)
    plot_raster_models(model, test_dl, train_length, forcast_window, test=True)

    attn_layers = get_raster_attn(model, test_data[idx_example][0].unsqueeze(0), test_data[idx_example][1].unsqueeze(0),
                                  test_data[idx_example][2].unsqueeze(0))
    show_raster_attn(test_data, attn_layers)


def Train_Gcn_Transform(Rp_best, normalize=True):
    G_timeseries_map = np.load('train_data/G_timeseries_map.npy')
    Graph = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    time_list = np.load('train_data/time_list.npy')  # [3444,]

    A = nx.adjacency_matrix(Graph)
    A = normalize_adj(A).todense()
    A = torch.tensor(A, dtype=torch.float32).to(device)
    if normalize:
        max_ = np.load('train_data/G_max_.npy')
        G_timeseries_map = G_timeseries_map / max_
        G_timeseries_map[G_timeseries_map == 0] = np.random.normal(
            np.zeros_like(G_timeseries_map[G_timeseries_map == 0]),
            0.001)
    train_dataset, test_dataset, X_train_time, X_test_time = \
        process_gcn_data(G_timeseries_map, time_list, training_length=train_length, forecast_window=forcast_window,
                         time=True)

    train_data = SensorDataset_GCN(train_dataset, X_train_time, train_length, forcast_window)
    test_data = SensorDataset_GCN(test_dataset, X_test_time, train_length, forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, Nodes, Times]
    test_dl = DataLoader(test_data, batch_size=1)
    args = get_model_args(name='Transformer_gcn', train_length=train_length, forcast_window=forcast_window,
                          X_train=train_dataset, adj_matrix=A)
    model = select_model('Transformer_gcn', args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    files = os.listdir("train_process/")
    if len(files):  # 初始化文件夹里面的文件命
        for file in files:
            if (file.split('.')[-1] == 'h5' and file.split('_')[1] != 'last'):
                filenames.append("train_process/" + file.replace('.h5', ''))
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(args['epochs'])):
        start = time.time()
        train_loss = []

        l_t = train_gcn_epoch(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp = test_GCN_Transform(model, test_dl, train_length, forcast_window)

        if Rp_best > Rp:
            Rp_best = Rp
            save_model(model, e + 1, train_length, Rp)
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print("Epoch {}: Train loss: {:.6f} \t R_p={:.3f}\tcost_time={:.4f}s".format(e + 1,
                                                                                     np.mean(train_loss), Rp,
                                                                                     end - start))
    print('结束训练时间:', datetime.datetime.now())
    torch.save(model, 'train_process/model_last_length{}.h5'.format(train_length))
    torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(train_length))
    save_loss(args)
    plot_gcn_models(model, train_dl, train_length, forcast_window)
    plot_gcn_models(model, test_dl, train_length, forcast_window, test=True)

    attn_layers = get_gcn_attn(model, test_dl.dataset[idx_example][0].unsqueeze(0),
                               test_dl.dataset[idx_example][1].unsqueeze(0),
                               test_dl.dataset[idx_example][2].unsqueeze(0))
    show_gcn_attn(test_data, attn_layers)


def Train_baseline(Rp_best):
    raster_map = np.load('train_data/raster_map.npy')  # [3444,5,5]
    time_list = np.load('train_data/time_list.npy')  # [3444,]
    max_ = np.load('train_data/max_raster.npy')
    raster_map = raster_map / max_
    raster_map[np.isnan(raster_map)] = 0
    raster_map[raster_map == 0] = np.random.normal(np.zeros_like(raster_map[raster_map == 0]), 0.001)
    # raster_map, time_list = get_date_data_PALO(raster_map, time_list, year_begin=2011, year_end=2017, month_end=8,
    #                                            day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_raster_data(raster_map, time_list, train_length,
                                                                                 forcast_window, time=True)
    train_data = SensorDataset_baseline_conv(train_dataset, X_train_time, train_length, forcast_window)
    test_data = SensorDataset_baseline_conv(test_dataset, X_test_time, train_length, forcast_window)
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)  # [batch_size, feature_size]
    test_dl = DataLoader(test_data, batch_size=1)
    args = get_model_args(name='conv_lstm', train_length=train_length,
                          forcast_window=forcast_window, X_train=train_dataset)
    model = select_model(name='conv_lstm', args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    files = os.listdir("train_process/")
    if len(files):  # 初始化文件夹里面的文件命
        for file in files:
            if (file.split('.')[-1] == 'h5' and file.split('_')[1] != 'last'):
                filenames.append("train_process/" + file.replace('.h5', ''))
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(args['epochs'])):
        start = time.time()
        train_loss = []

        l_t = train_baseline(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp = test_baseline(model, test_dl, train_length)

        if Rp_best > Rp:
            Rp_best = Rp
            save_model(model, e + 1, train_length, Rp, model_name='conv_lstm')
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print("Epoch {}: Train loss: {:.6f} \t R_p={:.3f}\tcost_time={:.4f}s".format(e + 1,
                                                                                     np.mean(train_loss), Rp,
                                                                                     end - start))
    print('结束训练时间:', datetime.datetime.now())
    save_loss(args)
    torch.save(model, 'train_process/model_last_length{}.h5'.format(train_length))
    torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(train_length))
    plot_train = plot_all_models(model, train_dl, args)
    plot_test = plot_all_models(model, test_dl, args)
    plot_train.plot_baseline(figsize=(20, 10))
    plot_test.plot_baseline(test=True, figsize=(20, 10))


def Train_gcn_baseline(Rp_best):
    G_timeseries_map = np.load('train_data/G_timeseries_map.npy')
    Graph = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    max_ = np.load('train_data/G_max_.npy')
    time_list = np.load('train_data/time_list.npy')  # [3444,]
    A = nx.adjacency_matrix(Graph)
    A = normalize_adj(A).todense()
    A = torch.tensor(A, dtype=torch.float32).to(device)
    G_timeseries_map = G_timeseries_map / max_
    G_timeseries_map[G_timeseries_map == 0] = np.random.normal(np.zeros_like(G_timeseries_map[G_timeseries_map == 0]),
                                                               0.001)
    train_dataset, test_dataset, X_train_time, X_test_time = \
        process_gcn_data(G_timeseries_map, time_list, training_length=train_length, forecast_window=forcast_window,
                         time=True)
    X_train_time, X_test_time = process_time(time_list, train_length=train_length, forcast_window=forcast_window)
    train_data = SensorDataset_GCN(train_dataset, X_train_time, train_length, forcast_window, baseline=True)
    test_data = SensorDataset_GCN(test_dataset, X_test_time, train_length, forcast_window, baseline=True)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, Nodes, Times]
    test_dl = DataLoader(test_data, batch_size=1)
    args = get_model_args(name='gcn_lstm', train_length=train_length,
                          forcast_window=forcast_window, X_train=train_dataset, adj_matrix=A)
    model = select_model(name='gcn_lstm', args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    files = os.listdir("train_process/")
    if len(files):  # 初始化文件夹里面的文件命
        for file in files:
            if (file.split('.')[-1] == 'h5' and file.split('_')[1] != 'last'):
                filenames.append("train_process/" + file.replace('.h5', ''))
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(args['epochs'])):
        start = time.time()
        train_loss = []

        l_t = train_baseline(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp = test_gcn_baseline(model, test_dl, train_length)

        if Rp_best > Rp:
            Rp_best = Rp
            save_model(model, e + 1, train_length, Rp, model_name='gcn_lstm')
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print("Epoch {}: Train loss: {:.6f} \t R_p={:.3f}\tcost_time={:.4f}s".format(e + 1,
                                                                                     np.mean(train_loss), Rp,
                                                                                     end - start))
    print('结束训练时间:', datetime.datetime.now())
    save_loss(args)
    torch.save(model, 'train_process/model_last_length{}.h5'.format(train_length))
    torch.save(model.state_dict(), 'train_process/model_last_length{}.pkl'.format(train_length))
    plot_train = plot_all_models(model, train_dl, args)
    plot_test = plot_all_models(model, test_dl, args)
    plot_train.plot_baseline(figsize=(20, 10))
    plot_test.plot_baseline(test=True, figsize=(20, 10))


def Train_split(dataset_num=0, name='', long=False, random=False):  # 将训练，测试集按比例分配
    if dataset_num == 0:
        if long:
            dataset = np.load('train_data_matrix/day_long_matrix_lags.npy')
        else:
            dataset = np.load('train_data_matrix/day_matrix_lags.npy')
        max_ = np.load('train_data/max_.npy')
    elif dataset_num == 1:
        if long:
            dataset = np.load('train_data_matrix/raster_long_matrix_lags.npy')
        else:
            dataset = np.load('train_data_matrix/raster_matrix_lags.npy')
        max_ = np.load('train_data/max_raster.npy')
    elif dataset_num == 2:
        if long:
            dataset = np.load('train_data_matrix/G_long_matrix_lags.npy')
        else:
            dataset = np.load('train_data_matrix/G_matrix_lags.npy')
        max_ = np.load('train_data/G_max_.npy')
    elif dataset_num == 3:
        if long:
            dataset = np.load('train_data_matrix/G_long_matrix_log_lags.npy')
        else:
            dataset = np.load('train_data_matrix/G_matrix_log_lags.npy')
        max_ = np.load('train_data/G_max_logs.npy')
    else:
        raise ValueError('错误的数字{}'.format(dataset_num))
    if long:
        time_list = np.load('train_data_matrix/time_long_matrix_lags.npy')
    else:
        time_list = np.load('train_data_matrix/time_matrix_lags.npy')
    Graph = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    A = nx.adjacency_matrix(Graph)
    A = normalize_adj(A).todense()
    A = torch.tensor(A, dtype=torch.float32).to(device)
    Train = Train_all_models(name=name, dataset=dataset, time_list=time_list,
                             train_length=train_length, forcast_window=forcast_window, A=A, max_=max_, random=random)
    Train.train(device)


def Train_split_select(dataset_num=0, name=''):  # 将训练，测试集18年前按比例分配
    time_list = np.load('train_data/time_list.npy')
    if dataset_num == 0:
        day_map = np.load('train_data/normalized.npy')
        day_map, time_list = get_date_data_PALO(day_map, time_list, year_begin=2011, year_end=2017, month_end=8,
                                                day_end=1)
        dataset, time_list = process_data(day_map, time_list, train_length, forcast_window, matrix=True)
        max_ = np.load('train_data/max_.npy')

    elif dataset_num == 1:
        raster_map = np.load('train_data/raster_map.npy')
        max_ = np.load('train_data/max_raster.npy')
        raster_map = raster_map / max_
        raster_map[np.isnan(raster_map)] = 0
        raster_map, time_list = get_date_data_PALO(raster_map, time_list, year_begin=2011, year_end=2017, month_end=8,
                                                   day_end=1)
        dataset, time_list = process_raster_data(raster_map, time_list, train_length, forcast_window, matrix=True)


    elif dataset_num == 2:
        G_timeseries_map = np.load('train_data/G_timeseries_map.npy')
        max_ = np.load('train_data/G_max_.npy')
        G_timeseries_map = G_timeseries_map / max_
        G_timeseries_map, time_list = get_gcn_date_PALO(G_timeseries_map, time_list, year_begin=2011, year_end=2017,
                                                        month_end=8,
                                                        day_end=1)
        dataset, time_list = process_gcn_data(G_timeseries_map, time_list, train_length,
                                              forcast_window, matrix=True)
    elif dataset_num == 3:
        G_timeseries_map = np.load('train_data/G_timeseries_log_map.npy')
        max_ = np.load('train_data/G_max_logs.npy')
        G_timeseries_map = G_timeseries_map / max_
        G_timeseries_map, time_list = get_gcn_date_PALO(G_timeseries_map, time_list, year_begin=2011, year_end=2017,
                                                        month_end=8,
                                                        day_end=1)
        dataset, time_list = process_gcn_log_data(G_timeseries_map, time_list, train_length,
                                                  forcast_window, matrix=True)
        os.system('pause')

    else:
        raise ValueError('错误的数字{}'.format(dataset_num))
    Graph = nx.read_gpickle('train_data/GCN_Graph.gpickle')
    A = nx.adjacency_matrix(Graph)
    A = normalize_adj(A).todense()
    A = torch.tensor(A, dtype=torch.float32).to(device)
    Train = Train_all_models(name=name, dataset=dataset, time_list=time_list,
                             train_length=train_length, forcast_window=forcast_window, A=A, max_=max_)
    Train.train(device)


if __name__ == '__main__':
    # train_day()
    # Train_raster_day(Rp_best)
    # Train_Gcn_Transform(Rp_best)
    # Train_baseline(Rp_best)
    # Train_gcn_baseline(Rp_best)

    Train_split(dataset_num=1, name='conv_lstm2', long=False, random=True)
    # Train_split_select(dataset_num=3, name='Transformer_gcn')
