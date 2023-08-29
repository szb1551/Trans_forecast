import numpy
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def change_to_date(date_num):  # 更改数字日期到真实日期
    date_num = date_num.cpu().numpy()
    date_list = []
    for date in date_num:
        date_list.append(str(date)[:4] + '-' + str(date)[4:6] + '-' + str(date)[6:8])
    return date_list


class plot_all_models:
    def __init__(self, model, dl, args, n_plots=5):
        self.model = model.eval()
        self.dl = dl
        self.args = args
        self.n_plots = n_plots
        self.step = 0

    def get_output_baseline(self, max_=None, time_=False, device='cpu', station=False, grid=False):
        outputs = []
        date_lists = []
        if station or grid:
            output_contrast = []
        with torch.no_grad():
            for step, (x, src, tar, times) in enumerate(self.dl):
                if step >= self.n_plots:
                    break
                output = self.model(src.to(device))
                if max_ is not None:
                    output = output.cpu() * max_
                    tar = tar * max_
                data_list = change_to_date(times[0])
                outputs.append(output)
                date_lists.append(data_list)
                if station or grid:
                    output_contrast.append(tar)
        if time_:
            return outputs, date_lists
        else:
            if station or grid:
                return outputs, output_contrast
            return outputs

    def get_output_Transformer(self, max_=None, grid=False):
        outputs = []
        if grid:
            outputs_grids = []
        with torch.no_grad():
            for step, (x, src, tar, times) in enumerate(self.dl):
                if step >= self.n_plots:
                    break
                tar_in = src[:, -1].unsqueeze(1)  # 取最后一个日子的输入当作transformer的起始输入
                tar_out = tar[:, 1:]
                next_input = tar_in
                all_predictions = []
                for i in range(self.args['forcast_window']):
                    output = self.model(x.to(device).float(), src.to(device), next_input.to(device).float())
                    if all_predictions == []:
                        all_predictions = output[:, -1].squeeze(-1).unsqueeze(1)
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, -1].squeeze(-1).unsqueeze(1).detach()),
                                                    dim=1)
                    next_input = torch.cat((next_input.to(device), output[:, -1].squeeze(-1).unsqueeze(1)), dim=1)
                if max_ is not None:
                    all_predictions = np.array(all_predictions.cpu()) * max_
                    tar_out = np.array(tar_out) * max_
                outputs.append(all_predictions)
                if grid:
                    outputs_grids.append(tar_out)
            if grid:
                return outputs, outputs_grids
            return outputs

    def get_output_Transformer_gcn(self, max_=None, station=False):
        outputs = []
        if station:
            tar_output = []
        with torch.no_grad():
            self.model.eval()
            for step, (x, src, tar, times) in enumerate(self.dl):
                tar_in = src[:, :, -1].unsqueeze(-1)  # 取最后一个日子的输入当作transformer的起始输入
                tar_out = tar[:, :, 1:]
                next_input = tar_in.to(device)
                all_predictions = []
                for i in range(self.args['forcast_window']):
                    output = self.model(x.to(device).float(), src.to(device), next_input.to(device).float())
                    if all_predictions == []:
                        all_predictions = output[:, :, -1].unsqueeze(-1)
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                    next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-1)), dim=-1)
                if max_ is not None:
                    all_predictions = np.array(all_predictions.cpu()) * max_
                    tar_out = np.array(tar_out) * max_
                outputs.append(all_predictions)
                if station:
                    tar_output.append(tar_out)
            if station:
                return outputs, tar_output
            return outputs

    def get_target(self, max_=None):
        targets = []
        with torch.no_grad():
            for step, (x, src, tar, times) in enumerate(self.dl):
                if step >= self.n_plots:
                    break
                tar = tar[:, 1:]
                if max_ is not None:
                    tar = tar * max_
                targets.append(tar)
        return targets

    def recover_data(self, diff_data, diff_begin):
        pad_diff_data = np.pad(diff_data, ((0, 0), (0, 0), (1, 0)), mode='constant')
        pad_diff_data[:, :, 0] = diff_begin[:, :, 0]
        recovered_data = np.cumsum(pad_diff_data, axis=-1)
        return recovered_data

    def plot_baseline(self, plot=False, test=False, max_=None, figsize=(10, 10)):
        with torch.no_grad():
            for step, (x, src, tar, times) in enumerate(self.dl):
                self.step = step
                if step >= self.n_plots:
                    break
                # src[batch, 30, 5, 5]
                output = self.model(src.to(device))
                plt.figure(figsize=figsize)
                data_list = change_to_date(times[0])
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if max_ is not None:
                    src = src * max_
                    tar = tar * max_
                    output = output.cpu() * max_
                plot_this = self.plot()
                plot_this(x, src, output, tar)
                plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0+{7})_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                self.plot_with_test(plot, test)

    def plot_gcn(self, plot=False, test=False, max_=None):
        with torch.no_grad():
            for step, (x, src, tar, times) in enumerate(self.dl):
                self.step = step
                if step >= self.n_plots:
                    break
                # src[batch, 30, 5, 5]
                tar_in = src[:, :, -1].unsqueeze(-1)  # 取最后一个日子的输入当作transformer的起始输入
                tar_out = tar[:, :, 1:]
                next_input = tar_in.to(device)
                all_predictions = []
                for i in range(self.args['forcast_window']):
                    output = self.model(x.to(device).float(), src.to(device), next_input.to(device).float())
                    if all_predictions == []:
                        all_predictions = output[:, :, -1].unsqueeze(-1)
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                    next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-1)), dim=-1)
                plt.figure(figsize=(10, 10))
                data_list = change_to_date(times[0])
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if max_ is not None:
                    src = src * max_
                    tar_out = tar_out * max_
                    all_predictions = all_predictions.cpu() * max_
                plot_this = self.plot()
                plot_this(x, src, all_predictions, tar_out)
                plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0+{7})_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                self.plot_with_test(plot, test)

    def plot_gcn_diff(self, plot=False, test=False, max_=None):
        with torch.no_grad():
            for step, (x, src, tar, times, src_begin, tar_begin) in enumerate(self.dl):
                self.step = step
                if step >= self.n_plots:
                    break
                # src[batch, 30, 5, 5]
                tar_in = tar[:, :, 0].unsqueeze(-1)  # 取最后一个日子的输入当作transformer的起始输入
                tar_out = tar[:, :, 1:]
                next_input = tar_in.to(device)
                all_predictions = []
                for i in range(self.args['forcast_window']):
                    output = self.model(x.to(device).float(), src.to(device), next_input.to(device).float())
                    if all_predictions == []:
                        all_predictions = output[:, :, -1].unsqueeze(-1)
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-1).detach()), dim=-1)
                    next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-1)), dim=-1)
                all_predictions = all_predictions.cpu().numpy()
                plt.figure(figsize=(10, 10))
                data_list = change_to_date(times[0])
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                src = self.recover_data(src, src_begin)
                tar_out = self.recover_data(tar_out, tar_begin)
                all_predictions = self.recover_data(all_predictions, tar_begin)
                if max_ is not None:
                    src = src * max_
                    tar_out = tar_out * max_
                    all_predictions = all_predictions * max_
                src = np.exp(src) - np.ones(src.shape)
                tar_out = np.exp(tar_out) - np.ones(tar_out.shape)
                all_predictions = np.exp(all_predictions) - np.ones(all_predictions.shape)
                plot_this = self.plot()
                plot_this(x, src, all_predictions, tar_out)
                plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0+{7})_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                self.plot_with_test(plot, test, add_num=1)
    def plot_Trans(self, plot=False, test=False, max_=None):
        with torch.no_grad():
            for step, (x, src, tar, times) in enumerate(self.dl):
                self.step = step
                if step >= self.n_plots:
                    break
                # src[batch, 30, 5, 5]
                tar_in = src[:, -1].squeeze(-1).unsqueeze(1)  # 取最后一个日子的输入当作transformer的起始输入
                tar_out = tar[:, 1:]
                next_input = tar_in.to(device)
                all_predictions = []
                for i in range(self.args['forcast_window']):
                    output = self.model(x.to(device).float(), src.to(device), next_input.to(device).float())
                    if all_predictions == []:
                        all_predictions = output[:, -1].squeeze(-1).unsqueeze(1)
                    else:
                        all_predictions = torch.cat((all_predictions, output[:, -1].squeeze(-1).unsqueeze(1).detach()), dim=1)
                    next_input = torch.cat((next_input.to(device), output[:, -1].squeeze(-1).unsqueeze(1)), dim=1)
                plt.figure(figsize=(10, 10))
                data_list = change_to_date(times[0])
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if max_ is not None:
                    src = src * max_
                    tar_out = tar_out * max_
                    all_predictions = all_predictions.cpu() * max_
                plot_this = self.plot()
                plot_this(x, src, all_predictions, tar_out)
                plt.xlabel("x", fontsize=20)
                plt.legend(["$[0,t_0+{7})_{his}$", "$[t_0,t_0+{7})_{predicted}$", "$[t_0,t_0+{7})_{true}$"])
                plt.grid()
                self.plot_with_test(plot, test)

    def plot(self):
        if self.args['name'] == 'conv_lstm' or self.args['name'] == 'raster' or self.args['name']=='conv_lstm2':
            return self.plot_raster_baseline
        elif self.args['name'] == 'gcn_lstm' or self.args['name'] == 'gcn' or\
                self.args['name'] == 'day_station':
            return self.plot_gcn_baseline
        elif self.args['name'] == 'gcn_diff':
            return self.plot_gcn_diff_baseline
        elif self.args['name'] == 'day':
            return self.plot_day_baseline
        else:
            raise ValueError('传入无效参数{}'.format(self.args['name']))

    def plot_gcn_baseline(self, x, src, output, tar):
        plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                 src[0].sum(axis=0).cpu().detach().numpy(), 'g--', linewidth=3)
        plt.plot(x[0][self.args['train_length']:self.args['train_length'] + self.args['forcast_window']],
                 output[0].sum(axis=0).detach().cpu().numpy(),
                 'b--', linewidth=3)
        plt.plot(x[0][self.args['train_length']:self.args['train_length'] + self.args['forcast_window']],
                 tar[0].sum(axis=0), 'r--', linewidth=3)  # not missing data

    def plot_gcn_diff_baseline(self, x, src, output, tar):
        plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                 src[0].sum(axis=0), 'g--', linewidth=3)
        plt.plot(x[0][src.shape[2]:src.shape[2] + tar.shape[2]],
                 output[0].sum(axis=0),
                 'b--', linewidth=3)
        plt.plot(x[0][src.shape[2]:src.shape[2] + tar.shape[2]],
                 tar[0].sum(axis=0), 'r--', linewidth=3)  # not missing data

    def plot_raster_baseline(self, x, src, output, tar):
        plt.plot(x[0][:src.shape[1]].detach().numpy(),
                 src[0].sum(axis=-1).sum(axis=-1).detach().numpy(), 'g--', linewidth=3)
        plt.plot(x[0][self.args['train_length']:self.args['train_length'] + self.args['forcast_window']],
                 output[0].sum(axis=-1).sum(axis=-1).detach().cpu().numpy(),
                 'b--', linewidth=3)
        plt.plot(x[0][self.args['train_length']:self.args['train_length'] + self.args['forcast_window']],
                 tar[0].sum(axis=-1).sum(axis=-1), 'r--', linewidth=3)

    def plot_day_baseline(self, x, src, output, tar):
        plt.plot(x[0][:src.shape[1]].cpu().detach().squeeze(-1).numpy(),
                 src[0].cpu().detach().squeeze(-1).numpy(), 'g--', linewidth=3)
        # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
        plt.plot(x[0][self.args['train_length']:self.args['train_length'] + self.args['forcast_window']],
                 output[0, :].cpu().detach().numpy(),
                 'b--', linewidth=3)
        plt.plot(x[0][self.args['train_length']:self.args['train_length'] + self.args['forcast_window']],
                 tar[0], 'r--', linewidth=3)  # not missing data

    def plot_with_test(self, plot=False, test=False, add_num=0):
        if test:
            plt.title('test{}_length{}'.format(self.step + 1, self.args['train_length'] + add_num))
            if plot:
                plt.show()
            else:
                plt.savefig('train_result/test_val{}_length{}.jpg'.format(self.step + 1, self.args['train_length']))
        else:
            plt.title('train{}_length{}'.format(self.step + 1, self.args['train_length'] + add_num))
            if plot:
                plt.show()
            else:
                plt.savefig('train_result/train_val{}_length{}.jpg'.format(self.step + 1, self.args['train_length']))
