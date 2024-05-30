import matplotlib.pyplot as plt
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Dp(y_pred, y_true, q):  # softmax更新
    return max([q * (y_pred - y_true), (1 - q) * (y_true - y_pred)])


def diff_squre(y_pred, y_true):
    return (y_pred - y_true) ** 2


def mape_ins(y_pred, y_true):
    return abs((y_pred - y_true) / y_true)


def Rp_num_den(y_preds, y_trues, q):  # RP_loss
    numerator = np.sum([Dp(y_pred, y_true, q) for y_pred, y_true in zip(y_preds, y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator, denominator


def rmse_diff(predictions, targets):
    differences_squared = np.sum(diff_squre(y_pred, y_true) for y_pred, y_true in zip(predictions, targets))
    num = len(predictions)
    return differences_squared, num


def mape(predictions, targets):
    mape_sum = np.sum(mape_ins(y_pred, y_true) for y_pred, y_true in zip(predictions, targets))
    num = len(predictions)
    return mape_sum, num


def change_to_date(date_num):  # 更改数字日期到真实日期
    date_num = date_num.cpu().numpy()
    date_list = []
    for date in date_num:
        date_list.append(str(date)[:4] + '-' + str(date)[4:6] + '-' + str(date)[6:8])
    return date_list


def plot_Dalian_gcn_models(model, dl, train_length, forcast_window, test=False, n_plots=5, plot=False, max_=None):
    # 画出每个点的预测图片
    with torch.no_grad():
        model.eval()
        for step, (x, src, tar, times) in enumerate(dl):
            tar_in = src[:, :, -1].unsqueeze(-2)  # 取最后一个日子的输入当作transformer的起始输入
            tar_out = tar[:, :, 1:]
            next_input = tar_in.to(device)
            all_predictions = []
            for i in range(forcast_window):
                output = model(x.to(device).float(), src.to(device), next_input.to(device).float())
                if all_predictions == []:
                    all_predictions = output[:, :, -1].unsqueeze(-2)
                else:
                    all_predictions = torch.cat((all_predictions, output[:, :, -1].unsqueeze(-2).detach()), dim=-2)
                next_input = torch.cat((next_input.to(device), output[:, :, -1].unsqueeze(-2)), dim=-2)
            # x[batch, seq_len]
            # y[batch, seq_len, 2]
            # all_predictions [batch, Node, time ,2]
            # src[batch, Node, time ,2]
            if step >= n_plots:
                break
            with torch.no_grad():
                # 总预测图
                plt.figure(figsize=(10, 10))
                data_list = change_to_date(times[0])
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
                if train_length + forcast_window <= x[0].shape[0]:
                    x_forecast = x[0][train_length:train_length + forcast_window]
                else:
                    x_forecast = np.zeros(train_length + forcast_window + 1 - x[0].shape[0])
                    for i in range(train_length + forcast_window + 1 - x[0].shape[0]):
                        x_forecast[i] = x[0][-1] + i + 1

                if max_ is not None:
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             (src[0] * max_).sum(axis=0)[:, 0].cpu().detach().numpy(), 'g--', linewidth=3)
                    plt.plot(x_forecast,
                             (all_predictions[0].cpu() * max_).sum(axis=0)[:, 0].detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x_forecast,
                             (tar_out[0] * max_).sum(axis=0)[:, 0], 'r--', linewidth=3)  # not missing data
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             src[0].sum(axis=0)[:, 1].cpu().detach().numpy(), 'c--', linewidth=3)
                    plt.plot(x_forecast,
                             (all_predictions[0].cpu() * max_).sum(axis=0)[:, 1].cpu().detach().numpy(),
                             'm--', linewidth=3)
                    plt.plot(x_forecast,
                             (tar_out[0] * max_).sum(axis=0)[:, 1], 'y--', linewidth=3)  # not missing data
                else:
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             src[0].sum(axis=0)[:, 0].cpu().detach().numpy(), 'g--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x_forecast,
                             all_predictions[0].sum(axis=0)[:, 0].cpu().detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x_forecast,
                             tar_out[0].sum(axis=0)[:, 0], 'r--', linewidth=3)  # not missing data
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             src[0].sum(axis=0)[:, 1].cpu().detach().numpy(), 'c--', linewidth=3)
                    # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                    plt.plot(x_forecast,
                             all_predictions[0].sum(axis=0)[:, 1].cpu().detach().numpy(),
                             'm--', linewidth=3)
                    plt.plot(x_forecast,
                             tar_out[0].sum(axis=0)[:, 1], 'y--', linewidth=3)  # not missing data
                plt.xlabel("x", fontsize=20)
                plt.legend(["$ELEC_{his}$", "$ELEC_{predicted}$", "$ELEC_{true}$", "$HYD_{his}$", "$HYD_{predicted}$",
                            "$HYD_{true}$"])
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
                             (src[0] * max_)[4, :, 0].cpu().detach().numpy(), 'g--', linewidth=3)
                    plt.plot(x_forecast,
                             (all_predictions[0].cpu() * max_)[4, :, 0].detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x_forecast,
                             (tar_out[0] * max_)[4, :, 0], 'r--', linewidth=3)  # not missing data
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             (src[0] * max_)[4, :, 1].cpu().detach().numpy(), 'c--', linewidth=3)
                    plt.plot(x_forecast,
                             (all_predictions[0].cpu() * max_)[4, :, 1].detach().numpy(),
                             'm--', linewidth=3)
                    plt.plot(x_forecast,
                             (tar_out[0] * max_)[4, :, 1], 'y--', linewidth=3)  # not missing data
                else:
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             src[0, 4, :, 0].cpu().detach().numpy(), 'g--', linewidth=3)
                    plt.plot(x_forecast,
                             all_predictions[0, 4, :, 0].cpu().detach().numpy(),
                             'b--', linewidth=3)
                    plt.plot(x_forecast,
                             tar_out[0, 4, :, 0], 'r--', linewidth=3)  # not missing data
                    plt.plot(x[0][:src.shape[2]].cpu().detach().numpy(),
                             src[0, 4, :, 1].cpu().detach().numpy(), 'c--', linewidth=3)
                    plt.plot(x_forecast,
                             all_predictions[0, 4, :, 1].cpu().detach().numpy(),
                             'm--', linewidth=3)
                    plt.plot(x_forecast,
                             tar_out[0, 4, :, 1], 'y--', linewidth=3)  # not missing data
                plt.xlabel("x", fontsize=20)
                plt.legend(["$ELEC_{his}$", "$ELEC_{predicted}$", "$ELEC_{true}$", "$HYD_{his}$", "$HYD_{predicted}$",
                            "$HYD_{true}$"])
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
