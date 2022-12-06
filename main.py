import time
import numpy as np
import torch
from DataLoader import SensorDataset4
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Transformer, Transformer2
from ProcessData import process_time
import datetime

train_length = 30
forcast_window = 7
csv_name = "Palo Alto"
criterion = torch.nn.MSELoss()
lr = 0.00001
epochs = 100
train_epoch_loss = []
Rp_best = 10
idx_example = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Dp(y_pred, y_true, q):  # softmax更新
    return max([q * (y_pred - y_true), (q - 1) * (y_pred - y_true)])


def Rp_num_den(y_preds, y_trues, q):  # RPloss
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


def save_loss(train_length):
    plt.figure(figsize=(10, 10))
    plt.plot(train_epoch_loss)
    plt.legend(['Train Loss'], fontsize=25)
    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("MSE Loss", fontsize=25)
    plt.grid()
    plt.savefig('train_result/loss_length{}.jpg'.format(train_length))
    np.savetxt('train_result/loss_length{}.txt'.format(train_length), train_epoch_loss, fmt='%6f')


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
            for p, o in zip(all_predictions.cpu().numpy().sum(axis=-1).sum(axis=-1), tar_out.numpy().sum(axis=-1).sum(axis=-1)):  # not missing data
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


def plot_models(model, dl, train_length, forcast_window, test=False, n_plots=5, plot=False):
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
                plt.plot(x[0][:src.shape[1]].cpu().detach().squeeze(-1).numpy(),
                         src[0].cpu().detach().squeeze(-1).numpy(), 'g--', linewidth=3)
                # plt.plot(x[0]+forcast_window,tar[0].cpu().detach().numpy(),'g--',linewidth=3)
                plt.plot(x[0][train_length:train_length + forcast_window], all_predictions[0, :].cpu().detach().numpy(),
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
                        plt.savefig('train_result/test_val{}_length{}.jpg'.format(step + 1, train_length))
                    else:
                        plt.savefig('train_result/train_val{}_length{}.jpg'.format(step + 1, train_length))


def plot_raster_models(model, dl, train_length, forcast_window, test=False, n_plots=5, plot=False):
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
                        plt.savefig('train_result/raster_test_val{}_length{}.jpg'.format(step + 1, train_length))
                    else:
                        plt.savefig('train_result/raster_train_val{}_length{}.jpg'.format(step + 1, train_length))

                # 单点预测图
                plt.figure(figsize=(10, 10))
                plt.xticks(x[0], data_list)
                plt.xticks(rotation=90)
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
                        plt.savefig('train_result/raster_test_val{}_length{}_point{}.jpg'.format(step + 1, train_length, '04'))
                    else:
                        plt.savefig('train_result/raster_train_val{}_length{}_point{}.jpg'.format(step + 1, train_length, '04'))


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

    # plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[0][119-10]) # missing data
    # plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[1][119-10]) # missing data
    # plt.plot(train_dataset[idx_example][0].numpy(),attn_layers[2][119-10]) # missing data

    plt.legend(["attn score in layer 1", "attn score in layer 2", "attn score in layer 3", "attn score in layer 4"])
    plt.title("Attn for t = 30+7")  # not missing data
    plt.savefig('train_result/raster_attn3_length{}.jpg'.format(train_length))

def process_data(day_map, time_list, training_length, forecast_window, time=False):
    matrix_lags = np.zeros(
        (day_map.shape[0] - (training_length + forecast_window), training_length + forecast_window))

    print('matrix_lags.shape:', matrix_lags.shape)  # [3407. 37]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = day_map[i:i + training_length + forecast_window]  # 批次 + num_lags+prediction_horizont天数

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


def process_raster_data(raster_map, time_list, training_length, forecast_window, nx=5, ny=5, time=False):
    matrix_lags = np.zeros(
        (raster_map.shape[0] - (training_length + forecast_window), training_length + forecast_window, nx, ny))

    print('matrix_lags.shape:', matrix_lags.shape)  # [3407. 37, 5, 5]

    i_train = matrix_lags.shape[0] - forecast_window  # 3400 # 2020-11-17 2019-9-6----2020-9-6 366+24+30+17=437
    i_test = matrix_lags.shape[0]
    # 3407-437 = 2970
    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = raster_map[i:i + training_length + forecast_window, :,
                         :]  # 批次 + num_lags+prediction_horizont天数

    # ---------------- Train/test split
    train_dataset = np.zeros((i_train, training_length + forecast_window, nx, ny))  # [3400, 37]
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


def select_model(num=0, train_length=train_length, forcast_window=forcast_window):
    if num == 0:
        model = Transformer(train_length, forcast_window).to(device)
    elif num == 1:
        model = Transformer2(train_length, forcast_window).to(device)
    else:
        raise ValueError("{}必须是一个数字0或1".format(num))
    return model


def train_day(Rp_best=10):
    day_map = np.load('train_data/normalized.npy')
    time_list = np.load('train_data/time_list.npy')
    day_map, time_list = get_date_data_PALO(day_map, time_list, year_begin=2011, year_end=2017, month_end=8,
                                            day_end=1)  # 筛选日期
    day_map[np.isnan(day_map)] = 0
    day_map[day_map == 0] = np.random.normal(np.zeros_like(day_map[day_map == 0]), 0.001)
    train_dataset, test_dataset, X_train_time, X_test_time = process_data(day_map, time_list, train_length,
                                                                          forcast_window, time=True)
    train_data = SensorDataset4(train_dataset, X_train_time, train_length, forcast_window)
    test_data = SensorDataset4(test_dataset, X_test_time, train_length, forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, feature_size]
    test_dl = DataLoader(test_data, batch_size=1)
    model = select_model(num=0, train_length=train_length, forcast_window=forcast_window)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(epochs)):
        start = time.time()
        train_loss = []

        l_t = train_epoch(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp = test_epoch(model, test_dl, train_length, forcast_window)

        if Rp_best > Rp:
            Rp_best = Rp
            torch.save(model, 'train_process/model_epoch{}_length{}_Rp_{:.3f}.h5'.format(e + 1, train_length, Rp))
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print("Epoch {}: Train loss: {:.6f} \t R_p={:.3f}\tcost_time={:.3f}s".format(e + 1,
                                                                                     np.mean(train_loss), Rp,
                                                                                     end - start))
    print('结束训练时间:', datetime.datetime.now())
    save_loss(train_length)
    torch.save(model, 'train_result/model_last_length{}.h5'.format(train_length))
    plot_models(model, train_dl, train_length, forcast_window)
    plot_models(model, test_dl, train_length, forcast_window, test=True)

    attn_layers = get_attn(model, test_data[idx_example][0].unsqueeze(0), test_data[idx_example][1].unsqueeze(0),
                           test_data[idx_example][2].unsqueeze(0))
    show_attn(test_data, attn_layers)


def train_raster_day(Rp_best):
    raster_map = np.load('train_data/raster_map.npy')  # [3444,5,5]
    time_list = np.load('train_data/time_list.npy')  # [3444,]
    max_ = np.max(raster_map, axis=0)
    np.save('train_data/max_raster.npy', max_)
    raster_map = raster_map / max_
    raster_map[np.isnan(raster_map)] = 0
    raster_map[raster_map == 0] = np.random.normal(np.zeros_like(raster_map[raster_map == 0]), 0.001)
    raster_map, time_list = get_date_data_PALO(raster_map, time_list, year_begin=2011, year_end=2017, month_end=8,
                                               day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_raster_data(raster_map, time_list, train_length,
                                                                                 forcast_window, time=True)
    train_data = SensorDataset4(train_dataset, X_train_time, train_length, forcast_window)
    test_data = SensorDataset4(test_dataset, X_test_time, train_length, forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)  # [batch_size, feature_size]
    test_dl = DataLoader(test_data, batch_size=1)
    model = select_model(num=1, train_length=train_length, forcast_window=forcast_window)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('开始训练时间:', datetime.datetime.now())
    for e, epoch in enumerate(range(epochs)):
        start = time.time()
        train_loss = []

        l_t = train_epoch(model, train_dl, optimizer=optimizer)
        train_loss.append(l_t)

        Rp = test_raster_epoch(model, test_dl, train_length, forcast_window)

        if Rp_best > Rp:
            Rp_best = Rp
            torch.save(model, 'train_process/model_epoch{}_length{}_Rp_{:.4f}.h5'.format(e + 1, train_length, Rp))
        train_epoch_loss.append(np.mean(train_loss))
        end = time.time()
        print("Epoch {}: Train loss: {:.6f} \t R_p={:.3f}\tcost_time={:.4f}s".format(e + 1,
                                                                                     np.mean(train_loss), Rp,
                                                                            end - start))
    print('结束训练时间:', datetime.datetime.now())
    save_loss(train_length)
    torch.save(model, 'train_result/model_last_length{}.h5'.format(train_length))
    plot_raster_models(model, train_dl, train_length, forcast_window)
    plot_raster_models(model, test_dl, train_length, forcast_window, test=True)

    attn_layers = get_raster_attn(model, test_data[idx_example][0].unsqueeze(0), test_data[idx_example][1].unsqueeze(0),
                           test_data[idx_example][2].unsqueeze(0))
    show_raster_attn(test_data, attn_layers)


if __name__ == '__main__':
    # train_day()
    train_raster_day(Rp_best)
