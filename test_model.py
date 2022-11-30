import numpy as np
import torch
import datetime
from DataLoader import SensorDataset4
from main import plot_models, process_data
from torch.utils.data import DataLoader

train_length = 30
forcast_window = 7
print(torch.__version__)
if __name__ == '__main__':
    print(datetime.datetime.now())
    # model = Transformer(train_length, forcast_window).cuda()
    # print(model)
    # test_model = torch.load('server_result/model_epoch6_length30_Rp_0.321.h5')
    model = torch.load('train_process/model_last.h5')
    day_map = np.load('train_data/normalized.npy')
    time_list = np.load('train_data/time_list.npy')
    train_dataset, test_dataset, X_train_time, X_test_time = process_data(day_map, time_list, train_length,
                                                                          forcast_window, time=True)
    train_data = SensorDataset4(train_dataset, X_train_time, train_length=train_length, forecast_window=forcast_window)
    test_data = SensorDataset4(test_dataset, X_test_time, train_length=train_length, forecast_window=forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1)
    plot_models(model, test_dl, train_length, forcast_window, test=True, plot=True)
    print(datetime.datetime.now())
