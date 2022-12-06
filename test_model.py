import numpy as np
import torch
import datetime
from DataLoader import SensorDataset4
from main import plot_models, process_data, get_date_data_PALO
from main import plot_raster_models, process_raster_data
from torch.utils.data import DataLoader

train_length = 30
forcast_window = 7
print(torch.__version__)

def test_day_model(select_date=False):
    print(datetime.datetime.now())
    # model = Transformer(train_length, forcast_window).cuda()
    # print(model)
    # test_model = torch.load('server_result/model_epoch6_length30_Rp_0.321.h5')
    model = torch.load('train_process/model_last.h5')
    day_map = np.load('train_data/normalized.npy')
    time_list = np.load('train_data/time_list.npy')
    if select_date:
        raster_map, time_list = get_date_data_PALO(day_map, time_list, year_begin=2011, year_end=2017, month_end=8,
                                                   day_end=1)
    train_dataset, test_dataset, X_train_time, X_test_time = process_data(day_map, time_list, train_length,
                                                                          forcast_window, time=True)
    train_data = SensorDataset4(train_dataset, X_train_time, train_length=train_length, forecast_window=forcast_window)
    test_data = SensorDataset4(test_dataset, X_test_time, train_length=train_length, forecast_window=forcast_window)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=1)
    plot_models(model, test_dl, train_length, forcast_window, test=True, plot=True)
    print(datetime.datetime.now())

def test_raster_model(select_date=False):
    print(datetime.datetime.now())
    # model = Transformer(train_length, forcast_window).cuda()
    # print(model)
    # test_model = torch.load('server_result/model_epoch6_length30_Rp_0.321.h5')
    model = torch.load('train_process/model_epoch86_length30_Rp_0.066.h5')
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
    plot_raster_models(model, train_dl, train_length, forcast_window, plot=True)
    plot_raster_models(model, test_dl, train_length, forcast_window, test=True, plot=True)
    print(datetime.datetime.now())


if __name__ == '__main__':
    test_raster_model(select_date=True)
