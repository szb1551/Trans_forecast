from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class SensorDataset(Dataset):
    def __init__(self, dataset, train_length, forecast_window):
        # csv_file = os.path.join(root_dir, csv_name)  # 打开csv文件
        self.y = torch.tensor(dataset, dtype=torch.float32)
        self.x = torch.cat(self.y.shape[0] * [torch.arange(0, train_length + forecast_window).unsqueeze(0)])
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.masks = self._generate_square_subsequent_mask(train_length, forecast_window)
        # self.x = self.x / torch.max(self.x)

        print('x.shape:', self.x.shape)

    def __len__(self):
        # print(len(self.df.groupby(by=["Start"]))) # 3400|37
        return len(self.y)

    def __getitem__(self, idx):
        sample = (self.x[idx, :],
                  self.y[idx, :],
                  self.masks)
        return sample

    def _generate_square_subsequent_mask(self, train_length, forcast_window):
        mask = torch.zeros(train_length + forcast_window, train_length + forcast_window)
        for i in range(0, train_length):
            mask[i, train_length:] = 1
        for i in range(train_length, train_length + forcast_window):
            mask[i, i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # .masked_fill(mask == 1, float(0.0))
        return mask


class SensorDataset3(Dataset):
    def __init__(self, dataset, time_dataset, train_length, forecast_window):
        # csv_file = os.path.join(root_dir, csv_name)  # 打开csv文件
        self.y = torch.tensor(dataset, dtype=torch.float32)
        self.x = torch.cat(self.y.shape[0] * [torch.arange(0, train_length + forecast_window).unsqueeze(0)])
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.time = time_dataset
        self.masks = self._generate_square_subsequent_mask(train_length, forecast_window)
        # self.x = self.x / torch.max(self.x)

        print('x.shape:', self.x.shape)

    def __len__(self):
        # print(len(self.df.groupby(by=["Start"]))) # 3400|37
        return len(self.y)

    def __getitem__(self, idx):
        sample = [self.x[idx, :],
                  self.y[idx, :],
                  self.time[idx, :]]
        return sample

    def _generate_square_subsequent_mask(self, train_length, forcast_window):
        mask = torch.zeros(train_length + forcast_window, train_length + forcast_window)
        for i in range(0, train_length):
            mask[i, train_length:] = 1
        for i in range(train_length, train_length + forcast_window):
            mask[i, i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # .masked_fill(mask == 1, float(0.0))
        return mask


class SensorDataset2(Dataset):  # 得到其他数据31
    def __init__(self, dataset, time_lists, train_length, forecast_window, num_more=1):
        # csv_file = os.path.join(root_dir, csv_name)  # 打开csv文件
        self.src = torch.tensor(dataset[:, :train_length + num_more], dtype=torch.float32)
        self.tar = torch.tensor(dataset[:, train_length:], dtype=torch.float32)
        self.x = torch.cat(self.src.shape[0] * [torch.arange(0, train_length + forecast_window).unsqueeze(0)])
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.time_list = time_lists
        print(self.src.shape)
        print(self.tar.shape)
        print(self.x.shape)
        print(self.time_list.shape)

    def __len__(self):
        # print(len(self.df.groupby(by=["Start"]))) # 3400|37
        return len(self.tar)

    def __getitem__(self, idx):
        sample = (self.x[idx, :],
                  self.src[idx, :],
                  self.tar[idx, :],
                  self.time_list[idx, :])
        return sample


class SensorDataset4(Dataset):  # 得到其他数据31 得到decoder数据 raster使用
    def __init__(self, dataset, time_lists, train_length, forecast_window, baseline=False):
        # csv_file = os.path.join(root_dir, csv_name)  # 打开csv文件
        self.src = torch.tensor(dataset[:, :train_length], dtype=torch.float32)
        if baseline:
            self.tar = torch.tensor(dataset[:, train_length:], dtype=torch.float32)  # [3400, 7, 5, 5]
        else:
            self.tar = torch.tensor(dataset[:, train_length - 1:], dtype=torch.float32)
        self.x = torch.cat(self.src.shape[0] * [torch.arange(0, train_length + forecast_window).unsqueeze(0)])
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.time_list = time_lists
        print('src:', self.src.shape)  # [3400, 30, ...]
        print('tar:', self.tar.shape)  # [3400, 7, ...]
        print('x:', self.x.shape)
        print('time_list:', self.time_list.shape)

    def __len__(self):
        # print(len(self.df.groupby(by=["Start"]))) # 3400|37
        return len(self.tar)

    def __getitem__(self, idx):
        sample = (self.x[idx, :],
                  self.src[idx, :],
                  self.tar[idx, :],
                  self.time_list[idx, :])
        return sample


class SensorDataset_baseline_conv(Dataset):  # 得baseline raster使用
    def __init__(self, dataset, time_lists, train_length, forecast_window):
        # dataset [3400, 37, 5, 5]
        self.src = torch.tensor(dataset[:, :train_length], dtype=torch.float32)  # [3400, 30, 5, 5]
        self.tar = torch.tensor(dataset[:, train_length:], dtype=torch.float32)  # [3400, 7, 5, 5]
        self.x = torch.cat(self.src.shape[0] * [torch.arange(0, train_length + forecast_window).unsqueeze(0)])
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.time_list = time_lists
        print('src:', self.src.shape)  # [3400, 30, ...]
        print('tar:', self.tar.shape)  # [3400, 7, ...]
        print('x:', self.x.shape)
        print('time_list:', self.time_list.shape)

    def __len__(self):
        # print(len(self.df.groupby(by=["Start"]))) # 3400|37
        return len(self.tar)

    def __getitem__(self, idx):
        sample = (self.x[idx, :],
                  self.src[idx, :],
                  self.tar[idx, :],
                  self.time_list[idx, :])
        return sample


class SensorDataset_GCN(Dataset):  # GCN使用
    def __init__(self, dataset, time_lists, train_length, forecast_window, baseline=False):
        # csv_file = os.path.join(root_dir, csv_name)  # 打开csv文件
        self.src = torch.tensor(dataset[:, :, :train_length], dtype=torch.float32)
        if baseline:
            self.tar = torch.tensor(dataset[:, :, train_length:], dtype=torch.float32)
        else:
            self.tar = torch.tensor(dataset[:, :, train_length - 1:], dtype=torch.float32)
        # self.x = torch.cat(self.src.shape[0] * [torch.arange(0, train_length + forecast_window).unsqueeze(0)]) # 只会被当作位置编码
        self.x = torch.zeros((self.src.shape[0], train_length+forecast_window))
        for i in range(self.src.shape[0]):
            self.x[i] = torch.arange(i, i+train_length+forecast_window)
        # self.x = torch.tensor(self.x, dtype=torch.float32)
        self.time_list = time_lists
        print('src:', self.src.shape)  # [3400, N, 30, ...]
        print('tar:', self.tar.shape)  # [3400, N, 8, ...]
        print('x:', self.x.shape)  # [3400, 37]
        print('time_list:', self.time_list.shape)  # [3400, 37]

    def __len__(self):
        # print(len(self.df.groupby(by=["Start"]))) # 3400|37
        return len(self.tar)

    def __getitem__(self, idx):
        sample = (self.x[idx, :],
                  self.src[idx, :],
                  self.tar[idx, :],
                  self.time_list[idx, :])
        return sample


class SensorDataset_GCN_diff(Dataset):  # GCN使用
    def __init__(self, dataset, time_lists, train_length, forecast_window, baseline=False):
        # csv_file = os.path.join(root_dir, csv_name)  # 打开csv文件
        self.src = torch.tensor(dataset[:, :, :train_length], dtype=torch.float32)
        if baseline:
            self.tar = torch.tensor(dataset[:, :, train_length:], dtype=torch.float32)
        else:
            self.tar = torch.tensor(dataset[:, :, train_length - 1:], dtype=torch.float32)
        self.src_diff = torch.tensor(np.diff(self.src))
        self.tar_diff = torch.tensor(np.diff(self.tar))
        self.x = torch.cat(self.src.shape[0] * [torch.arange(0, train_length + forecast_window).unsqueeze(0)])
        # self.x = torch.tensor(self.x, dtype=torch.float32)
        self.time_list = time_lists
        print('src_diff:', self.src_diff.shape)  # [3400, N, 30, ...]
        print('tar_diff:', self.tar_diff.shape)  # [3400, N, 7, ...]
        print('x:', self.x.shape)  # [3400, 37]
        print('time_list:', self.time_list.shape)  # [3400, 37]

    def __len__(self):
        # print(len(self.df.groupby(by=["Start"]))) # 3400|37
        return len(self.tar)

    def __getitem__(self, idx):
        sample = (self.x[idx, :],
                  self.src_diff[idx, :],
                  self.tar_diff[idx, :],
                  self.time_list[idx, :],
                  self.src[idx, :, 0].unsqueeze(-1),
                  self.tar[idx, :, 1].unsqueeze(-1)
                  )
        return sample


class all_dataset(Dataset):
    def __init__(self, dataset, time_list, train_length=30, forecast_window=7):
        self.dataset = dataset
        self.time_list = time_list

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        sample = (self.dataset[idx, :],
                  self.time_list[idx, :])
        return sample


class time_series_decoder_paper(Dataset):
    """synthetic time series dataset from section 5.1"""

    def __init__(self, t0=96, N=4500, transform=None):
        """
        Args:
            t0: previous t0 data points to predict from
            N: number of data points
            transform: any transformations to be applied to time series
        """
        self.t0 = t0
        self.N = N
        self.transform = None

        # time points
        self.x = torch.cat(N * [torch.arange(0, t0 + 24).type(torch.float).unsqueeze(0)])

        # sinuisoidal signal
        A1, A2, A3 = 60 * torch.rand(3, N)
        A4 = torch.max(A1, A2)
        self.fx = torch.cat([A1.unsqueeze(1) * torch.sin(np.pi * self.x[0, 0:12] / 6) + 72,
                             A2.unsqueeze(1) * torch.sin(np.pi * self.x[0, 12:24] / 6) + 72,
                             A3.unsqueeze(1) * torch.sin(np.pi * self.x[0, 24:t0] / 6) + 72,
                             A4.unsqueeze(1) * torch.sin(np.pi * self.x[0, t0:t0 + 24] / 12) + 72], 1)

        # add noise
        self.fx = self.fx + torch.randn(self.fx.shape)

        self.masks = self._generate_square_subsequent_mask(t0)

        # print out shapes to confirm desired output
        print("x: {}*{}".format(*list(self.x.shape)),
              "fx: {}*{}".format(*list(self.fx.shape)))

    def __len__(self):
        return len(self.fx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.x[idx, :],
                  self.fx[idx, :],
                  self.masks)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _generate_square_subsequent_mask(self, t0):
        mask = torch.zeros(t0 + 24, t0 + 24)
        for i in range(0, t0):
            mask[i, t0:] = 1
        for i in range(t0, t0 + 24):
            mask[i, i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # .masked_fill(mask == 1, float(0.0))
        return mask


class time_series_decoder_paper2(Dataset):
    """synthetic time series dataset from section 5.1"""

    def __init__(self, t0=96, t1=7, N=4500, transform=None):
        """
        Args:
            t0: previous t0 data points to predict from
            N: number of data points
            transform: any transformations to be applied to time series
        """
        self.t0 = t0
        self.N = N
        self.transform = None

        # time points
        self.x = torch.cat(N * [torch.arange(0, t0).type(torch.float).unsqueeze(0)])

        # sinuisoidal signal
        A1, A2, A3 = 60 * torch.rand(3, N)
        A4 = torch.max(A1, A2)
        self.fx = torch.cat([A1.unsqueeze(1) * torch.sin(np.pi * self.x[0, 0:12] / 6) + 72,
                             A2.unsqueeze(1) * torch.sin(np.pi * self.x[0, 12:24] / 6) + 72,
                             A3.unsqueeze(1) * torch.sin(np.pi * self.x[0, 24:t0] / 6) + 72,
                             A4.unsqueeze(1) * torch.sin(np.pi * torch.arange(t0, t0 + t1) / 12) + 72], 1)

        # add noise
        self.fx = self.fx + torch.randn(self.fx.shape)
        self.train_fx = self.fx[:, :t0]
        self.tar_fx = self.fx[:, t0:]

        # print out shapes to confirm desired output
        print("x: {}*{}".format(*list(self.x.shape)),
              "fx: {}*{}".format(*list(self.fx.shape)),
              "train_fx: {}*{}".format(*list(self.train_fx.shape)),
              "tar_fx: {}*{}".format(*list(self.tar_fx.shape)))

    def __len__(self):
        return len(self.fx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.x[idx, :],
                  self.train_fx[idx, :],
                  self.tar_fx[idx, :])

        if self.transform:
            sample = self.transform(sample)

        return sample
