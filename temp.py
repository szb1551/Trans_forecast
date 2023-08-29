import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import deque
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import random_split
from DataLoader import all_dataset, SensorDataset4
from ProcessData import process_all_time

a = np.load('train_data/G_timeseries_map.npy')
a_ones = np.ones(a.shape)
a = a+a_ones
print(a)
a_log = np.log(a)
print(a_log)

# b = np.diff(a)
# print(b[1][1])
#
# pad_diff_data = np.pad(b, ((0,0),(0,0),(1,0)), mode='constant')
#
# pad_diff_data[:,:,0] = a[:,:,0]
#
# recovered_data = np.cumsum(pad_diff_data, axis=-1)
# print(recovered_data.shape)
# print(recovered_data[1][1])

# a = torch.zeros((2,3,4))
# b = torch.diff(a)
# print(b.shape)

