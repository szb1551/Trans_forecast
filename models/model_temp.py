from models.model_utils import get_model_args, select_model
from model import Transformer, Transformer2, Transformer3, baseline_conv, Transformer4
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_length = 30
forcast_window = 7
batch_size = 32
A = torch.zeros(150, 150)
model_name = 'Transformer_gcn'
# x = torch.zeros(3, 30, 5, 5)
x = torch.zeros((batch_size, train_length+forcast_window))
# y = torch.zeros(3, 30, 1)  # 代表汽车保有量
args = get_model_args(model_name, train_length, forcast_window, x, A)
model = select_model(model_name, args)
print(model)

# src = torch.zeros((batch_size, 150, train_length,2))  # .to(device)
# tar = torch.zeros((batch_size, 150, forcast_window,2))
src = torch.zeros((batch_size, len(A), train_length))  # .to(device)
tar = torch.zeros((batch_size, len(A), forcast_window))
print(model(x, src, tar).shape)
