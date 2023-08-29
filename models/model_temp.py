from models.model_utils import get_model_args, select_model
from model import Transformer, Transformer2, Transformer3, baseline_conv, Transformer4
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_length = 30
forcast_window = 7
A = torch.zeros(47, 47)
model_name = 'conv_lstm2'
x = torch.zeros(3, 30, 5, 5)
args = get_model_args(model_name, train_length, forcast_window, x, A)
model = select_model(model_name, args)
print(model)
# x = torch.zeros((3,37))
# src = torch.zeros((3,47, 29)).to(device)
# tar = torch.zeros((3,47, 6)).to(device)
print(model(x).shape)
