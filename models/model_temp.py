from models.model_utils import get_model_args, select_model
from model import Transformer, Transformer2, Transformer3, baseline_conv, Transformer4
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_length = 7
forcast_window = 1
A = torch.zeros(150, 150)
model_name = 'Transformer_gcn_Dalian'
# x = torch.zeros(3, 30, 5, 5)
x = torch.zeros((3, 8))
# y = torch.zeros(3, 30, 1)  # 代表汽车保有量
args = get_model_args(model_name, train_length, forcast_window, x, A)
model = select_model(model_name, args)
print(model)

src = torch.zeros((3, 150, 7,2))  # .to(device)
tar = torch.zeros((3, 150, 1,2))
print(model(x, src, tar).shape)
