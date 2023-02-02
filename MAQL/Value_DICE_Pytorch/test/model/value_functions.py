import numpy as np
import torch
from model import NN_Paramters



param = NN_Paramters(state_dim=5, action_dim=4, hidden_layer_dim=[10],non_linearity=None, device=torch.device("cuda"))

from model import Q_Function_NN, Value_Function_NN
M = Q_Function_NN(param, "temp", "temp")


state = np.array([[0, 0, 0, 0, 0], [1,1,1,1,1]])
action =np.array([[0, 0, 0, 0], [1,1,1,1]])

print(M.get_value(state, action))
print(M.get_value(state, action, format = "torch"))

M_v = Value_Function_NN(param, "temp", "temp")

print(M_v.get_value(state, format="torch"))