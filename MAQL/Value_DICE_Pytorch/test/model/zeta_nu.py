import numpy as np
import torch
from model import NN_Paramters



param = NN_Paramters(state_dim=5, action_dim=4, hidden_layer_dim=[10], non_linearity=torch.tanh, device=torch.device("cuda"))

from model import Nu_NN, Zeta_NN


state = np.array([[0, 0, 0, 0, 0], [1,1,1,1,1]])
action =np.array([[0, 0, 0, 0], [1,1,1,1]])
inp =  np.concatenate((state, action), axis=1)


M_n = Nu_NN(param, "temp", "temp")
print(M_n.forward(state, action), M_n(state, action))

M_z = Zeta_NN(param, "temp", "temp", state_action=False)
print(M_n)
print(M_z)
print(M_z.forward(state, None))