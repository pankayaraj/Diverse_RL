import torch
from model import DiscretePolicyNN, NN_Paramters
from algo.dual_dice_desnity_estimation import Neural_Dual_Dice, Algo_Param
import  numpy as np

state = np.array([[0, 0, 0, 0, 0], [1,1,1,1,1]])
action =np.array([[0, 0, 0, 0], [1,1,1,1]])
weight =np.array([[1], [2]])

param = NN_Paramters(state_dim=5, action_dim=4, hidden_layer_dim=[10], non_linearity=None, device=torch.device("cuda"))
nu_param = NN_Paramters(state_dim=5, action_dim=4, hidden_layer_dim=[10], l_r=0.0001, non_linearity=None, device=torch.device("cuda"))
zeta_param = NN_Paramters(state_dim=5, action_dim=4, hidden_layer_dim=[10], lr=0.001, non_linearity=None, device=torch.device("cuda"))
a_param = Algo_Param()

T = DiscretePolicyNN(param, "temp", "temp")
N = Neural_Dual_Dice(param, param, a_param, T)

#N.compute_values(state, state, state, action, action, action)

N.train(state, state, state, action, action, action, weight)

gamma = 2
print(gamma**torch.Tensor(weight))

