import  numpy as np
import  matplotlib.pyplot as plt
from util.collect_trajectories import collect_data
from model import DiscretePolicyNN, NN_Paramters, Discrete_Q_Function_NN
from env.gridworld.environments import GridWalk
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk
from util.density_ratio import get_policy_ratio
from env.gridworld.sub_optimal_policy import Sub_Optim_Policy_Gridwalk
from value_dice import Algo_Param, Value_Dice
import torch
import numpy as np
import os
from util.count_frequency import collect_freqency
from util.q_learning_to_policy import Q_learner_Policy
from value_dice_eval_buffer import Value_Dice_Eval_Buffer

import sys
print(sys.path)

policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= torch.device("cpu"))
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.0001)
algo_param = Algo_Param()
algo_param.gamma = 0.995



grid_size = 10
env = GridWalk(grid_size, False)

V = Value_Dice_Eval_Buffer(nu_param, algo_param)
V.nu_network.load("nu_eval_buff/value_dice/nu_4")
class Data():
    def __init__(self):

        self.state = None
        self.action = None

data = [[0 for i in range(10)] for j in range(10)]
data_l = [[0 for i in range(10)] for j in range(10)]
for i in range(10):

    for j in range(10):
        x = 0
        s = np.array([j, i])

        a1 = np.array([0, 0, 0, 0, 1])
        a2 = np.array([0, 0, 0, 1, 0])
        a3 = np.array([0, 0, 1, 0, 0])
        a4 = np.array([0, 1, 0, 0, 0])
        a5 = np.array([1, 0, 0, 0, 0])

        D = Data()
        D.state = s
        D.action = a1
        x += V.get_log_state_action_density_ratio(D).item()
        D.action = a2
        x += V.get_log_state_action_density_ratio(D).item()
        D.action = a3
        x += V.get_log_state_action_density_ratio(D).item()
        D.action = a4
        x += V.get_log_state_action_density_ratio(D).item()
        D.action = a5
        x += V.get_log_state_action_density_ratio(D).item()



        data[i][j] = x
        data_l[i][j] = x

data = np.array(data)
data_l = np.array(data_l)

print(data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, data, cmap='RdYlBu_r')
plt.show()
