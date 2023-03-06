from value_dice import Algo_Param, Value_Dice
from util.count_frequency import collect_freqency
import torch
from model import NN_Paramters, DiscretePolicyNN, Discrete_Q_Function_NN
import numpy as np
from matplotlib import pyplot as plt
from util.q_learning_to_policy import Q_learner_Policy
from value_dice_eval_buffer import Value_Dice_Eval_Buffer

policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh,
                            device=torch.device("cpu"))
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh,
                        device=torch.device("cpu"), l_r=0.0001)
algo_param = Algo_Param()
algo_param.gamma = 0.995




V = Value_Dice_Eval_Buffer( nu_param, algo_param)
V.nu_network.load("nu_eval_buff/value_dice/nu_2")
grid_size = 10

Buffer = torch.load("behavior_sub_1")



no_states = grid_size * grid_size
log_ratio_estimate = np.zeros([no_states, ])
data = Buffer.sample(Buffer.no_data)
nu = V.get_log_state_action_density_ratio(data)


w = torch.reshape(torch.Tensor(algo_param.gamma ** data.time_step), shape=(Buffer.no_data, 1))
# print(w)
# print(grid_nu*w)
print((torch.sum(nu * w) / torch.sum(w)).item())

grid_nu = [[0 for i in range(grid_size)] for j in range(grid_size)]
k = 0
for i in range(grid_size):
    for j in range(grid_size):
        grid_nu[i][j] = nu[k].item()
        k += 1
grid_nu = np.array(grid_nu)

fig, ax = plt.subplots()
im = ax.imshow(grid_nu)
plt.show()