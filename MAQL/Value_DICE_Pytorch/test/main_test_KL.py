from value_dice import Algo_Param, Value_Dice
from util.count_frequency import collect_freqency
import torch
from model import NN_Paramters,DiscretePolicyNN, Discrete_Q_Function_NN
import numpy as np
from matplotlib import pyplot as plt
from util.q_learning_to_policy import Q_learner_Policy

policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= torch.device("cpu"))
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.0001)
algo_param = Algo_Param()
algo_param.gamma = 0.995


#target_policy = DiscretePolicyNN(policy_param, "1", "1")
#target_policy.load("train/1")

policy = Discrete_Q_Function_NN(policy_param, save_path="q", load_path="t_q")
policy.load("Q_models/q")
target_policy = Q_learner_Policy(policy, policy_param)

V = Value_Dice(target_policy, nu_param, algo_param)
V.nu_network.load("nu_1")
grid_size = 10

Buffer = torch.load("behavior_sub_1")
Target_Buffer = torch.load("target")

#++
# +torch.save(Buffer, "behavior_sub")
#torch.save(Target_Buffer, "target")

f_b = collect_freqency(Buffer, grid_size*grid_size, grid_size, algo_param.gamma )
f_t = collect_freqency(Target_Buffer, grid_size*grid_size, grid_size, algo_param.gamma )
ratio = f_t/f_b
log_ratio = np.log(ratio)

no_states = grid_size*grid_size
log_ratio_estimate = np.zeros([no_states,])
data = Buffer.sample(Buffer.no_data)
nu = V.get_log_state_action_density_ratio(data)



'''
nu = V.get_log_state_action_density_ratio(data)
states = data.state
for i in range(int(states.size/2)):

    x = states[i][0]
    y = states[i][1]
    log_ratio_estimate[x*grid_size + y] = nu[i][0]


print(log_ratio_estimate)
for i in range(no_states):
    if log_ratio_estimate[i] <= -4:
        print(log_ratio[i], ratio[i], log_ratio_estimate[i])
        
'''
w = torch.reshape(torch.Tensor(algo_param.gamma**data.time_step), shape=(Buffer.no_data, 1))
#print(w)
#print(grid_nu*w)
print(torch.sum(nu*w)/torch.sum(w))

grid_nu= [[0 for i in range(grid_size)] for j in range(grid_size)]
k = 0
for i in range(grid_size):
    for j in range(grid_size):
        grid_nu[i][j] = nu[k].item()
        k += 1
grid_nu = np.array(grid_nu)




fig, ax = plt.subplots()
im = ax.imshow(grid_nu)
plt.show()