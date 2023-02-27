from util.collect_trajectories import collect_data
from model import DiscretePolicyNN, NN_Paramters
from env.gridworld.environments import GridWalk
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk
from util.density_ratio import get_policy_ratio
from env.gridworld.sub_optimal_policy import Sub_Optim_Policy_Gridwalk
from value_dice import Algo_Param, Value_Dice
import torch
import numpy as np
import os
from util.count_frequency import collect_freqency

policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= torch.device("cpu"))
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.0001)
algo_param = Algo_Param()
algo_param.gamma = 0.995


grid_size = 10
env = GridWalk(grid_size, False)


#behaviour_policy = Optim_Policy_Gridwalk(env, action_dim=5, eps_explore=0.2)
target_policy = DiscretePolicyNN(policy_param, "1", "1")
target_policy.load("train/1")


V = Value_Dice(target_policy, nu_param, algo_param)
#Buffer, _ , _ = collect_data(env, behaviour_policy, 1000, 10)
#Target_Buffer, _ , _ = collect_data(env, target_policy, 1000, 10)

Buffer = torch.load("behaviour_buffer")

Target_Buffer = torch.load("target")


#torch.save(Buffer, "behavior_sub_1")
#torch.save(Target_Buffer, "target")

#f_b = collect_freqency(Buffer, grid_size*grid_size, grid_size, algo_param.gamma )
#f_t = collect_freqency(Target_Buffer, grid_size*grid_size, grid_size, algo_param.gamma )
#ratio = f_t/f_b
#print(ratio)
no_iterations = 30000
#V.nu_network.load("nu_1")

for i in range(no_iterations):

    data = Buffer.sample(400)
    V.train_KL(data=data)

    if i % 1000 == 0:
        print(i)
        B = Buffer.sample(1)
        s = B.state
        a = B.action
        n_s = B.next_state

        a_n  = 0
        for j in range(nu_param.action_dim):
            if a[0][j] == 1:
                a_n = j


        print(B.state, B.action)
        #s = np.array([[9, 9]])
        #print(V.get_log_state_action_density_ratio(B),
        # np.log(ratio[s[0][0]*grid_size+ s[0][1]]*get_policy_ratio(target_policy, behaviour_policy, s, a_n)) )

        print(V.debug())
        V.nu_network.save("nu_1")