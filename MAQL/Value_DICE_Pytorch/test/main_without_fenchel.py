from util.collect_trajectories import collect_data
from model import DiscretePolicyNN, NN_Paramters
from env.gridworld.environments import GridWalk
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk
from env.gridworld.sub_optimal_policy import Sub_Optim_Policy_Gridwalk
from algo.dual_dice_desnity_estimation_without_fenchel import Neural_Dual_Dice, Algo_Param
import torch
import numpy as np
import os

policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10], non_linearity=None, device= torch.device("cpu"))
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10], non_linearity=None, device=torch.device("cpu"), l_r=0.0001)

algo_param = Algo_Param()


env = GridWalk(10, False)
behaviour_policy = Optim_Policy_Gridwalk(env, action_dim=5)
target_policy = DiscretePolicyNN(policy_param, "1", "1")
target_policy.load("train/1")


N  = Neural_Dual_Dice(nu_param, algo_param, target_policy, fucntion_exponent=2.0)

Buffer, _ , _ = collect_data(env, behaviour_policy, 1000, 10)

no_iterations = 5000

for i in range(no_iterations):
    data = Buffer.sample(size=400)
    N.compute(data, target_policy, )

    B = Buffer.sample(1)
    s = B.state
    a = B.action
    n_s = B.next_state
    s = np.array([[9, 9]])
    print(N.get_state_action_density_ratio(s, a, n_s), a)