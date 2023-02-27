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
from memory import Replay_Memory

from q_learning.q_learning import Q_learning
from util.q_learning_to_policy import Q_learner_Policy


device = torch.device("cpu")

policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= torch.device("cpu"))
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.0001)
q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=device, l_r=0.05)
algo_param = Algo_Param()
algo_param.gamma = 0.995


grid_size = 10
env = GridWalk(grid_size, False)
#behaviour_policy = Optim_Policy_Gridwalk(env, action_dim=5, eps_explore=0.2)
behaviour_Q  = Q_learning(env, q_param, algo_param)
behaviour_policy = Q_learner_Policy(behaviour_Q.Q, q_param)
behaviour_policy.Q.load("q")
max_episodes = 100
