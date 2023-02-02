import torch
import numpy as np


from .maql import MAQL
from log_ratio.log_ratio import Log_Ratio
from q_learning.q_learning import Q_learning
from util.q_learning_to_policy import Q_learner_Policy

from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths

policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= torch.device("cpu"))
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.0001)
q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.05)
algo_param = Algo_Param()
algo_param.gamma = 0.9

def get_maql(env):
    return MAQL(env, q_param, nu_param, algo_param)

def get_q_learning(env):
    return Q_learning(env, q_param, algo_param)

def get_log_ratio(env):
    return Log_Ratio(nu_param, algo_param)

def get_q_learning_policy(q_learning):
    return Q_learner_Policy(q_learning.Q, q_learning.q_nn_param)