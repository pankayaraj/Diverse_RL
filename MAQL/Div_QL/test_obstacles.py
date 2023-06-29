import random

import torch
from DivQL_main import DivQL
from env.gridworld.environments_obs_1 import GridWalk
from model import NN_Paramters
from parameters import Algo_Param, Save_Paths, Load_Paths


q_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.05)
nu_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.0001)
algo_param = Algo_Param()
algo_param.gamma = 0.9


import numpy as np
import matplotlib.pyplot as plt

env = GridWalk(10, False)

inital_log_buffer = torch.load("gradual_models/10x10/q/mem0")


M  = DivQL(env, inital_log_buffer=inital_log_buffer, q_nn_param=q_param, nu_pram=nu_param,
           algo_param=algo_param, num_z=15)


M.load_main( "gradual_models/10x10/DivQL/q1", "gradual_models/10x10/DivQL/target_q1" )

z_no = 9


total = 200
rate = []


for i in range(10):
    sucess = 0
    for i in range(total):
        z = random.randint(0,z_no)

        data = [[0 for i in range(10)] for j in range(10)]

        s = env.reset()
        i_s = s
        rew = 0
        data[s[1]][s[0]] += 1
        S = []

        for j in range(100):
            a = M.get_action(s,z)
            s, r, d, _ = env.step(a)
            S.append(s)

            rew += r
            data[s[1]][s[0]] += 1

            if j == 100 - 1:
                d = True
            if d == True:
                break
        if rew > 0:
            sucess += 1



    rate.append(100*sucess/total)


print(rate)
print(np.mean(rate))
print(np.std(rate))