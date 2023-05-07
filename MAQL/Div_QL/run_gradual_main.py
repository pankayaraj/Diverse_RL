import math

import torch
import numpy as np

from DivQL import DivQL
from DivQL_Gradual import DivQL_Gradual

from q_learning import Q_learning

from model import NN_Paramters
from parameters import Algo_Param, Save_Paths, Load_Paths

from env.gridworld.environments import GridWalk
from env.gridworld.enviornment2 import GridWalk2
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk

from util.q_learning_to_policy import Q_learner_Policy
from util.collect_trajectories import collect_data
from util.density_ratio import get_policy_ratio
from memory import Transition_tuple, Replay_Memory
from math import log


device = torch.device("cpu")
policy_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device= device)
nu_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=device, l_r=0.0001)
q_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=device, l_r=0.05)
algo_param = Algo_Param(hard_update_interval=1)
algo_param.gamma = 0.9
num_z = 15
Z = [1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14 ]

grid_size = 10
env = GridWalk(grid_size, False)
env2 = GridWalk(grid_size, False)
env_tar = GridWalk(grid_size, False)

#behaviour_Q  = Q_learning(env, q_param, algo_param)
#behaviour_policy = Q_learner_Policy(behaviour_Q.Q, q_param)
#behaviour_policy.Q.load("q")

update_interval = 100
save_interval = 1000
eval_interval = 1000
max_episodes = 100
#max_episodes = 20

inital_log_buffer = torch.load("gradual_models/10x10/q/mem0")

from DivQL_main import DivQL
M  = DivQL(env, inital_log_buffer=inital_log_buffer, q_nn_param=q_param, nu_pram=nu_param,
           algo_param=algo_param, num_z=num_z)


for z in Z:

    data = [[0 for i in range(10)] for j in range(10)]

    for i in range(10):
        for j in range(10):
            p1 = 1 / 100
            p2 = M.prev_O_M[i][j]
            data[i][j] =  (p1 ) / (p2 + 0.01)
    print(np.array(data))

    state = M.initalize()
    print(M.current_index )

    print("Z " + str(z))
    for i in range(10000):

        M.train()
        state = M.step(state)

        if i%update_interval == 0:
            M.hard_update()
        #if i%500==0:
        #    torch.save(M.occupancy_measure, "gradual_models/10x10/3/occ_2_" + str(i))
        #    torch.save(M.occupancy_measure_prev, "gradual_models/10x10/3/occ_1_" + str(i))
        if i % save_interval == 0:
            print(M.current_index, z)
            print("saving")
            #M.save(z-1,"gradual_models/10x10/3/q" + str(z), "gradual_models/10x10/3/target_q" + str(z), "gradual_models/10x10/3/nu" + str(z) + "_1", "gradual_models/10x10/3/nu" + str(z) + "_2")
            M.save_main("gradual_models/10x10/tabular_DivQL/q", "gradual_models/10x10/tabular_DivQL/target_q")
        if i % eval_interval == 0:

            OM_P = M.prev_O_M
            S = []
            A = []
            s = env2.reset()
            i_s = s
            rew = 0
            a_r = 0
            for j in range(max_episodes):
                S.append(list(s))
                a = M.get_action(s, z-1)
                A.append(a)
                s, r, d, _ = env2.step(a)
                rew += r
                #p1 = OM[s[1]][s[0]]
                p1 = 1/100
                p2 = OM_P[s[1]][s[0]]
                ra = (p1 ) / (p2 + 0.01)
                l_p = math.log((p1 ) / (p2 + 0.01))

                a_r += ra
                #a_r += l_p

                #print(s, r, ra)
                if j == max_episodes - 1:
                    d = True
                if d == True:
                    break
            print("for z = " + str(z) + " reward at itr " + str(i) + " = " + str(rew) + ", " + str(rew+algo_param.alpha*a_r) + " at state: " + str(s) + " starting from: " + str(i_s) + " eps " + str(M.epsilon))

            print(S)
            print(A)
        # Q.memory = torch.load("mem")
        #torch.save(M.log_ratio_memory, "gradual_models/3/mem" + str(z))

    M.change_z(z)
