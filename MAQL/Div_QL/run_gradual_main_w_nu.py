import math

import torch
import numpy as np



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
nu_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[6,6], non_linearity=torch.tanh, device=device, l_r=0.0001)
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

from DivQL_main_w_Nu import DivQL, Data
import numpy as np
import matplotlib.pyplot as plt



M  = DivQL(env, inital_log_buffer=inital_log_buffer, q_nn_param=q_param, nu_pram=nu_param,
           algo_param=algo_param, num_z=num_z)

data = [[0 for i in range(10)] for j in range(10)]


print(np.array(data))
for z in Z:
    M.train_ratios(30000)

    data = [[0 for i in range(10)] for j in range(10)]
    data_l = [[0 for i in range(10)] for j in range(10)]

    for i in range(10):
        for j in range(10):

            s = np.array([j, i])
            x = 0




            a1 = np.array([0, 0, 0, 1])
            a2 = np.array([0, 0, 1, 0])
            a3 = np.array([0, 1, 0, 0])
            a4 = np.array([1, 0, 0, 0])


            action = torch.Tensor(a1)
            state = torch.Tensor(s)


            D = Data()
            D.state = s
            D.action = a1

            x += M.log_ratio[z-1].get_log_state_action_density_ratio(D, None).item()
            D.action = a2
            x += M.log_ratio[z-1].get_log_state_action_density_ratio(D, None).item()
            D.action = a3
            x += M.log_ratio[z-1].get_log_state_action_density_ratio(D, None).item()
            D.action = a4
            x += M.log_ratio[z-1].get_log_state_action_density_ratio(D, None).item()


            #data[i][j] = M.ratio[z].get_state_action_density_ratio(D, None).item()
            data_l[i][j] = x/4

    data = np.array(data)
    data_l = np.array(data_l)

    print(data_l)
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data, cmap='RdYlBu_r')
    plt.show()
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data_l, cmap='RdYlBu_r')
    plt.savefig("reward_landscape/z_" + str(z))
    #plt.show()

    state = M.initalize()
    print(M.current_index )

    print(data_l)

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
            M.save_main("gradual_models/10x10/DivQL/q", "gradual_models/10x10/DivQL/target_q")
        if i % eval_interval == 0:


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
                D = Data()
                D.state = s
                D.action = a

                sample_hot_vec = np.array([0.0 for i in range(M.q_nn_param.action_dim)])
                sample_hot_vec[a] = 1
                D.action = sample_hot_vec

                ra = M.ratio[z].get_state_action_density_ratio(D, None).item()
                l_p = M.log_ratio[z].get_log_state_action_density_ratio(D, None).item()

                a_r += 3*ra
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





