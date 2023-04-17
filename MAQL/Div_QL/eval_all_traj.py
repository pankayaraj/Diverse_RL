import torch
import numpy as np
import sys
from q_learning import Q_learning
from env.gridworld.environments import GridWalk
from model import NN_Paramters
from parameters import Algo_Param, Save_Paths, Load_Paths
from memory import Replay_Memory

q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.05)
algo_param = Algo_Param()
algo_param.gamma = 0.9

max_episodes = 100

grid_size = 10

env = GridWalk(grid_size, False)
env2 = GridWalk(grid_size, False)

#Q = Q_learning(env, q_param, algo_param)

num_z = 10
Z = [0, 1, 2, 3,4, 5, 6, 7, 8, 9]
loc = "gradual_models/5/"

All_traj = []
from q_learning_1 import Q_learning
for z in Z:

    Q = Q_learning(env, q_param, algo_param,)
    Q.load( loc + "q" + str(z), loc + "target_q" + str(z))

    update_interval = 10
    save_interval = 1000
    eval_interval = 1000
    state = Q.initalize()
    memory = Replay_Memory(5000)


    s = env2.reset()
    i_s = s
    rew = 0
    S = []
    for j in range(max_episodes):
        a = Q.get_action(s)

        S.append(s)
        n_s, r, d, _ = env2.step(a)

        sample_hot_vec = np.array([0.0 for i in range(q_param.action_dim)])
        sample_hot_vec[a] = 1
        a = sample_hot_vec
        rew += r

        if j == max_episodes-1:
            d = True

        if d == True:
            n_s = None

            break
        memory.push(s, a, r, n_s, i_s, j, True)
        s = n_s

    print("reward" + " = " + str(rew) + " at state: " + str(s) + " starting from: " + str(i_s))
    print(S)
    All_traj.append(S)

torch.save(All_traj, loc + "Trajectories")
