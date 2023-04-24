import torch
import numpy as np
import sys
from q_learning import Q_learning
from env.gridworld.environments import GridWalk
from model import NN_Paramters
from parameters import Algo_Param, Save_Paths, Load_Paths
from memory import Replay_Memory

q_param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.05)
algo_param = Algo_Param()
algo_param.gamma = 0.9

max_episodes = 100

grid_size = 10

env = GridWalk(grid_size, False)
env2 = GridWalk(grid_size, False)

#Q = Q_learning(env, q_param, algo_param)

from q_learning_1 import Q_learning
Q = Q_learning(env, q_param, algo_param,)
Q.load("gradual_models/10x10/q/q0", "gradual_models/10x10/q/target_q0")

update_interval = 10
save_interval = 1000
eval_interval = 1000
state = Q.initalize()
memory = Replay_Memory(5000)

for i in range(10000):
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
        memory.push(s, a, r, n_s, i_s, j, True)
        if j == max_episodes-1:
            d = True
            #memory.push(s, a, r, n_s, i_s, j, True )
        if d == True:
            s =  n_s
            n_s = None
            memory.push(s, a, r, n_s, i_s, j, True)
            break


        s = n_s


    if i % eval_interval == 0:
        print("reward at itr " + str(i) + " = " + str(rew) + " at state: " + str(s) + " starting from: " + str(i_s))
        print(S)
#Q.memory = torch.load("mem")
torch.save(memory, "gradual_models/10x10/q/mem0")
