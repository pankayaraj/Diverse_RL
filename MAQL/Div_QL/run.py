import torch
import numpy as np

import torch
import numpy as np

from DivQL import DivQL

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



device = torch.device("cpu")
policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= device)
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device=device, l_r=0.0001)
q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=device, l_r=0.05)
algo_param = Algo_Param(hard_update_interval=1)
algo_param.gamma = 0.9
num_z = 2

grid_size = 10
env = GridWalk(grid_size, False)
env2 = GridWalk(grid_size, False)
env_tar = GridWalk(grid_size, False)

#behaviour_Q  = Q_learning(env, q_param, algo_param)
#behaviour_policy = Q_learner_Policy(behaviour_Q.Q, q_param)
#behaviour_policy.Q.load("q")

update_interval = 1
save_interval = 10
eval_interval = 10
max_episodes = 100

M = DivQL(env, q_param, nu_param, algo_param, num_z)

for i in range(100):
    print(i)
    M.step(80)
for i in range(1000):
    M.step(80)

    if i % update_interval == 0:
        M.hard_update()
    if i % save_interval == 0:
        print("saving")
        M.save("q2", "target_q2", "nu2")

    if i % eval_interval == 0:
        z = np.random.randint(0, num_z)

        s = env2.reset()
        i_s = s
        rew = 0
        for j in range(max_episodes):
            a = M.get_action(s, z)
            s, r, d, _ = env2.step(a)
            rew += r
            if j == max_episodes - 1:
                d = True
            if d == True:
                break
        print("for z = " + str(z) + " reward at itr " + str(i) + " = " + str(rew) + " at state: " + str(s) + " starting from: " + str(i_s))
    # Q.memory = torch.load("mem")
    torch.save(M.memory, "mem")