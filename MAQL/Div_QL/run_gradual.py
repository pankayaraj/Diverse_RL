
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



device = torch.device("cpu")
policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device= device)
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=device, l_r=0.0001)
q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=device, l_r=0.05)
algo_param = Algo_Param(hard_update_interval=1)
algo_param.gamma = 0.9
num_z = 10
Z = [1, 2, 3, 4, 5, 6, 7, 8, 9]

grid_size = 10
env = GridWalk(grid_size, False)
env2 = GridWalk(grid_size, False)
env_tar = GridWalk(grid_size, False)

#behaviour_Q  = Q_learning(env, q_param, algo_param)
#behaviour_policy = Q_learner_Policy(behaviour_Q.Q, q_param)
#behaviour_policy.Q.load("q")

update_interval = 100
save_interval = 10
eval_interval = 10
max_episodes = 100

inital_log_buffer = torch.load("gradual_models/1/mem0")

from DivQL_Gradual_1 import DivQL_Gradual
M  = DivQL_Gradual(env, inital_log_buffer, q_param, nu_param, algo_param, num_z)


M.Q[0].load("gradual_models/1/q0")
M.Target_Q[0].load("gradual_models/1/target_q0")




for z in Z:

    M.initalize(80)
    M.add_new_replay_buffer(z)
    print(M.current_index )
    for i in range(100):
        print(i)
        M.step(80,z)
        M.eval(1,80,z)
    for i in range(100):

        M.step(80,z)

        if i % update_interval == 0:
            M.hard_update(z)
        if i % save_interval == 0:
            print("saving")
            M.save(z,"gradual_models/3/q" + str(z), "gradual_models/3/target_q" + str(z), "gradual_models/3/nu" + str(z) + "_1", "gradual_models/3/nu" + str(z) + "_2")

        if i % eval_interval == 0:


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
        torch.save(M.log_ratio_memory, "gradual_models/3/mem" + str(z))