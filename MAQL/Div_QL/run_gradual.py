
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
num_z = 7
Z = [1, 2, 3, 4, 5, 6 ]

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

inital_log_buffer = torch.load("gradual_models/10x10/q/mem0")

from DivQL_Gradual_3 import DivQL_Gradual
M  = DivQL_Gradual(env, inital_log_buffer, q_param, nu_param, algo_param, num_z)


#M.Q[0].load("gradual_models/10x10/q/q0")
#M.Target_Q[0].load("gradual_models/10x10/q/target_q0")




for z in Z:

    M.initalize(80)
    #M.add_new_replay_buffer(z)
    print(M.current_index )
    for i in range(100):
        print(i)
        M.step(80,z)
        M.eval(1,80,z)
    for i in range(3000):

        M.step(95,z)

        if i % update_interval == 0:
            M.hard_update(z)
        if i % save_interval == 0:
            if i >10000:
                print(M.dist)
            print("saving")
            M.save(z,"gradual_models/10x10/3/q" + str(z), "gradual_models/10x10/3/target_q" + str(z), "gradual_models/10x10/3/nu" + str(z) + "_1", "gradual_models/10x10/3/nu" + str(z) + "_2")

        if i % eval_interval == 0:
            OM = np.array(M.occupancy_measure)/np.sum(M.occupancy_measure)
            OM_P = M.occupancy_measure_prev
            #print(OM)
            #print(OM_P)
            S = []
            A = []
            s = env2.reset()
            i_s = s
            rew = 0
            a_r = 0
            for j in range(max_episodes):
                S.append(list(s))
                a = M.get_action(s, z)

                A.append(a)



                s, r, d, _ = env2.step(a)
                rew += r


                #p1 = OM[s[1]][s[0]]
                p1 = 1/100
                p2 = OM_P[s[1]][s[0]]
                ra = (p1 ) / (p2 + 0.01)
                l_p = (p1 ) / (p2 + 0.01)
                a_r += ra
                #print(s, r, ra)
                if j == max_episodes - 1:
                    d = True
                if d == True:
                    break
            print("for z = " + str(z) + " reward at itr " + str(i) + " = " + str(rew) + ", " + str(rew+algo_param.alpha*a_r) + " at state: " + str(s) + " starting from: " + str(i_s) + " eps " + str(M.epsilon))
            print("Instances = " + str(M.instances))
            print(S)
            print(A)
        # Q.memory = torch.load("mem")
        #torch.save(M.log_ratio_memory, "gradual_models/3/mem" + str(z))