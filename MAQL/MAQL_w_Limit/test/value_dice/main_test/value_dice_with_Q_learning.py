from value_dice import Algo_Param, Value_Dice
from util.count_frequency import collect_freqency
import torch
from model import NN_Paramters
import numpy as np

from parameters import Algo_Param, Save_Paths, Load_Paths
from q_learning.q_learning import Q_learning

from env.gridworld.environments import GridWalk
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk

from util.q_learning_to_policy import Q_learner_Policy
from util.collect_trajectories import collect_data
from util.density_ratio import get_policy_ratio


policy_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= torch.device("cpu"))
nu_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.0001)
q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[64], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.05)
algo_param = Algo_Param()
algo_param.gamma = 0.995

grid_size = 10
env = GridWalk(grid_size, False)

current_iter = 1000

Q = Q_learning(env, q_param, algo_param)
Q_behave = Q_learning(env, q_param, algo_param)
Q_behave.load("q", "target_q")
Q.load("q_val/q_" + str(current_iter), "tar_q_val/target_q_" + str(current_iter))

#target_policy = Q_learner_Policy(Q.Target_Q, Q.q_nn_param)
behaviour_policy = Optim_Policy_Gridwalk(grid_env=env, action_dim=5, eps_explore=0.2)
target_policy= Q_learner_Policy(Q_behave.Target_Q, Q_behave.q_nn_param)

print("initalization done")

V = Value_Dice(target_policy, nu_param, algo_param)
Buffer, _ , _ = collect_data(env, behaviour_policy, 1000, 10)
Target_Buffer, _ , _ = collect_data(env, target_policy, 1000, 10)

#Buffer = torch.load("behavior_q")
#Target_Buffer = torch.load("target")

print("initial memory collection done")
torch.save(Buffer, "behavior_q")
torch.save(Target_Buffer, "target")


f_b = collect_freqency(Buffer, grid_size*grid_size, grid_size, algo_param.gamma )
f_t = collect_freqency(Target_Buffer, grid_size*grid_size, grid_size, algo_param.gamma )


ratio = f_t/f_b


no_iterations = 30000
#V.nu_network.load("nu_1")

for i in range(no_iterations):

    data = Buffer.sample(size=400)

    V.train_KL(data=data)

    if i % 1000 == 0:
        print(i)
        B = Buffer.sample(1)
        s = B.state
        a = B.action
        n_s = B.next_state

        a_n  = 0
        for j in range(nu_param.action_dim):
            if a[0][j] == 1:
                a_n = j


        print(B.state, B.action)
        #s = np.array([[9, 9]])
        print(V.get_log_state_action_density_ratio(B),
              np.log(ratio[s[0][0]*grid_size+ s[0][1]]*get_policy_ratio(target_policy, behaviour_policy, s, a_n))
              )

        print(V.debug())
        V.nu_network.save("nu_" + str(current_iter))