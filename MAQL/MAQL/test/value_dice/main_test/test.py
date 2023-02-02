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
Q_behave = Q_learning(env, q_param, algo_param)
Q_behave.load("q", "target_q")

target_policy = Q_learner_Policy(Q_behave.Q, Q_behave.q_nn_param)

V = Value_Dice(target_policy, nu_param, algo_param)
V.nu_network.load("nu_" + str(current_iter))
grid_size = 10

Buffer = torch.load("behavior_q")
Target_Buffer = torch.load("target")

# ++
# +torch.save(Buffer, "behavior_sub")
# torch.save(Target_Buffer, "target")

f_b = collect_freqency(Buffer, grid_size * grid_size, grid_size, algo_param.gamma)
f_t = collect_freqency(Target_Buffer, grid_size * grid_size, grid_size, algo_param.gamma)
ratio = f_t / f_b
log_ratio = np.log(ratio)

no_states = grid_size * grid_size
log_ratio_estimate = np.zeros([no_states, ])
data = Target_Buffer.sample(Buffer.no_data)
nu = V.get_log_state_action_density_ratio(data)

w = torch.reshape(torch.Tensor(algo_param.gamma ** data.time_step), shape=(Buffer.no_data, 1))
print(w)
print(nu.size(), w.size())
print(nu)
print(nu * w)
print(torch.sum(w), torch.sum(nu * w))
print(torch.sum(nu * w) / torch.sum(w))


