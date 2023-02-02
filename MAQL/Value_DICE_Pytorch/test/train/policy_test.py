from model import DiscretePolicyNN, NN_Paramters
from env.gridworld.environments import GridWalk
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk
from util.collect_trajectories import collect_data
import torch

param = NN_Paramters(state_dim=2, action_dim=4, hidden_layer_dim=[10], non_linearity=None, device= torch.device("cpu"))

D = DiscretePolicyNN(param, "1", "1")

env = GridWalk(10, tabular_obs=False)
policy = Optim_Policy_Gridwalk(env)

data, _, _ = collect_data(env, policy, 10, 10)

D.load()

for i in range(100):

    d = data.sample(size=1)
    state = d.state
    action = d.action

    print(D.get_probabilities(state, format="numpy"), action)
