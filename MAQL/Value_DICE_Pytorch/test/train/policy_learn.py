from model import DiscretePolicyNN, NN_Paramters
from env.gridworld.environments import GridWalk
from env.gridworld.optimal_policy import Optim_Policy_Gridwalk
from util.collect_trajectories import collect_data
import torch

param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[6, 6], non_linearity=torch.tanh, device= torch.device("cpu"))

D = DiscretePolicyNN(param, "1", "1")

env = GridWalk(10, tabular_obs=False)
policy = Optim_Policy_Gridwalk(env, 5)

#data, _, _ = collect_data(env, policy, 1000, 10)
#torch.save(data, "data")
data = torch.load("data")

optimizer = torch.optim.Adam(D.parameters(),lr=0.01)
loss = torch.nn.CrossEntropyLoss()

d = data.sample(size = 40)
for i in range(1000):
    state = d.state
    a = torch.max(torch.Tensor(d.action), 1)[1]
    action_pred =  D.forward_temp(state)

    optimizer.zero_grad()
    l = loss(action_pred, a)
    l.backward()
    optimizer.step()
    d = data.sample(size = 20)
    print(l)

D.save()





