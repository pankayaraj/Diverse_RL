import numpy as np
import torch
from model import NN_Paramters
torch.device('cuda')

param = NN_Paramters(state_dim=5, action_dim=4, hidden_layer_dim=[10], non_linearity=None, device= torch.device("cuda"))

from model import DiscretePolicyNN
M = DiscretePolicyNN(nn_params=param, save_path="temp", load_path="temp")


state = np.array([[0, 0, 0, 0, 0], [1,1,1,1,1]])

print(M.get_probabilities(state, format="numpy"))
print(M.sample(state))
x = np.argmax(M.get_probabilities(state, format="numpy")[0:])

print(M.get_probabilities(state, format="numpy")[1:])
print(M.get_probability(state, 3, format="numpy"))
print(M.get_log_probability(state, 3))

M.save(None)
M.load(None)

j = 0
print(x)
print(M.sample(state))
for i in range(1000):
    if x == M.sample(state, format="numpy")[1]:
        j += 1
print (j)