import torch

batch_size = 2
act_dim = 4

one_hot_action = [torch.zeros([batch_size, act_dim]).to(torch.device("cuda")) for _ in range(act_dim)]
print(one_hot_action)
for action_i in range(act_dim):

    one_hot_action[action_i][:,action_i] = 1

print(one_hot_action)