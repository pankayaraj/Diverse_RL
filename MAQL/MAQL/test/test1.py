import torch

a = torch.load("mem")
b = torch.load("l_mem")

for i in b.iterate_through():
    print(i)