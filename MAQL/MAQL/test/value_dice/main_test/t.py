import  torch

a = torch.load("target")
for i in a.iterate_through():
    print(i)