import torch

# create two 2D tensors
tensor1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor2 = torch.tensor([[7, 8], [9, 10], [11, 12]])

# compute the dot product of the row elements of each tensor

tensor = tensor2-tensor1
tensor = tensor.float()
norms = torch.norm(tensor, dim=1)

print(norms)

import torch

# create a list of 1D tensors
tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([0, -1, 2])]

# create a tensor consisting of the minimum of each corresponding element
min_tensor = torch.stack(tensors).min(dim=0)[0]

print(min_tensor)