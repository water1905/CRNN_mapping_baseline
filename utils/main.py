import torch

a = torch.ones(4, 4, 4)
# print(torch.mean(a, 1, keepdim=True).shape)
print(torch.mean(a))
