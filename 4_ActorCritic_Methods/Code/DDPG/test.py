import torch

a= torch.tensor([1,2,3])
b= torch.full((2,),1)
c = torch.full(a.shape,1)
d= torch.normal(mean=torch.full(a.shape, 0.0), std=torch.full(a.shape, 0.1))
# e= torch.full(a.shape,1)
print()