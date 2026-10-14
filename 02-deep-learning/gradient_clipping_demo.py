
import torch
import torch.nn as nn

# Create a parameter and an SGD optimizer
weight = nn.Parameter(torch.tensor(2.0))
optimizer = torch.optim.SGD([weight], lr=0.01)

# Simulate a very large gradient
weight.grad = torch.tensor(1000.0)

print("Weight before update:", weight.item())
print("Original gradient:", weight.grad.item())

# Limit the gradient norm to 1.0
nn.utils.clip_grad_norm_([weight], max_norm=1.0)

print("Clipped gradient:", weight.grad.item())

# Update the parameter
optimizer.step()

print("Weight after update:", weight.item())