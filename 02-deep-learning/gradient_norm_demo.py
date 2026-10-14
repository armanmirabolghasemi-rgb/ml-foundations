
import torch
import torch.nn as nn

# Create two parameters
first = nn.Parameter(torch.tensor(0.0))
second = nn.Parameter(torch.tensor(0.0))

# Assign gradients manually
first.grad = torch.tensor(3.0)
second.grad = torch.tensor(4.0)

parameters = [first, second]

# Calculate the original gradient norm
original_norm = torch.sqrt(
    first.grad ** 2 + second.grad ** 2
)

print("Original gradients:", first.grad.item(), second.grad.item())
print("Original norm:", original_norm.item())

# Clip the combined gradient norm
nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

# Calculate the new gradient norm
clipped_norm = torch.sqrt(
    first.grad ** 2 + second.grad ** 2
)

print("Clipped gradients:", first.grad.item(), second.grad.item())
print("Clipped norm:", clipped_norm.item())