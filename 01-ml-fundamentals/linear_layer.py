import torch
import torch.nn as nn

layer = nn.Linear(2, 1)

x = torch.tensor([[2.0, 3.0]])

output = layer(x)

print("Input:", x)
print("Output:", output)

print("Weights:")
print(layer.weight)

print("Bias:")
print(layer.bias)