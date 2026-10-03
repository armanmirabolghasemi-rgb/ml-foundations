import torch
import torch.nn as nn

layer = nn.Linear(2, 1)

x = torch.tensor([[2.0, 3.0]])
target = torch.tensor([[10.0]])

prediction = layer(x)

loss = (prediction - target) ** 2

print("Prediction:", prediction)
print("Target:", target)
print("Loss:", loss)

loss.backward()

print("\nWeight:")
print(layer.weight)

print("\nWeight Gradient:")
print(layer.weight.grad)

print("\nBias:")
print(layer.bias)

print("\nBias Gradient:")
print(layer.bias.grad)