
import torch
import torch.nn as nn

# Define input values
x = torch.tensor([-3.0, 2.0], requires_grad=True)

# Define activation functions
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# Run the forward pass
relu_output = relu(x)
leaky_output = leaky_relu(x)

print("Input:", x.detach())
print("ReLU output:", relu_output.detach())
print("Leaky ReLU output:", leaky_output.detach())

# Calculate gradients for ReLU
relu_output.sum().backward(retain_graph=True)
print("ReLU gradient:", x.grad.clone())

# Reset gradients before the next calculation
x.grad.zero_()

# Calculate gradients for Leaky ReLU
leaky_output.sum().backward()
print("Leaky ReLU gradient:", x.grad)