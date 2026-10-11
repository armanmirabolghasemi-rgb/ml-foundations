import torch
import torch.nn as nn

# Define input values
values = [-4.0, -2.0, 0.0, 2.0, 4.0]

# Define activation functions
activations = {
    "ReLU": nn.ReLU(),
    "Leaky ReLU": nn.LeakyReLU(negative_slope=0.01),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
}

# Compare outputs and gradients
for name, activation in activations.items():
    x = torch.tensor(values, requires_grad=True)

    output = activation(x)
    output.sum().backward()

    print(f"\n{name}")
    print("Output:  ", output.detach().round(decimals=4))
    print("Gradient:", x.grad.round(decimals=4))