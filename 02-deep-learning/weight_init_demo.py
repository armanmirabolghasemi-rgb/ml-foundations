
import torch
import torch.nn as nn

torch.manual_seed(42)

# Create a linear layer with default initialization
default_layer = nn.Linear(4, 16)

# Create a linear layer with Xavier initialization
xavier_layer = nn.Linear(4, 16)
nn.init.xavier_uniform_(xavier_layer.weight)
nn.init.zeros_(xavier_layer.bias)

# Create a linear layer with He initialization
he_layer = nn.Linear(4, 16)
nn.init.kaiming_normal_(
    he_layer.weight,
    nonlinearity="relu"
)
nn.init.zeros_(he_layer.bias)


def show_stats(name, layer):
    weights = layer.weight.detach()

    print(f"\n{name}")
    print(f"Weight mean: {weights.mean().item():.4f}")
    print(f"Weight std: {weights.std().item():.4f}")
    print(f"Weight shape: {weights.shape}")


show_stats("Default initialization", default_layer)
show_stats("Xavier initialization", xavier_layer)
show_stats("He initialization", he_layer)


print("\nFirst 5 weights:")

print("Default:")
print(default_layer.weight.flatten()[:5])

print("Xavier:")
print(xavier_layer.weight.flatten()[:5])

print("He:")
print(he_layer.weight.flatten()[:5])