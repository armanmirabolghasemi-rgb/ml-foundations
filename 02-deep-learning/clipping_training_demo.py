
import torch
import torch.nn as nn

x = torch.tensor(2.0)
target = torch.tensor(6.0)

weight_normal = nn.Parameter(torch.tensor(1.0))
weight_clipped = nn.Parameter(torch.tensor(1.0))

optimizer_normal = torch.optim.SGD([weight_normal], lr=0.1)
optimizer_clipped = torch.optim.SGD([weight_clipped], lr=0.1)

loss_fn = nn.MSELoss()

for step in range(5):
    # Train without gradient clipping
    optimizer_normal.zero_grad()

    prediction_normal = weight_normal * x
    loss_normal = loss_fn(prediction_normal, target)
    loss_normal.backward()

    gradient_normal = weight_normal.grad.item()
    optimizer_normal.step()

    # Train with gradient clipping
    optimizer_clipped.zero_grad()

    prediction_clipped = weight_clipped * x
    loss_clipped = loss_fn(prediction_clipped, target)
    loss_clipped.backward()

    original_gradient = weight_clipped.grad.item()
    nn.utils.clip_grad_norm_([weight_clipped], max_norm=5.0)
    clipped_gradient = weight_clipped.grad.item()

    optimizer_clipped.step()

    print(f"Step: {step + 1}")
    print(f"Without clipping - Gradient: {gradient_normal:.4f}, Weight: {weight_normal.item():.4f}")
    print(f"With clipping    - Gradient: {original_gradient:.4f} -> {clipped_gradient:.4f}, Weight: {weight_clipped.item():.4f}")
    print("-" * 65)