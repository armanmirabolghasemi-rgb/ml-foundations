import torch
import torch.nn as nn

# Model
layer = nn.Linear(2, 1)

# Input
x = torch.tensor([[2.0, 3.0]])

# Target
target = torch.tensor([[10.0]])

# Optimizer
optimizer = torch.optim.SGD(
    layer.parameters(),
    lr=0.1
)


for epoch in range(20):

    # Prediction
    prediction = layer(x)

    # Loss
    loss = (prediction - target) ** 2

    # Clear old gradients
    optimizer.zero_grad()

    # Calculate gradients
    loss.backward()

    # Update weights and bias
    optimizer.step()

    print(
        "Epoch:", epoch,
        "| Prediction:", round(prediction.item(), 4),
        "| Loss:", round(loss.item(), 4)
    )