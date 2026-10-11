
import torch
import torch.nn as nn

torch.manual_seed(42)

# Create a neural network with two linear layers
model = nn.Sequential(
    nn.Linear(1, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)

# Set known initial parameters to make the gradients easier to inspect
with torch.no_grad():
    model[0].weight.copy_(torch.tensor([[1.0], [2.0]]))
    model[0].bias.copy_(torch.tensor([0.5, -1.0]))
    model[2].weight.copy_(torch.tensor([[1.0, -1.0]]))
    model[2].bias.fill_(0.0)

# Define input and target
x = torch.tensor([[2.0]])
target = torch.tensor([[3.0]])

# Define loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Initial prediction:", model(x).item())

# Train the network
for epoch in range(101):
    prediction = model(x)
    loss = loss_fn(prediction, target)

    optimizer.zero_grad()
    loss.backward()

    # Inspect gradients before the first update
    if epoch == 0:
        print("Initial loss:", loss.item())

        for name, parameter in model.named_parameters():
            print(f"{name} gradient:\n{parameter.grad}")

    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            current_prediction = model(x).item()

        print(
            f"Epoch: {epoch} | "
            f"Loss: {loss.item():.6f} | "
            f"Prediction: {current_prediction:.6f}"
        )

# Display final results
print("Final prediction:", model(x).item())

print("\nFinal parameters:")
for name, parameter in model.named_parameters():
    print(f"{name}:\n{parameter.detach()}")