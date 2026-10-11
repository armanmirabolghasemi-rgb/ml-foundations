
import torch
import torch.nn as nn

# Create a model with one input and one output
model = nn.Linear(1, 1, bias=False)

# Set the initial weight to 1
with torch.no_grad():
    model.weight.fill_(1.0)

# Define input and expected output
x = torch.tensor([[5.0]])
target = torch.tensor([[20.0]])

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Initial weight:", model.weight.item())

# Train the model
for epoch in range(100):
    # Calculate prediction and loss
    prediction = model(x)
    loss = loss_fn(prediction, target)

    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Display details for the first update
    if epoch == 0:
        print("Prediction:", prediction.item())
        print("Loss:", loss.item())
        print("Gradient:", model.weight.grad.item())
        print("Weight before update:", model.weight.item())

    # Update model weights
    optimizer.step()

    if epoch == 0:
        print("Weight after update:", model.weight.item())

    # Display training progress
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | "
            f"Loss: {loss.item():.4f} | "
            f"Weight: {model.weight.item():.4f}"
        )

# Display final result
print("Final weight:", model.weight.item())
print("Final prediction:", model(x).item())