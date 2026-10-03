import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# Dataset
# -------------------------

x = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0],
    [6.0, 7.0],
    [7.0, 8.0],
    [8.0, 9.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0],
    [13.0],
    [15.0],
    [17.0]
])

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# -------------------------
# Model
# -------------------------

model = nn.Linear(2, 1)

# -------------------------
# Loss & Optimizer
# -------------------------

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# -------------------------
# Training
# -------------------------

for epoch in range(100):

    for batch_x, batch_y in loader:

        # Prediction
        prediction = model(batch_x)

        # Loss
        loss = loss_function(prediction, batch_y)

        # Clear old gradients
        optimizer.zero_grad()

        # Calculate gradients
        loss.backward()

        # Update weights
        optimizer.step()

    if epoch % 10 == 0:
        print(
            "Epoch:", epoch,
            "| Loss:", round(loss.item(), 4)
        )
        test_x = torch.tensor([
    [9.0, 10.0],
    [10.0, 11.0]
])

with torch.no_grad():
    result = model(test_x)

print("\nPrediction:")
print(result)