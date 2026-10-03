import torch
import torch.nn as nn
import torch.optim as optim

# داده‌های آموزشی
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

# جواب‌های درست
y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])

# ساخت Neural Network
model = nn.Sequential(
    nn.Linear(1, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# Loss
loss_function = nn.MSELoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# آموزش
for step in range(1000):

    prediction = model(X)

    loss = loss_function(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print("Step:", step, "Loss:", loss.item())

# تست
test = torch.tensor([[5.0]])

result = model(test)

print("Input:", test.item())
print("Prediction:", result.item())