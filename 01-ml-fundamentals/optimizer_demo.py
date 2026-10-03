import torch
import torch.nn as nn

# مدل
layer = nn.Linear(2, 1)

# ورودی
x = torch.tensor([[2.0, 3.0]])

# جواب واقعی
target = torch.tensor([[10.0]])

# Optimizer
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

# -------------------------
# قبل از یادگیری
# -------------------------

prediction = layer(x)
loss = (prediction - target) ** 2

print("BEFORE")
print("Prediction:", prediction.item())
print("Loss:", loss.item())


# -------------------------
# یادگیری
# -------------------------

optimizer.zero_grad()

loss.backward()

optimizer.step()


# -------------------------
# بعد از یادگیری
# -------------------------

new_prediction = layer(x)
new_loss = (new_prediction - target) ** 2

print("\nAFTER")
print("Prediction:", new_prediction.item())
print("Loss:", new_loss.item())