import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# گرفتن داده‌ها
iris = load_iris()

X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)

# تقسیم داده به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ساخت Neural Network
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

# Loss مخصوص Classification
loss_function = nn.CrossEntropyLoss()

# روش بهینه‌سازی
optimizer = optim.Adam(model.parameters(), lr=0.01)

# آموزش
for step in range(500):

    prediction = model(X_train)

    loss = loss_function(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Step:", step, "Loss:", loss.item())

# تست مدل
with torch.no_grad():

    prediction = model(X_test)

    predicted_class = torch.argmax(prediction, dim=1)

    accuracy = (predicted_class == y_test).float().mean()

print("Accuracy:", accuracy.item())