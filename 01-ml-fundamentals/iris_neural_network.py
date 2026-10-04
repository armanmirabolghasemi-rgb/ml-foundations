import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

for step in range(500):

    prediction = model(X_train)

    loss = loss_function(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Step:", step, "Loss:", loss.item())

with torch.no_grad():

    prediction = model(X_test)

    predicted_class = torch.argmax(prediction, dim=1)

    accuracy = (predicted_class == y_test).float().mean()

print("Accuracy:", accuracy.item())