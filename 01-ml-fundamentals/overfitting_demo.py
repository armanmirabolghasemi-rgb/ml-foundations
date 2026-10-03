
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(201):
    model.train()

    prediction = model(X_train)
    loss = loss_function(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            test_pred = model(X_test).argmax(dim=1)

            train_acc = (train_pred == y_train).float().mean()
            test_acc = (test_pred == y_test).float().mean()

        print(
            f"Epoch: {epoch} | "
            f"Loss: {loss.item():.4f} | "
            f"Train: {train_acc.item():.2%} | "
            f"Test: {test_acc.item():.2%}"
        )

        
from sklearn.metrics import confusion_matrix, classification_report

model.eval()

with torch.no_grad():
    prediction = model(X_test).argmax(dim=1)

print("Confusion Matrix:")
print(confusion_matrix(y_test.numpy(), prediction.numpy()))

print("\nClassification Report:")
print(classification_report(
    y_test.numpy(),
    prediction.numpy(),
    target_names=iris.target_names
))