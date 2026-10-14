
import torch
import torch.nn as nn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


torch.manual_seed(42)

iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.125,
    random_state=42,
    stratify=y_train_val
)

# Standardize features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


for hidden_size in [8, 16, 32]:
    torch.manual_seed(42)

    # Create a model with the selected hidden layer size
    model = nn.Sequential(
        nn.Linear(4, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 3)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    # Train the model
    for epoch in range(150):
        model.train()

        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()

    with torch.no_grad():
        train_outputs = model(X_train)
        train_predictions = train_outputs.argmax(dim=1)
        train_accuracy = (
            train_predictions == y_train
        ).float().mean().item()

        val_outputs = model(X_val)
        val_predictions = val_outputs.argmax(dim=1)
        val_accuracy = (
            val_predictions == y_val
        ).float().mean().item()
        val_loss = criterion(val_outputs, y_val).item()

        test_outputs = model(X_test)
        test_predictions = test_outputs.argmax(dim=1)
        test_accuracy = (
            test_predictions == y_test
        ).float().mean().item()

    print(f"Hidden neurons: {hidden_size}")
    print(f"Train accuracy: {train_accuracy:.2%}")
    print(f"Validation accuracy: {val_accuracy:.2%}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print("-" * 30)