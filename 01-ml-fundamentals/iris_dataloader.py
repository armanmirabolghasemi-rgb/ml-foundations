import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load dataset
iris = load_iris()

features = torch.tensor(
    iris.data,
    dtype=torch.float32
)

labels = torch.tensor(
    iris.target,
    dtype=torch.long
)


# Split data into train+validation and test sets
features_train_val, features_test, labels_train_val, labels_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
)


# Split train+validation into train and validation sets
features_train, features_validation, labels_train, labels_validation = train_test_split(
    features_train_val,
    labels_train_val,
    test_size=15,
    random_state=42
)


# Create datasets
train_dataset = TensorDataset(
    features_train,
    labels_train
)

validation_dataset = TensorDataset(
    features_validation,
    labels_validation
)

test_dataset = TensorDataset(
    features_test,
    labels_test
)


# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# Create model
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)


# Create loss function and optimizer
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# Early stopping settings
best_validation_loss = float("inf")
patience = 3
epochs_without_improvement = 0

best_model_state = None


# Training
for epoch in range(50):

    model.train()

    total_loss = 0

    for features_batch, labels_batch in train_loader:

        prediction = model(features_batch)

        loss = loss_function(
            prediction,
            labels_batch
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_train_loss = (
        total_loss / len(train_loader)
    )


    # Validation
    model.eval()

    total_validation_loss = 0
    correct_validation = 0
    total_validation = 0

    with torch.no_grad():

        for features_batch, labels_batch in validation_loader:

            prediction = model(features_batch)

            loss = loss_function(
                prediction,
                labels_batch
            )

            total_validation_loss += loss.item()

            predicted_class = torch.argmax(
                prediction,
                dim=1
            )

            correct_validation += (
                predicted_class == labels_batch
            ).sum().item()

            total_validation += labels_batch.size(0)


    average_validation_loss = (
        total_validation_loss / len(validation_loader)
    )

    validation_accuracy = (
        correct_validation / total_validation
    )


    print(
        f"Epoch: {epoch + 1} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Loss: {average_validation_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )


    # Check validation loss improvement
    if average_validation_loss < best_validation_loss:

        best_validation_loss = average_validation_loss

        epochs_without_improvement = 0

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        print("Best model updated.")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/{patience}"
        )


    # Stop training when validation loss does not improve
    if epochs_without_improvement >= patience:

        print("Early stopping triggered.")

        break


# Restore the best model
model.load_state_dict(
    best_model_state
)

# Save the best model
torch.save(
    model.state_dict(),
    "iris_model.pth"
)

print("Model saved.")

# Final test evaluation
model.eval()

correct_test = 0
total_test = 0

with torch.no_grad():

    for features_batch, labels_batch in test_loader:

        prediction = model(features_batch)

        predicted_class = torch.argmax(
            prediction,
            dim=1
        )

        correct_test += (
            predicted_class == labels_batch
        ).sum().item()

        total_test += labels_batch.size(0)


test_accuracy = correct_test / total_test

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)