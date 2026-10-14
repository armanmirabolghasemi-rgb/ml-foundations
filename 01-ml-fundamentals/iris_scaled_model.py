import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib


iris = load_iris()

X = iris.data
y = iris.target


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


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

test_dataset = TensorDataset(
    X_test,
    y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)


loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


best_validation_loss = float("inf")

patience = 3

epochs_without_improvement = 0

best_model_state = None


for epoch in range(50):

    model.train()

    total_train_loss = 0

    for batch_features, batch_labels in train_loader:

        predictions = model(batch_features)

        loss = loss_function(
            predictions,
            batch_labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()


    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    model.eval()

    total_validation_loss = 0

    correct = 0

    total = 0


    with torch.no_grad():

        for batch_features, batch_labels in val_loader:

            predictions = model(batch_features)

            loss = loss_function(
                predictions,
                batch_labels
            )

            total_validation_loss += loss.item()

            predicted_classes = torch.argmax(
                predictions,
                dim=1
            )

            correct += (
                predicted_classes == batch_labels
            ).sum().item()

            total += batch_labels.size(0)


    average_validation_loss = (
        total_validation_loss / len(val_loader)
    )

    validation_accuracy = correct / total


    print(
        f"Epoch: {epoch + 1} | "
        f"Train Loss: {average_train_loss:.4f} | "
        f"Validation Loss: {average_validation_loss:.4f} | "
        f"Validation Accuracy: {validation_accuracy:.4f}"
    )


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


    if epochs_without_improvement >= patience:

        print("Early stopping triggered.")

        break


model.load_state_dict(
    best_model_state
)


from sklearn.metrics import confusion_matrix, classification_report

model.eval()

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch_features, batch_labels in test_loader:

        predictions = model(batch_features)

        predicted_classes = torch.argmax(
            predictions,
            dim=1
        )

        correct += (
            predicted_classes == batch_labels
        ).sum().item()

        total += batch_labels.size(0)

        all_predictions.extend(
            predicted_classes.tolist()
        )

        all_labels.extend(
            batch_labels.tolist()
        )

test_accuracy = correct / total

print()
print(f"Test Accuracy: {test_accuracy:.4f}")

confusion = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("Confusion Matrix:")
print(confusion)

print()
print("Classification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=iris.target_names
    )
)

print()
print("Misclassified Samples:")

for index in range(len(all_labels)):

    if all_labels[index] != all_predictions[index]:

        scaled_sample = X_test[index].numpy()

        original_sample = scaler.inverse_transform(
            scaled_sample.reshape(1, -1)
        )[0]

        print("Scaled Features:", scaled_sample.tolist())

        print(
            "Original Features:",
            original_sample.tolist()
        )

        print(
            "Actual:",
            iris.target_names[all_labels[index]]
        )

        print(
            "Predicted:",
            iris.target_names[all_predictions[index]]
        )

        print()
        
# Save the model and scaler
script_dir = Path(__file__).resolve().parent

torch.save(
    model.state_dict(),
    script_dir / "iris_scaled_model.pth"
)

joblib.dump(
    scaler,
    script_dir / "iris_scaler.pkl"
)

print("Model and scaler saved successfully.")
