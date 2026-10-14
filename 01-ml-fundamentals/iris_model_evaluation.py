
from pathlib import Path

import joblib
import torch
import torch.nn as nn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# Define paths
script_dir = Path(__file__).resolve().parent
model_path = script_dir / "iris_scaled_model.pth"
scaler_path = script_dir / "iris_scaler.pkl"

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Recreate the same train and test split
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

# Load the saved scaler and transform test features
scaler = joblib.load(scaler_path)
X_test_scaled = scaler.transform(X_test)

# Convert test data to tensors
X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

# Define the same model architecture
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

# Load saved weights
model.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=True)
)
model.eval()

# Predict all test samples
with torch.no_grad():
    logits = model(X_test_tensor)
    predictions = torch.argmax(logits, dim=1).numpy()

# Display confusion matrix
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2]
)

print("Confusion Matrix:")
print(matrix)

# Display classification metrics
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )
)

# Display incorrectly classified samples
print("\nMisclassified Samples:")

wrong_indices = [
    i for i in range(len(y_test))
    if y_test[i] != predictions[i]
]

if not wrong_indices:
    print("No misclassified samples.")
else:
    for i in wrong_indices:
        print(
            f"Actual: {class_names[y_test[i]]} | "
            f"Predicted: {class_names[predictions[i]]} | "
            f"Features: {X_test[i].tolist()}"
        )