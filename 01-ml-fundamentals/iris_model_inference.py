
from pathlib import Path

import joblib
import torch
import torch.nn as nn


# Define file paths
script_dir = Path(__file__).resolve().parent
model_path = script_dir / "iris_scaled_model.pth"
scaler_path = script_dir / "iris_scaler.pkl"

# Define the same model architecture used during training
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

# Load the saved model weights
model.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=True)
)

# Set the model to evaluation mode
model.eval()

# Load the saved scaler
scaler = joblib.load(scaler_path)

# Define one new flower sample in its original measurements

# Define multiple flower samples using original measurements
flower_samples = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.7, 3.0, 5.0, 1.7],
]

class_names = ["setosa", "versicolor", "virginica"]

# Scale all samples using the saved scaler
scaled_samples = scaler.transform(flower_samples)

# Convert scaled samples to a PyTorch tensor
input_tensor = torch.tensor(
    scaled_samples,
    dtype=torch.float32
)

# Predict all samples
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(
        probabilities,
        dim=1
    )

# Display the results
for i, (flower, predicted_class, probs) in enumerate(
    zip(flower_samples, predicted_classes, probabilities),
    start=1
):
    predicted_name = class_names[predicted_class.item()]
    confidence = probs[predicted_class].item()

    print(f"\nFlower {i}")
    print("Original features:", flower)
    print("Predicted flower:", predicted_name)
    print(f"Confidence: {confidence:.2%}")

    for name, probability in zip(class_names, probs.tolist()):
        print(f"  {name}: {probability:.2%}")