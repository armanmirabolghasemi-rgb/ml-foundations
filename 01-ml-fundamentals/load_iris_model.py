import torch
import torch.nn as nn


# Create the same model architecture
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)


# Load saved model weights
model.load_state_dict(
    torch.load("iris_model.pth")
)


# Set model to evaluation mode
model.eval()


def predict(sample):

    # Convert input data to tensor
    sample = torch.tensor(sample, dtype=torch.float32)

    # Add batch dimension
    sample = sample.unsqueeze(0)

    # Make prediction without calculating gradients
    with torch.no_grad():

        output = model(sample)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        )

    return (
        predicted_class.item(),
        probabilities[0].tolist()
    )


# Predict a new Iris sample
samples = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.2, 2.0]
]


for sample in samples:

    predicted_class, probabilities = predict(sample)

    print("Sample:", sample)
    print("Probabilities:", probabilities)
    print(f"Predicted class: {predicted_class}")
    print()