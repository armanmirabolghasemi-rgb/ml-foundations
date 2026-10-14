import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create scaler
scaler = StandardScaler()


# Learn scaling parameters from training data
X_train = scaler.fit_transform(X_train)


# Apply the same scaling parameters to test data
X_test = scaler.transform(X_test)


# Convert data to PyTorch tensors
X_train = torch.tensor(
    X_train,
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

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


print("Original feature example:")
print(iris.data[0])

print()

print("Scaled feature example:")
print(X_train[0])

print()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)