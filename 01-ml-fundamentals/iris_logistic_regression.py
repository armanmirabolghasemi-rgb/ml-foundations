from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

print("X shape:", X.shape)
print("y shape:", y.shape)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Input:", sample)
print("Prediction:", prediction)