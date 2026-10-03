import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0],
    [6.0, 7.0],
    [7.0, 8.0],
    [8.0, 9.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0],
    [13.0],
    [15.0],
    [17.0]
])

dataset = TensorDataset(x, y)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for batch_x, batch_y in loader:

    print("X:")
    print(batch_x)

    print("Y:")
    print(batch_y)

    print("----------------")