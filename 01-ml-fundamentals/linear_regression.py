import torch

x = torch.tensor([2.0])

weight = torch.tensor([0.5], requires_grad=True)
bias = torch.tensor([0.0], requires_grad=True)

target = torch.tensor([4.0])

for step in range(10):

    prediction = x * weight + bias
    loss = (prediction - target) ** 2

    loss.backward()

    with torch.no_grad():
        weight -= 0.1 * weight.grad
        bias -= 0.1 * bias.grad

    weight.grad.zero_()
    bias.grad.zero_()

    print(
        "Step:", step,
        "Prediction:", round(prediction.item(), 3),
        "Loss:", round(loss.item(), 3)
    )