
# Simulate repeated multiplication along a gradient path
initial_gradient = 1.0

cases = {
    "Small factor (0.1)": 0.1,
    "Neutral factor (1.0)": 1.0,
    "Large factor (3.0)": 3.0,
}

for name, factor in cases.items():
    gradient = initial_gradient

    for layer in range(8):
        gradient *= factor

    print(f"{name}: {gradient:.10f}")