import torch


# Defining the function
def f(x):
    return torch.pow((x - 2.0), 2)


# Initializing the parameter
initial_x = torch.tensor([-3.5], requires_grad=True)
x_param = torch.nn.Parameter(initial_x)


# Optimizer
optimizer_sgd = torch.optim.SGD([x_param], lr=0.1)


# Number of Iterations
N_ITERATIONS = 60
for epoch in range(60):
    optimizer_sgd.zero_grad()
    loss_incurred = f(x_param)
    loss_incurred.backward()
    optimizer_sgd.step()


# Result
print(x_param.data)