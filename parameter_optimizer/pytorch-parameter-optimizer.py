import torch


def f(x):
    return torch.pow((x-2.0), 2)


x_param = torch.nn.Parameter(torch.tensor([-3.5], requires_grad=True))
sgd_optim = torch.optim.SGD([x_param], lr=0.1)

for epoch in range(0, 100):
    sgd_optim.zero_grad()
    loss_value = f(x_param)
    loss_value.backward()
    sgd_optim.step()

print('Solution to the equation is: {}'.format(x_param.data))