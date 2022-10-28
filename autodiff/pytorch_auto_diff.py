import torch


def f(x):
    return torch.pow((x - 2.0), 2)


x = torch.tensor([-3.5], requires_grad=True)

x_cur = x.clone()
x_prev = x_cur * 100

eps = 1e-5
eta = 0.1

while torch.linalg.norm(x_cur - x_prev) > eps:
    x_prev = x_cur.clone()
    value = f(x)
    value.backward()

    x.data = x.data - (eta * x.grad)
    x.grad.zero_()
    x_cur = x.data

print("Solution: {}".format(x_cur))
