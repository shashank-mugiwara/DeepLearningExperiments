import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import *
from tqdm.autonotebook import tqdm


class Simple1DRegressionDataset(Dataset):

    def __init__(self, X, y):
        super(Simple1DRegressionDataset, self).__init__()
        self.X = X.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

    def __getitem__(self, index):
        inputs = torch.tensor(self.X[index, :], dtype=torch.float32)
        targets = torch.tensor(self.y[index], dtype=torch.float32)
        return inputs, targets

    def __len__(self):
        return self.X.shape[0]


def move_to(obj, device):
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    elif isinstance(obj, set):
        return set(move_to(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[move_to(key, device)] = move_to(value, device)
        return to_ret
    else:
        return obj


def train_simple_network(model, loss_function, training_loader, epochs=20, device="cpu"):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model = model.to(device)

    for epoch in tqdm(range(epochs), desc="Epoch:"):
        model = model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(training_loader, desc="Batch:", leave=False):
            inputs = move_to(inputs, device)
            labels = move_to(labels, device)

            optimizer.zero_grad()

            y_hat = model(inputs)
            loss = loss_function(y_hat, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            print("\nRunning Loss: ", running_loss)


# Creating a sample dataset
X = np.linspace(0, 20, num=20)
y = X + np.sin(X) * 2 + np.random.normal(size=X.shape)
training_loader = Simple1DRegressionDataset(X, y)

in_features = 1
out_features = 1
model = nn.Linear(in_features, out_features)
loss_func = nn.MSELoss()

device = torch.device("cpu")
train_simple_network(model, loss_func, training_loader, 20, device)

with torch.no_grad():
    y_pred = model(torch.tensor(X.reshape(-1, 1), device=device, dtype=torch.float32)).numpy()

sns.scatterplot(x=X, y=y, color="blue", label="Data")
sns.lineplot(x=X, y=y_pred.ravel(), color="red", label="Linear Model")
plt.show()
