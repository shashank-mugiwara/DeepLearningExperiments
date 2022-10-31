import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml


class SimpleDataset(Dataset):

    def __init__(self, X, y):
        super(SimpleDataset, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X_key = 'pixel' + str(index+1)
        inputs = torch.tensor(self.X[X_key], dtype=torch.float32)
        targets = torch.tensor(int(self.y[index]), dtype=torch.int64)
        return inputs, targets

    def __len__(self):
        return self.X.shape[0]


# Load the MNIST Dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
dataset = SimpleDataset(X, y)


# Returns - 7000
print('Total Number of data rows: {}'.format(len(dataset)))

# Returns - 784
example, label = dataset[0]
print('Total Number of features: {}'.format(example.shape))

# Returns - tensor(5)
print('Label of index 0: {}'.format(label))