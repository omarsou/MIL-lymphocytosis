from torch.utils import data
from PIL import Image


class LymphoDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, filenames, transform=None):
        "Initialization"
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.filenames)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        filename = self.filenames[index]
        X = Image.open(filename)

        if self.transform:
            X = self.transform(X)     # transform
        return X


class InferLymphoDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, filenames, names, transform=None):
        "Initialization"
        self.filenames = filenames
        self.names = names
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.filenames)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        filename = self.filenames[index]
        name = [self.names[index]]
        X = Image.open(filename)

        if self.transform:
            X = self.transform(X)     # transform
        # Convert everything to tensor

        return X, name
