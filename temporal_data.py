import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

class TemporalDataset(Dataset):
    def __init__(self, data, p, q):
        """
        data: np.ndarray of shape (B, T, N)
        p: number of past time steps for input
        q: number of future time steps for target
        """
        self.X, self.Y = self._create_samples(data, p, q)

    def _create_samples(self, data, p, q):
        B, T, N = data.shape
        X, Y = [], []
        for b in range(B):
            for t in range(p, T - q + 1):
                x_t = data[b, t - p:t][::-1].reshape(-1)
                y_t = data[b, t:t + q].reshape(-1)
                X.append(x_t)
                Y.append(y_t)
        X = torch.tensor(np.stack(X), dtype=torch.float32)
        Y = torch.tensor(np.stack(Y), dtype=torch.float32)
        return X, Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class TemporalDataModule(L.LightningDataModule):
    def __init__(self, data, p, q, batch_size=8, val_split=0.2, shuffle=True):
        """
        data: np.ndarray of shape (B, T, N)
        """
        super().__init__()
        self.data = data
        self.p = p
        self.q = q
        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle = shuffle

    def setup(self, stage=None):
        dataset = TemporalDataset(self.data, self.p, self.q)
        n_total = len(dataset)
        n_val = int(self.val_split * n_total)
        n_train = n_total - n_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

