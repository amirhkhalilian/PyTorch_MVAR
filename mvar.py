import torch
import torch.nn as nn
import lightning
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn.functional as F

class DiagonalLinear(nn.Module):
    def __init__(self, n_vars, p):
        super().__init__()
        self.n_vars = n_vars
        self.p = p
        # Only learn self-connections across lags (diagonal in each lag block)
        # this is not an efficient way to implement this but it works well
        # with how the data is presented for other cases
        self.weight = nn.Parameter(torch.zeros(n_vars, n_vars * p))
        self.register_buffer("mask", self._create_mask())

    def _create_mask(self):
        mask = torch.zeros(self.n_vars, self.n_vars * self.p)
        for i in range(self.n_vars):
            for lag in range(self.p):
                mask[i, lag * self.n_vars + i] = 1.0
        return mask

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return torch.matmul(x, masked_weight.T)

class OffDiagonalLinear(nn.Module):
    def __init__(self, n_vars, p):
        super().__init__()
        self.n_vars = n_vars
        self.p = p
        # the oposite of the DiagonalLinear
        self.weight = nn.Parameter(torch.zeros(n_vars, n_vars * p))
        self.register_buffer("mask", self._create_mask())

    def _create_mask(self):
        mask = torch.ones(self.n_vars, self.n_vars * self.p)
        for i in range(self.n_vars):
            for lag in range(self.p):
                # Zero out the self-connections
                mask[i, lag * self.n_vars + i] = 0.0
        return mask

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return torch.matmul(x, masked_weight.T)


class mvar(lightning.LightningModule):
    def __init__(self, n_vars, p, lr=1e-3, l1_ratio=0.0, mode='full', penalty=None):
        super().__init__()
        self.save_hyperparameters()
        if mode=='full':
            # full MVAR: self+others
            self.fc = nn.Linear(n_vars*p, n_vars, bias=False)
        elif mode=='diag':
            # diag only
            self.fc = DiagonalLinear(n_vars, p)
        elif mode=='off_diag':
            # off diag only
            self.fc = OffDiagonalLinear(n_vars, p)
        else:
            raise ValueError('accepted modes are full, diag, off_diag')
        nn.init.zeros_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)

    def _compute_penalty(self):
        W = self.fc.weight
        n_vars = self.hparams.n_vars
        p = self.hparams.p
        penalty = self.hparams.penalty
        if penalty is None:
            return 0.0
        else:
            if penalty=="l1_loss":
                penalty_loss = self.hparams.l1_ratio * W.abs().mean()
            elif penalty=="group_loss":
                W = W.view(n_vars, p, n_vars).permute(0, 2, 1)
                ll = torch.norm(W,dim=-1).abs().mean()
                penalty_loss = self.hparams.l1_ratio * ll
            elif penalty=="sc_group_loss":
                W = W.view(n_vars, p, n_vars).permute(0, 2, 1)
                off_diag_mask = ~torch.eye(n_vars, dtype=torch.bool, device=W.device)
                ll = torch.norm(W,dim=-1)
                ll = ll[off_diag_mask].abs().mean()
                penalty_loss = self.hparams.l1_ratio * ll
            else:
                raise ValueError('unsupported penalty type')
            return penalty_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = nn.functional.mse_loss(y_hat,y)
        penalty_loss = self._compute_penalty()
        loss = mse_loss +  penalty_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x), y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def plot_weights(self):
        n_vars = self.hparams.n_vars
        p = self.hparams.p
        with torch.no_grad():
            W = self.fc.weight.cpu().numpy()
        fig, axes = plt.subplots(p, 1, figsize=(6, 3 * p), sharex=True)
        if p == 1:
            axes = [axes]
        vmax = np.max(np.abs(W))

        for lag in range(p):
            ax = axes[lag]
            A_lag = W[:, lag * n_vars:(lag + 1) * n_vars]
            sns.heatmap(A_lag,
                        annot=True,
                        fmt=".2f",
                        cmap='coolwarm',
                        center=0,
                        cbar=True,
                        ax=ax,
                        square=True,
                        vmin = -vmax,
                        vmax = vmax)
            ax.set_title(f"Lag {lag + 1}")
            ax.set_ylabel("Output variable")
            ax.set_xticks(range(n_vars))
            ax.set_yticks(range(n_vars))
            ax.set_xticklabels([f"x[{j}]" for j in range(n_vars)])
            ax.set_yticklabels([f"x[{i}]" for i in range(n_vars)])
        axes[-1].set_xlabel("Input variable")
        plt.tight_layout()
        plt.show()

class sequential_mvar(mvar):
    def __init__(self, n_vars, p, q, lr=1e-3, l1_ratio=0.0, mode='full', penalty=None):
        super().__init__(n_vars=n_vars, p=p, lr=lr, l1_ratio=l1_ratio, mode=mode, penalty=penalty)
        self.hparams.q = q

    def forward(self, x):
        '''
        x should be of shape [B, p, N]
        gives predictions of shape [B, q, N]
        '''
        preds = []
        B = x.shape[0]
        x_input = x
        for _ in range(self.hparams.q):
            y_t = self.fc(torch.flip(x_input,dims=(1,)).reshape(B,-1)).unsqueeze(1)
            preds.append(y_t)
            x_input = torch.cat([x_input[:, 1:, :], y_t], dim=1)
        return torch.cat(preds, dim=1)

class conv_mvar(mvar):
    def __init__(self, n_vars, p, lr=1e-3, l1_ratio=0.0, mode='full', penalty=None):
        super().__init__(n_vars, p)
        self.save_hyperparameters()
        if mode=='full':
            # full MVAR: self+others
            self.fc = nn.Conv1d(n_vars, n_vars, kernel_size=p, bias=False)
        elif mode=='diag':
            # diag only
            self.fc = nn.Conv1d(n_vars, n_vars, kernel_size=p, groups=n_vars, bias=False)
        else:
            raise ValueError('accepted modes are full')
        nn.init.zeros_(self.fc.weight)

    def forward(self, x):
        x = x.transpose(1,2)
        return self.fc(x).transpose(1,2)[:,:-1,:]

    def _compute_penalty(self):
        W = self.fc.weight
        penalty = self.hparams.penalty
        if penalty is None:
            return 0.0
        else:
            if penalty=="l1_loss":
                penalty_loss = self.hparams.l1_ratio * W.abs().mean()
            elif penalty=="group_loss":
                ll = torch.norm(W,dim=-1).abs().mean()
                penalty_loss = self.hparams.l1_ratio * ll
            elif penalty=="sc_group_loss":
                off_diag_mask = ~torch.eye(self.hparams.n_vars,
                                           dtype=torch.bool, device=W.device)
                ll = torch.norm(W,dim=-1)
                ll = ll[off_diag_mask].abs().mean()
                penalty_loss = self.hparams.l1_ratio * ll
            else:
                raise ValueError('unsupported penalty type')
            return penalty_loss

    def plot_weights(self):
        n_vars = self.hparams.n_vars
        p = self.hparams.p
        with torch.no_grad():
            W = self.fc.weight.cpu().numpy()
        fig, axes = plt.subplots(p, 1, figsize=(6, 3 * p), sharex=True)
        if p == 1:
            axes = [axes]
        vmax = np.max(np.abs(W))
        for lag in range(p):
            ax = axes[lag]
            A_lag = W[:, :, lag].squeeze()
            sns.heatmap(A_lag,
                        annot=True,
                        fmt=".2f",
                        cmap='coolwarm',
                        center=0,
                        cbar=True,
                        ax=ax,
                        square=True,
                        vmin = -vmax,
                        vmax = vmax)
            ax.set_title(f"Lag {lag + 1}")
            ax.set_ylabel("Output variable")
            ax.set_xticks(range(n_vars))
            ax.set_yticks(range(n_vars))
            ax.set_xticklabels([f"x[{j}]" for j in range(n_vars)])
            ax.set_yticklabels([f"x[{i}]" for i in range(n_vars)])
        axes[-1].set_xlabel("Input variable")
        plt.tight_layout()
        plt.show()
