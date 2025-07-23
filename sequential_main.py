import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from utils import mvar_trial_generator
from temporal_data import SequentialTemporalDataModule
from temporal_data import SequentialTemporalDataset
from mvar import sequential_mvar
from utils import plot_predictions
from utils import plot_mat

# setup model params
p = 3
q_train = 1
q_test = 6

def main():
    # ------------------------
    # Data
    # ------------------------
    data = mvar_trial_generator(
        t_trial=100,
        num_trial=100,
        noise_level=1e-1,
        model_num = 0)
    data_module = SequentialTemporalDataModule(
        data,
        p=p,
        q=q_train,
        batch_size=64,
        shuffle=True)
    data_module.setup()
    print(data.shape)

    # ------------------------
    # Model
    # ------------------------
    model = sequential_mvar(
        n_vars=data.shape[-1],
        p=p,
        q=q_train,
        lr=1e-3,
        l1_ratio=0.0,
        mode = 'full',
        penalty = None )

    # ------------------------
    # Trainer
    # ------------------------
    trainer = Trainer(
        max_epochs=50,
        logger=CSVLogger('logs', name='mvar'),
        callbacks=[ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")],
        accelerator="auto",
    )

    # ------------------------
    # Training
    # ------------------------
    trainer.fit(model, datamodule=data_module)

    # ------------------------
    # Evaluation
    # ------------------------
    # Plot weights
    model.plot_weights()
    #
    data_test = mvar_trial_generator(
        t_trial=100,
        num_trial=3,
        noise_level=1e-1,
        model_num = 0)
    test_dataloader = DataLoader(
        SequentialTemporalDataset(data_test, p, q_test),
        batch_size=1,
        shuffle=False)
    model.hparams.q = q_test
    results = trainer.predict(model, dataloaders=test_dataloader)
    # Unpack predictions
    y_pred = torch.cat([y_hat for y_hat, _ in results], dim=0)
    y_true = torch.cat([y for _, y in results], dim=0)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    print(y_true.shape)
    print(y_pred.shape)
    test_mse = np.sqrt(np.mean((y_true - y_pred)**2))
    test_nmse = test_mse / np.std(y_true)
    test_mae = np.max(np.abs(y_true - y_pred))
    print(f"test rmse: {test_mse:1.4e},\
            test nrmse: {test_nmse:1.4e},\
            test mae: {test_mae:1.4e}")
    for i in range(q_test):
        y_t_i = y_true[:,i,:].squeeze()
        y_p_i = y_pred[:,i,:].squeeze()
        test_mse = np.sqrt(np.mean((y_t_i - y_p_i)**2))
        test_nmse = test_mse / np.std(y_t_i)
        test_mae = np.max(np.abs(y_t_i - y_p_i))
        print(f"lag: {i}, test rmse: {test_mse:1.4e},\
                test nrmse:{test_nmse:1.4e},\
                test mae: {test_mae:1.4e}")
        plot_predictions(y_t_i, y_p_i)
        error_cov = np.cov(y_t_i.T-y_p_i.T)
        plot_mat([error_cov])


if __name__ == "__main__":
    main()
