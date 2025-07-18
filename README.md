# Multivariate Autoregressive (MVAR) Model with PyTorch Lightning

This repository provides an implementation of a multivariate autoregressive (MVAR) model using PyTorch Lightning for learning temporal dependencies from synthetic time-series data. It includes data generation, training, evaluation, and visualization.

## Installation

Install the required packages:

```bash
pip install torch pytorch-lightning matplotlib numpy
```

## Usage
Run the main training and evaluation script:

```bash
python main.py
```

## Repository Structure
main.py – main training and evaluation script

mvar.py – MVAR model implementation

temporal_data.py – data module and dataset class

utils.py – synthetic data generator and plotting utilities

