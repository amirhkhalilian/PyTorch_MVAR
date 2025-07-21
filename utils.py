import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_predictions(y_true, y_pred=None,
                     n_plots=None,
                     title="Predictions vs Ground Truth"):
    """
    y_true, y_pred: numpy array of shape (num_samples, N)
    n_plots: number of variables to plot (max N)
    """
    N = y_true.shape[1]
    if n_plots is None:
        n_plots = N
    n_plots = min(n_plots, N)

    fig, axes = plt.subplots(n_plots, 1,
                             figsize=(10, 2.5 * n_plots),
                             sharex=True)
    if n_plots == 1:
        axes = [axes]

    for i in range(n_plots):
        axes[i].plot(y_true[:, i], label="Ground Truth", color="black")
        if y_pred is not None:
            axes[i].plot(y_pred[:, i], '--', label="Prediction",
                         color="red", alpha=0.75)
        axes[i].set_ylabel(f"Var {i}")
        axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_mat(A, n_plots=None):
    """
    A: list of matrices to plot
    use for covariance matrix plotting or connectivity
    """
    N = len(A)
    if n_plots is None:
        n_plots = N
    n_plots = min(n_plots, N)

    fig, axes = plt.subplots(n_plots, 1,
                             figsize=(10, 2.5 * n_plots),
                             sharex=True)
    if n_plots == 1:
        axes = [axes]

    for i in range(n_plots):
        vmax = np.max(np.abs(A[i]))
        sns.heatmap(A[i],
                    annot=True,
                    ax=axes[i],
                    fmt=".3f",
                    cmap='coolwarm',
                    center=0,
                    cbar=True,
                    square=True,
                    vmin = -vmax,
                    vmax = vmax)
    plt.tight_layout()
    plt.show()

def get_mvar_weights(model_num=0, N=5):
    '''
    get different example models given model
    '''
    if model_num == 0:
        # Initialize 3 coefficient matrices for lags 1, 2, and 3
        # model from Fig2 PDC paper
        A1 = np.zeros((5, 5))  # x[n-1]
        A2 = np.zeros((5, 5))  # x[n-2]
        A3 = np.zeros((5, 5))  # x[n-3]
        sqrt2 = np.sqrt(2)
        # Fill A1 (lag-1)
        A1[0, 0] = 0.95 * sqrt2           # x0[n-1] → x0[n]
        A1[3, 3] = 0.25 * sqrt2           # x3[n-1] → x3[n]
        A1[3, 4] = 0.25 * sqrt2           # x4[n-1] → x3[n]
        A1[4, 3] = -0.25 * sqrt2          # x3[n-1] → x4[n]
        A1[4, 4] = 0.25 * sqrt2           # x4[n-1] → x4[n]
        # Fill A2 (lag-2)
        A2[0, 0] = -0.9025                # x0[n-2] → x0[n]
        A2[1, 0] = 0.5                    # x0[n-2] → x1[n]
        A2[3, 0] = -0.5                   # x0[n-2] → x3[n]
        # Fill A3 (lag-3)
        A3[2, 0] = -0.4                   # x0[n-3] → x2[n]
        return [A1,A2,A3]
    elif model_num == 1:
        A1 = np.array([
            [0.5, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.4, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.5]])
        A2 = np.eye(N) * 0.1
        return [A1,A2]
    elif model_num == 2:
        np.random.seed(42)
        A1 = 0.2 * np.eye(N) + 0.3 * np.random.randn(N, N)
        A2 = 0.1 * np.random.randn(N, N)
        return [A1,A2]


def generate_mvar_data_from_adj(A, T, x0 = [0.1,0.1,0.1], noise_level=1.0, burn_in=0):
    """
    Generate MVAR data given coefficient matrices A.
    A: list of length p, each element is an (N x N) matrix for lag-k.
    T: number of time points to return (after burn-in).
    """
    p = len(A)
    N = A[0].shape[0]
    total_T = T + burn_in
    X = np.zeros((total_T, N))
    X[0,0] = x0[0]
    X[1,0] = x0[1]
    X[2,0] = x0[2]
    noise = np.random.randn(total_T, N) * noise_level
    for t in range(p, total_T):
        x_t = sum(A[k] @ X[t - k - 1] for k in range(p))
        X[t] = x_t + noise[t]
    return X[burn_in:]

def mvar_trial_generator(t_trial = 200,
                         num_trial = 128,
                         init_mode = 'random',
                         noise_level = 1e-4,
                         burn_in = 0,
                         model_num = 0):
    data = np.array([]).reshape(0,t_trial,5) # final np array of num_trial x t_trial x node
    A = get_mvar_weights(model_num)
    for i in range(num_trial):
        if init_mode == 'random':
            x0 = np.random.randn(3)
        else:
            x0 = [0.1, 0.1, 0.1]
        x = generate_mvar_data_from_adj(A,
                                        T = t_trial,
                                        x0 = x0,
                                        noise_level = noise_level,
                                        burn_in = burn_in)[None,...] # x in shape (1 x T x 5)
        data = np.concatenate((data, x), axis=0)
    return data

def _generate_custom_mvar_plotter():
    A = get_mvar_weights(model_num=0)
    sample_data = generate_mvar_data_from_adj(A, T=500, noise_level=0.01)
    plot_predictions(sample_data)


if __name__ == "__main__":
    _generate_custom_mvar_plotter()

