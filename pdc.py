import numpy as np
import matplotlib.pyplot as plt

def compute_A_bar(A_matrices, freqs):
    """
    Compute Ā(f) = I - sum_k A(k) * exp(-j*2*pi*f*k)
    Returns: array of shape (F, M, M)
    """
    p = len(A_matrices)
    M = A_matrices[0].shape[0]
    F = len(freqs)

    A_bar = np.zeros((F, M, M), dtype=np.complex128)
    I = np.eye(M, dtype=np.complex128)

    for fi, f in enumerate(freqs):
        A_f = I.copy()
        for k, A_k in enumerate(A_matrices):
            # k+1 because lag indexing starts from 1
            A_f -= A_k * np.exp(-2j * np.pi * f * (k + 1))
        A_bar[fi] = A_f

    return A_bar

def compute_gPDC(A_matrices, Phi, freqs):
    """
    Compute gPDC for all signal pairs over frequency.
    A_matrices: list of A(k), each (M, M)
    Phi: inverse covariance matrix (M, M)
    freqs: list or array of normalized frequencies [0, 0.5]

    Returns: gPDC of shape (F, M, M) where
             gPDC[f, i, j] = gPDC from j to i at freq f
    """
    A_bar = compute_A_bar(A_matrices, freqs)  # shape (F, M, M)
    F, M, _ = A_bar.shape
    gPDC = np.zeros((F, M, M))

    phi_diag = np.real(np.diag(Phi))  # shape (M,)

    for fi in range(F):
        A_f = A_bar[fi]

        A_abs2 = np.abs(A_f)**2  # |Ā_ij(f)|^2, shape (M, M)
        for j in range(M):
            denom = np.sum(phi_diag * A_abs2[:, j])  # ∑_m φ_mm |Ā_mj(f)|²
            if denom < 1e-12:
                continue  # Avoid division by 0
            for i in range(M):
                num = phi_diag[i] * A_abs2[i, j]
                gPDC[fi, i, j] = num / denom

    return gPDC

def compute_bPDC(A_matrices, Phi, freqs):
    """
    Compute bPDC(f) for all pairs (i, j) using Faes & Nollo Eq. (18).
    A_matrices: list of A(k) matrices, each of shape (M, M)
    Phi: inverse error covariance matrix, shape (M, M)
    freqs: array of normalized frequencies
    Returns: bPDC matrix of shape (M, M),
    broadband version integrated over frequencies
    """
    M = A_matrices[0].shape[0]
    A_bar = compute_A_bar(A_matrices, freqs)

    # Compute bPDC_ij(f) at each freq and average (integration)
    bPDC_freq = np.zeros((len(freqs), M, M))
    for idx, A_f in enumerate(A_bar):
        for i in range(M):
            for j in range(M):
                # Compute numerator
                a_ij = A_f[i, j]
                phi_ii = Phi[i, i]
                num = phi_ii * np.abs(a_ij)**2

                # Compute denominator
                denom = 0
                for m in range(M):
                    a_mj = A_f[m, j]
                    phi_mm = Phi[m, m]
                    denom += phi_mm * np.abs(a_mj)**2

                bPDC_freq[idx, i, j] = num / denom if denom > 0 else 0.0
    # # Integrate over frequency (average since uniform spacing assumed)
    # bPDC = np.mean(bPDC_freq, axis=0)
    return bPDC_freq

def plot_PDC_grid(gPDC, freqs, figsize=(12, 10), vmin=0.0, vmax=1.0):
    """
    gPDC: array of shape (F, M, M)
    freqs: array of shape (F,)
    Produces M x M grid of plots showing gPDC from j -> i.
    """
    F, M, _ = gPDC.shape
    fig, axes = plt.subplots(M, M, figsize=figsize, sharex=True, sharey=True)

    for i in range(M):
        for j in range(M):
            ax = axes[i, j]
            ax.plot(freqs, gPDC[:, i, j], color='black')
            ax.set_ylim(vmin, vmax)
            if i == M - 1:
                ax.set_xlabel(f"j={j}")
            if j == 0:
                ax.set_ylabel(f"i={i}")
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.ylim([-0.05,1.05])

    plt.suptitle("gPDC from j → i over frequency", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__=="__main__":
    from utils import get_mvar_weights
    M = 5
    A = get_mvar_weights(1)
    Phi = np.eye(M)
    freqs = np.linspace(0, 0.5, 256)
    gPDC = compute_gPDC(A, Phi, freqs)
    plot_PDC_grid(gPDC, freqs)
    bPDC = compute_bPDC(A, Phi, freqs)
    plot_PDC_grid(bPDC, freqs)
