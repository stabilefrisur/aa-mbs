import numpy as np
import matplotlib.pyplot as plt


def generate_gbm(
    mu: float, sigma: float, X0: float, T: float, N: int, seed: int = None
) -> np.ndarray:
    """
    Generate a geometric Brownian motion (GBM) time series using the Euler-Maruyama method.

    Parameters:
    - mu: Drift coefficient.
    - sigma: Volatility coefficient.
    - X0: Initial value.
    - T: Total time.
    - N: Number of time steps.
    - seed: Random seed for reproducibility (default: None).

    Returns:
    - t: Array of time steps.
    - X: Array of GBM values.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t = np.linspace(0, T, N + 1)
    X = np.zeros(N + 1)
    X[0] = X0

    for n in range(N):
        Z = np.random.normal()
        X[n + 1] = X[n] + mu * X[n] * dt + sigma * X[n] * np.sqrt(dt) * Z

    return t, X


def generate_mean_reversion(
    mu: float, theta: float, sigma: float, X0: float, T: float, N: int, seed: int = None
) -> np.ndarray:
    """
    Generate a mean reversion process time series using the Euler-Maruyama method.

    Parameters:
    - mu: Long-term mean level.
    - theta: Speed of reversion.
    - sigma: Volatility coefficient.
    - X0: Initial value.
    - T: Total time.
    - N: Number of time steps.
    - seed: Random seed for reproducibility (default: None).

    Returns:
    - t: Array of time steps.
    - X: Array of mean reversion process values.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t = np.linspace(0, T, N + 1)
    X = np.zeros(N + 1)
    X[0] = X0

    for n in range(N):
        Z = np.random.normal()
        X[n + 1] = X[n] + theta * (mu - X[n]) * dt + sigma * np.sqrt(dt) * Z

    return t, X


def generate_correlated_mean_reversion(
    mu1: float,
    theta1: float,
    sigma1: float,
    X01: float,
    mu2: float,
    theta2: float,
    sigma2: float,
    X02: float,
    T: float,
    N: int,
    rho: float,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate two correlated mean reversion processes using the Euler-Maruyama method.

    Parameters:
    - mu1, mu2: Long-term mean levels for the two processes.
    - theta1, theta2: Speed of reversion for the two processes.
    - sigma1, sigma2: Volatility coefficients for the two processes.
    - X01, X02: Initial values for the two processes.
    - T: Total time.
    - N: Number of time steps.
    - rho: Correlation coefficient between the two processes.
    - seed: Random seed for reproducibility (default: None).

    Returns:
    - t: Array of time steps.
    - X1: Array of values for the first mean reversion process.
    - X2: Array of values for the second mean reversion process.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t = np.linspace(0, T, N + 1)
    X1 = np.zeros(N + 1)
    X2 = np.zeros(N + 1)
    X1[0] = X01
    X2[0] = X02

    # Cholesky decomposition for correlated normal variables
    L = np.linalg.cholesky([[1, rho], [rho, 1]])

    for n in range(N):
        Z = np.random.normal(size=2)
        Z_corr = L @ Z
        X1[n + 1] = (
            X1[n] + theta1 * (mu1 - X1[n]) * dt + sigma1 * np.sqrt(dt) * Z_corr[0]
        )
        X2[n + 1] = (
            X2[n] + theta2 * (mu2 - X2[n]) * dt + sigma2 * np.sqrt(dt) * Z_corr[1]
        )

    return t, X1, X2


def plot_process(
    processes: dict[str, tuple[np.ndarray, np.ndarray]], title: str
) -> None:
    """
    Plot multiple stochastic processes in the same chart for comparison.

    Parameters:
    - processes: Dictionary where keys are process names and values are tuples of (time steps, process values).
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    for name, (t, X) in processes.items():
        plt.plot(t, X, label=name)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Example usage for GBM
    mu_gbm = 1.0
    sigma_gbm = 0.2
    X0_gbm = mu_gbm
    T_gbm = 1.0
    N_gbm = 1000
    seed_gbm = 42

    t_gbm, X_gbm = generate_gbm(mu_gbm, sigma_gbm, X0_gbm, T_gbm, N_gbm, seed_gbm)

    # Example usage for Mean Reversion
    mu_mr = 1.0
    theta_mr = 0.15
    sigma_mr = 0.2
    X0_mr = mu_mr
    T_mr = 1.0
    N_mr = 1000
    seed_mr = 42

    t_mr, X_mr = generate_mean_reversion(
        mu_mr, theta_mr, sigma_mr, X0_mr, T_mr, N_mr, seed_mr
    )

    # Example usage for Correlated Mean Reversion
    mu1 = 1.0
    theta1 = 0.15
    sigma1 = 0.2
    X01 = mu1
    mu2 = 1.0
    theta2 = 0.15
    sigma2 = 0.2
    X02 = mu2
    T_corr = 1.0
    N_corr = 1000
    rho = 0.8
    seed_corr = 42

    t_corr, X1_corr, X2_corr = generate_correlated_mean_reversion(
        mu1,
        theta1,
        sigma1,
        X01,
        mu2,
        theta2,
        sigma2,
        X02,
        T_corr,
        N_corr,
        rho,
        seed_corr,
    )

    # Plot all processes for comparison
    processes = {
        'Geometric Brownian Motion': (t_gbm, X_gbm),
        'Mean Reversion Process': (t_mr, X_mr),
        'Correlated Mean Reversion Process 1': (t_corr, X1_corr),
        'Correlated Mean Reversion Process 2': (t_corr, X2_corr),
    }
    plot_process(processes, 'Comparison of Stochastic Processes')
