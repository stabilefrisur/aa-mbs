import numpy as np
import matplotlib.pyplot as plt

def generate_gbm(mu: float, sigma: float, X0: float, T: float, N: int, seed: int = None) -> np.ndarray:
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
    t = np.linspace(0, T, N+1)
    X = np.zeros(N+1)
    X[0] = X0

    for n in range(N):
        Z = np.random.normal()
        X[n+1] = X[n] + mu * X[n] * dt + sigma * X[n] * np.sqrt(dt) * Z

    return t, X

def generate_mean_reversion(mu: float, theta: float, sigma: float, X0: float, T: float, N: int, seed: int = None) -> np.ndarray:
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
    t = np.linspace(0, T, N+1)
    X = np.zeros(N+1)
    X[0] = X0

    for n in range(N):
        Z = np.random.normal()
        X[n+1] = X[n] + theta * (mu - X[n]) * dt + sigma * np.sqrt(dt) * Z

    return t, X

def plot_process(processes: dict[str, tuple[np.ndarray, np.ndarray]], title: str) -> None:
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

    t_mr, X_mr = generate_mean_reversion(mu_mr, theta_mr, sigma_mr, X0_mr, T_mr, N_mr, seed_mr)

    # Plot both processes for comparison
    processes = {
        'Geometric Brownian Motion': (t_gbm, X_gbm),
        'Mean Reversion Process': (t_mr, X_mr)
    }
    plot_process(processes, 'Comparison of Stochastic Processes')
