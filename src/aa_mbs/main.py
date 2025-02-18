import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from typing import Tuple, List
from mock_data import generate_mock_data


class JointReversionModel:
    def __init__(
        self,
        kappa: float,
        lambda_: float,
        gamma: List[float],
        beta: List[float],
        sigma_O: np.ndarray,
        sigma_C: float,
        dt: float,
    ):
        self.kappa = kappa
        self.lambda_ = lambda_
        self.gamma = gamma
        self.beta = beta
        self.sigma_O = sigma_O
        self.sigma_C = sigma_C
        self.dt = dt

    # ruff: noqa
    def simulate(
        self,
        S_OAS_init: float,
        C_init: float,
        S_OAS_inf: float,
        sigma_r: np.ndarray,
        nu_r: np.ndarray,
        enable_convexity: bool = True,
        enable_volatility: bool = True,
        steps: int = 252,
        seed: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)

        S_OAS = np.zeros(steps)
        C = np.zeros(steps)
        S_OAS[0] = S_OAS_init
        C[0] = C_init

        for t in range(1, steps):
            Z_O = np.random.normal()
            Z_C = np.random.normal()

            if enable_convexity:
                gamma_term = self.gamma[0] * float(C[t - 1])
                beta_term = self.beta[0] * float(S_OAS[t - 1])
            else:
                gamma_term = 0
                beta_term = 0

            if enable_volatility:
                gamma_term += (
                    self.gamma[1] * float(sigma_r[t - 1]) 
                    + self.gamma[2] * float(nu_r[t - 1])
                )
                beta_term += (
                    self.beta[1] * float(sigma_r[t - 1]) 
                    + self.beta[2] * float(nu_r[t - 1])
                )

            S_OAS[t] = (
                float(S_OAS[t - 1])
                - self.kappa * (float(S_OAS[t - 1]) - S_OAS_inf - gamma_term) * self.dt
                + self.sigma_O[t] * np.sqrt(self.dt) * Z_O
            )

            C[t] = (
                float(C[t - 1])
                - self.lambda_ * (float(C[t - 1]) - beta_term) * self.dt
                + self.sigma_C * np.sqrt(self.dt) * Z_C
            )

        return S_OAS, C
    # ruff: enable


def estimate_parameters(
    data: pd.DataFrame,
    initial_guess: Tuple[float, float] = (0.05, 0.2)
) -> Tuple[float, float, List[float], List[float], np.ndarray, float]:
    # Extract historical data
    S_OAS = data['OAS']
    C = data['Convexity']
    sigma_r = data['Sigma_r']
    nu_r = data['Nu_r']

    # Mean Reversion Parameter Estimation using OLS
    X_OAS = np.column_stack(
        [np.ones(len(S_OAS) - 1), S_OAS[:-1], C[:-1], sigma_r[:-1], nu_r[:-1]]
    )
    y_OAS = S_OAS[1:]
    kappa, gamma_0, gamma_1, gamma_2 = np.linalg.lstsq(X_OAS, y_OAS, rcond=None)[0][1:]

    X_C = np.column_stack(
        [np.ones(len(C) - 1), C[:-1], S_OAS[:-1], sigma_r[:-1], nu_r[:-1]]
    )
    y_C = C[1:]
    lambda_, beta_0, beta_1, beta_2 = np.linalg.lstsq(X_C, y_C, rcond=None)[0][1:]

    # Residual Variance Calibration
    residuals_OAS = (
        y_OAS - kappa * (
            S_OAS[:-1] 
            - gamma_0 * C[:-1] 
            - gamma_1 * sigma_r[:-1] 
            - gamma_2 * nu_r[:-1]
        )
    )
    residuals_C = (
        y_C - lambda_ * (
            C[:-1] 
            - beta_0 * S_OAS[:-1] 
            - beta_1 * sigma_r[:-1] 
            - beta_2 * nu_r[:-1]
        )
    )

    def variance_function(params):
        sigma_O_0, delta = params
        return np.sum((residuals_OAS**2 - (sigma_O_0**2 + delta * C[:-1] ** 2)) ** 2)

    sigma_O_0, delta = minimize(variance_function, initial_guess).x

    sigma_O = np.sqrt(sigma_O_0**2 + delta * C**2).values
    sigma_C = np.std(residuals_C)

    return (
        kappa,
        lambda_,
        [gamma_0, gamma_1, gamma_2],
        [beta_0, beta_1, beta_2],
        sigma_O,
        sigma_C,
    )


def stationarity_tests(
    data: pd.DataFrame,
) -> Tuple[
    Tuple[float, float, int, int, dict, float],
    Tuple[float, float, int, int, dict, float],
]:
    S_OAS = data['OAS']
    C = data['Convexity']

    adf_OAS = adfuller(S_OAS)
    adf_C = adfuller(C)

    return adf_OAS, adf_C


def monte_carlo_simulation(
    model: JointReversionModel,
    S_OAS_init: float,
    C_init: float,
    S_OAS_inf: float,
    sigma_r: np.ndarray,
    nu_r: np.ndarray,
    num_paths: int,
    steps: int,
    seed: int = None,
) -> Tuple[float, float, List[np.ndarray], List[np.ndarray]]:
    results_OAS = []
    results_C = []
    paths_OAS = []
    paths_C = []

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_paths):
        S_OAS, C = model.simulate(
            S_OAS_init,
            C_init,
            S_OAS_inf,
            sigma_r,
            nu_r,
            enable_convexity=True,
            enable_volatility=True,
            steps=steps,
        )
        results_OAS.append(S_OAS[-1])
        results_C.append(C[-1])
        paths_OAS.append(S_OAS)
        paths_C.append(C)

    expected_value_OAS = np.mean(results_OAS)
    expected_value_C = np.mean(results_C)
    return expected_value_OAS, expected_value_C, paths_OAS, paths_C

def main() -> None:
    # Set seed for reproducibility
    seed = 42

    # Load historical data
    data = generate_mock_data(seed=seed)

    # Stationarity tests
    adf_OAS, adf_C = stationarity_tests(data)
    print(f'ADF Test for OAS: {adf_OAS}')
    print(f'ADF Test for Convexity: {adf_C}')

    # Estimate model parameters
    kappa, lambda_, gamma, beta, sigma_O, sigma_C = estimate_parameters(data)

    # Simulation parameters
    dt = 0.01
    steps = 252  # Assuming 252 trading days in a year
    S_OAS_init = float(data['OAS'].iloc[-1])
    C_init = float(data['Convexity'].iloc[-1])
    S_OAS_inf = float(np.mean(data['OAS']))
    sigma_r = data['Sigma_r'].values
    nu_r = data['Nu_r'].values
    num_paths = 1000  # Number of Monte Carlo paths

    # Create model instance
    model = JointReversionModel(kappa, lambda_, gamma, beta, sigma_O, sigma_C, dt)

    # Perform Monte Carlo simulation
    expected_value_OAS, expected_value_C, paths_OAS, paths_C = monte_carlo_simulation(
        model,
        S_OAS_init,
        C_init,
        S_OAS_inf,
        sigma_r,
        nu_r,
        num_paths,
        steps,
        seed
    )

    print(f'Expected value of OAS in one year: {expected_value_OAS}')
    print(f'Expected value of Convexity in one year: {expected_value_C}')

    # Plot historical data and Monte Carlo paths
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot historical OAS data

    paths_OAS = pd.DataFrame(paths_OAS).T
    paths_OAS.index = pd.date_range(start=data.index[-1], periods=steps, freq='B')
    OAS = pd.concat([data['OAS'], paths_OAS], axis=1)
    axs[0].plot(OAS, color='blue', alpha=0.1)
    axs[0].plot(data['OAS'], color='black', label='OAS')
    axs[0].axhline(y=expected_value_OAS, color='red', linestyle='--', label='Expected Value')
    axs[0].axhline(y=data['OAS'].mean(), color='green', linestyle='--', label='Historical Mean')
    axs[0].set_title('Monte Carlo Simulation of OAS')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('OAS')
    axs[0].legend()

    # Plot historical Convexity data
    paths_C = pd.DataFrame(paths_C).T
    paths_C.index = pd.date_range(start=data.index[-1], periods=steps, freq='B')
    C = pd.concat([data['Convexity'], paths_C], axis=1)
    axs[1].plot(C, color='green', alpha=0.1)
    axs[1].plot(data['Convexity'], color='black', label='Convexity')
    axs[1].axhline(y=expected_value_C, color='red', linestyle='--', label='Expected Value')
    axs[1].axhline(y=data['Convexity'].mean(), color='green', linestyle='--', label='Historical Mean')
    axs[1].set_title('Monte Carlo Simulation of Convexity')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Convexity')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
