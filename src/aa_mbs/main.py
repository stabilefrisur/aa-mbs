import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from typing import Tuple, List
from mock_data import generate_training_data, generate_forward_data


class JointReversionModel:
    def __init__(
        self,
        kappa: float,
        lambda_: float,
        gamma: List[float],
        beta: List[float],
        sigma_O_0: float,
        delta: float,
        sigma_C: float,
        dt: float,
    ):
        self.kappa = kappa
        self.lambda_ = lambda_
        self.gamma = gamma
        self.beta = beta
        self.sigma_O_0 = sigma_O_0
        self.delta = delta
        self.sigma_C = sigma_C
        self.dt = dt

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)

        S_OAS = np.zeros(steps)
        C = np.zeros(steps)
        sigma_O = np.zeros(steps)
        S_OAS[0] = S_OAS_init
        C[0] = C_init
        sigma_O[0] = np.sqrt(self.sigma_O_0**2 + self.delta * C_init**2)

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
                + sigma_O[t - 1] * np.sqrt(self.dt) * Z_O
            )

            C[t] = (
                float(C[t - 1])
                - self.lambda_ * (float(C[t - 1]) - beta_term) * self.dt
                + self.sigma_C * np.sqrt(self.dt) * Z_C
            )

            sigma_O[t] = np.sqrt(self.sigma_O_0**2 + self.delta * C[t]**2)

        return S_OAS, C, sigma_O
    # ruff: enable


def estimate_parameters(
    S_OAS: np.ndarray,
    C: np.ndarray,
    sigma_r: np.ndarray,
    nu_r: np.ndarray,
    initial_guess: Tuple[float, float] = (0.05, 0.2)
) -> Tuple[float, float, List[float], List[float], float, float, float]:
    """Estimate the parameters of the joint reversion model using OLS.

    Args:
        S_OAS (np.ndarray): _description_
        C (np.ndarray): _description_
        sigma_r (np.ndarray): _description_
        nu_r (np.ndarray): _description_
        initial_guess (Tuple[float, float], optional): _description_. Defaults to (0.05, 0.2).

    Returns:
        Tuple[float, float, List[float], List[float], float, float, float]: _description_
    """
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

    sigma_C = np.std(residuals_C)

    return (
        kappa,
        lambda_,
        [gamma_0, gamma_1, gamma_2],
        [beta_0, beta_1, beta_2],
        sigma_O_0,
        delta,
        sigma_C,
    )


def stationarity_tests(
    S_OAS: pd.Series,
    C: pd.Series
) -> Tuple[
    Tuple[float, float, int, int, dict, float],
    Tuple[float, float, int, int, dict, float],
]:
    """Test the stationarity of the time series using the Augmented Dickey-Fuller test.

    Args:
        S_OAS (pd.Series): _description_
        C (pd.Series): _description_

    Returns:
        Tuple[ Tuple[float, float, int, int, dict, float], Tuple[float, float, int, int, dict, float], ]: _description_
    """
    adf_OAS = adfuller(S_OAS)
    adf_C = adfuller(C)

    return adf_OAS, adf_C


def monte_carlo_simulation(
    model: JointReversionModel,
    S_OAS_init: float,
    C_init: float,
    S_OAS_inf: float,
    sigma_r: float | np.ndarray,
    nu_r: float | np.ndarray,
    num_paths: int,
    steps: int,
    seed: int = None,
) -> Tuple[float, float, float, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    results_OAS = []
    results_C = []
    results_sigma_O = []
    paths_OAS = []
    paths_C = []
    paths_sigma_O = []

    sigma_r_series = np.full(steps, sigma_r) if isinstance(sigma_r, float) else sigma_r
    nu_r_series = np.full(steps, nu_r) if isinstance(nu_r, float) else nu_r

    if seed is not None:
        np.random.seed(seed)

    for i in range(num_paths):
        S_OAS, C, sigma_O = model.simulate(
            S_OAS_init,
            C_init,
            S_OAS_inf,
            sigma_r_series,
            nu_r_series,
            enable_convexity=True,
            enable_volatility=True,
            steps=steps,
        )
        results_OAS.append(S_OAS[-1])
        results_C.append(C[-1])
        results_sigma_O.append(sigma_O[-1])
        paths_OAS.append(S_OAS)
        paths_C.append(C)
        paths_sigma_O.append(sigma_O)

    expected_value_OAS = np.mean(results_OAS)
    expected_value_C = np.mean(results_C)
    expected_value_sigma_O = np.mean(results_sigma_O)
    return expected_value_OAS, expected_value_C, expected_value_sigma_O, paths_OAS, paths_C, paths_sigma_O

def main() -> None:
    # Set seed for reproducibility
    seed = 42

    # Define variable parameters
    zv_params = {'mu': 0.02, 'theta': 0.01, 'sigma': 0.005, 'X0': 0.02}
    oas_params = {'mu': 0.01, 'theta': 0.01, 'sigma': 0.005, 'X0': 0.01}
    sigma_r_params = {'mu': 0.02, 'theta': 0.01, 'sigma': 0.002, 'X0': 0.02}
    nu_r_params = {'mu': 0.01, 'theta': 0.01, 'sigma': 0.002, 'X0': 0.01}

    # Mock data simulation parameters
    train_start_date = '2013-01-01'
    train_end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    project_num_days = 252
    freq = 'B'

    # Generate training data
    zv_data, oas_data, sigma_r_data, nu_r_data = generate_training_data(
        zv_params, oas_params, sigma_r_params, nu_r_params,
        train_start_date, train_end_date, freq, seed
    )
    cvx_data = zv_data - oas_data

    # Update X0 for forward data
    sigma_r_params['X0'] = sigma_r_data[-1]
    nu_r_params['X0'] = nu_r_data[-1]

    # Generate forward data
    sigma_r_forward, nu_r_forward = generate_forward_data(
        sigma_r_params, nu_r_params, train_end_date, project_num_days, freq, seed
    )

    # Stationarity tests
    adf_OAS, adf_C = stationarity_tests(oas_data, cvx_data)
    print(f'ADF Test for OAS: {adf_OAS}')
    print(f'ADF Test for Convexity: {adf_C}')

    # Estimate model parameters
    kappa, lambda_, gamma, beta, sigma_O_0, delta, sigma_C = estimate_parameters(
        oas_data.values, cvx_data.values, sigma_r_data.values, nu_r_data.values
    )

    # Monte Carlo simulation parameters
    dt = 0.01
    steps = 252  # Assuming 252 trading days in a year
    S_OAS_init = float(oas_data.iloc[-1])
    C_init = float(cvx_data.iloc[-1])
    S_OAS_inf = float(np.mean(oas_data))
    num_paths = 1000  # Number of Monte Carlo paths

    # Create model instance
    model = JointReversionModel(kappa, lambda_, gamma, beta, sigma_O_0, delta, sigma_C, dt)

    # Perform Monte Carlo simulation
    expected_value_OAS, expected_value_C, expected_value_sigma_O, paths_OAS, paths_C, paths_sigma_O = monte_carlo_simulation(
        model,
        S_OAS_init,
        C_init,
        S_OAS_inf,
        sigma_r_forward.values,
        nu_r_forward.values,
        num_paths,
        steps,
        seed
    )

    print(f'Expected value of OAS in one year: {expected_value_OAS * 1e4:.0f} bps')
    print(f'Expected value of Convexity in one year: {expected_value_C * 1e4:.0f} bps')
    print(f'Expected value of Sigma_O in one year: {expected_value_sigma_O * 1e4:.0f} bps')

    # Plot historical data and Monte Carlo paths
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot OAS
    paths_OAS = pd.DataFrame(paths_OAS).T
    paths_OAS.index = pd.date_range(start=oas_data.index[-1], periods=steps, freq='B')
    OAS = pd.concat([oas_data, paths_OAS], axis=1)
    axs[0].plot(OAS, color='lightblue', alpha=0.1)
    axs[0].plot(oas_data, color='darkblue', label='OAS')
    axs[0].axhline(y=expected_value_OAS, color='lightblue', linestyle='--', label='Projected OAS')
    axs[0].axhline(y=oas_data.mean(), color='darkblue', linestyle='--', label='Historical Mean')
    axs[0].set_title('Monte Carlo Simulation of OAS')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('OAS')
    axs[0].legend()

    # Plot Convexity
    paths_C = pd.DataFrame(paths_C).T
    paths_C.index = pd.date_range(start=cvx_data.index[-1], periods=steps, freq='B')
    C = pd.concat([cvx_data, paths_C], axis=1)
    axs[1].plot(C, color='lightgreen', alpha=0.1)
    axs[1].plot(cvx_data, color='darkgreen', label='Convexity')
    axs[1].axhline(y=expected_value_C, color='lightgreen', linestyle='--', label='Projected Convexity')
    axs[1].axhline(y=cvx_data.mean(), color='darkgreen', linestyle='--', label='Historical Mean')
    axs[1].set_title('Monte Carlo Simulation of Convexity')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Convexity')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
