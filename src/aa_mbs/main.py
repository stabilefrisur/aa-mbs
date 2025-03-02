import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aa_mbs.mock_data import generate_forward_data, generate_historical_data
from aa_mbs.value_model import (
    JointReversionModel,
    estimate_parameters_ols,
    estimate_parameters_mle,
    monte_carlo_simulation,
    stationarity_tests,
)


def main() -> None:
    """Main function running the MBS valuation process.

    The main function performs the following steps:
    1. Set seed for reproducibility.
    2. Define variable parameters for the model.
    3. Generate training data for ZV, OAS, Rates Vol, and Rates Vol of Vol.
    4. Generate forward data for Rates Vol and Rates Vol of Vol.
    5. Perform stationarity tests for OAS and Convexity.
    6. Estimate model parameters using OLS to generate initial guess for MLE.
    7. Estimate model parameters using MLE.
    8. Perform Monte Carlo simulation.
    9. Plot historical data and Monte Carlo paths.
    """
    # Set seed for reproducibility
    seed = 43

    # Define training data parameters
    zv_params = {'mu': 0.005, 'theta': 0.01, 'sigma': 0.002, 'X0': 0.008}
    oas_params = {'mu': 0.003, 'theta': 0.02, 'sigma': 0.001, 'X0': 0.002}
    zv_oas_rho = 0.8
    sigma_r_params = {'mu': 0.002, 'theta': 0.01, 'sigma': 0.001, 'X0': 0.002}
    nu_r_params = {'mu': 0.001, 'theta': 0.01, 'sigma': 0.001, 'X0': 0.001}

    # Mock data simulation parameters
    train_start_date = '2013-01-01'
    train_end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    project_num_days = 252
    dt = 1 / project_num_days
    freq = 'B'

    # Generate training data
    zv_data, oas_data, sigma_r_data, nu_r_data = generate_historical_data(
        zv_params,
        oas_params,
        zv_oas_rho,
        sigma_r_params,
        nu_r_params,
        train_start_date,
        train_end_date,
        freq,
        seed,
    )
    cvx_data = zv_data - oas_data

    # Update X0 for forward data
    sigma_r_params['X0'] = sigma_r_data.iloc[-1]
    nu_r_params['X0'] = nu_r_data.iloc[-1]

    # Generate forward data
    sigma_r_forward, nu_r_forward = generate_forward_data(
        sigma_r_params, nu_r_params, train_end_date, project_num_days, freq, seed
    )

    # Stationarity tests
    adf_OAS, adf_C = stationarity_tests(oas_data, cvx_data)

    # Model complexity parameters
    enable_convexity = True
    enable_volatility = True

    # Estimate model parameters using OLS
    S_OAS_inf = float(np.mean(oas_data))
    C_CC = 0.004

    (
        kappa_ols,
        gamma_ols,
        sigma_O_0_ols,
        delta_ols,
        lambda_ols,
        beta_ols,
        sigma_C_ols,
    ) = estimate_parameters_ols(
        oas_data.values,
        cvx_data.values,
        sigma_r_data.values,
        nu_r_data.values,
        S_OAS_inf,
        C_CC,
        enable_convexity=enable_convexity,
        enable_volatility=enable_volatility,
    )

    # Use OLS estimates as initial guess for MLE
    initial_guess = [
        kappa_ols,
        *gamma_ols,
        sigma_O_0_ols,
        delta_ols,
        lambda_ols,
        *beta_ols,
        sigma_C_ols,
    ]
    
    # Estimate model parameters using MLE
    (
        kappa_mle,
        gamma_mle,
        sigma_O_0_mle,
        delta_mle,
        lambda_mle,
        beta_mle,
        sigma_C_mle,
    ) = estimate_parameters_mle(
        oas_data.values,
        cvx_data.values,
        sigma_r_data.values,
        nu_r_data.values,
        S_OAS_inf,
        C_CC,
        dt,
        initial_guess=initial_guess,
        enable_convexity=enable_convexity,
        enable_volatility=enable_volatility,
    )

    # Monte Carlo simulation parameters
    steps = project_num_days  # Assuming 252 trading days in a year
    S_OAS_init = float(oas_data.iloc[-1])
    C_init = float(cvx_data.iloc[-1])
    num_paths = 1000  # Number of Monte Carlo paths

    # Create model instance with MLE parameters
    model = JointReversionModel(
        kappa_mle,
        lambda_mle,
        gamma_mle,
        beta_mle,
        sigma_O_0_mle,
        delta_mle,
        sigma_C_mle,
        dt,
    )

    # Perform Monte Carlo simulation
    (
        paths_OAS,
        paths_C,
        paths_sigma_O,
    ) = monte_carlo_simulation(
        model,
        S_OAS_init,
        C_init,
        S_OAS_inf,
        C_CC,
        sigma_r_forward.values,
        nu_r_forward.values,
        enable_convexity,
        enable_volatility,
        num_paths,
        steps,
        seed,
    )

    # Plot historical data and Monte Carlo paths
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot OAS
    paths_OAS = pd.DataFrame(paths_OAS).T
    paths_OAS.index = pd.date_range(start=oas_data.index[-1], periods=steps, freq='B')
    OAS = pd.concat([oas_data, paths_OAS], axis=1).mul(1e4)
    axs[0].plot(OAS.iloc[:, 0], color='darkblue', label='OAS')
    axs[0].plot(OAS.iloc[:, 1:], color='lightblue', alpha=0.1)
    axs[0].plot(OAS.iloc[:, 1:].mean(axis=1), color='darkblue', linestyle=':', label='Projected OAS', alpha=1.0)
    axs[0].axhline(y=S_OAS_inf * 1e4, color='darkblue', linestyle='--', label='Reversion Level')
    axs[0].set_title('Monte Carlo Simulation of OAS')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('OAS')
    axs[0].legend()

    # Plot Convexity
    paths_C = pd.DataFrame(paths_C).T
    paths_C.index = pd.date_range(start=cvx_data.index[-1], periods=steps, freq='B')
    C = pd.concat([cvx_data, paths_C], axis=1).mul(1e4)
    axs[1].plot(C.iloc[:, 0], color='darkgreen', label='Convexity')
    axs[1].plot(C.iloc[:, 1:], color='lightgreen', alpha=0.1)
    axs[1].plot(C.iloc[:, 1:].mean(axis=1), color='darkgreen', linestyle=':', label='Projected Convexity')
    axs[1].axhline(y=C_CC * 1e4, color='darkgreen', linestyle='--', label='Reversion Level')
    axs[1].set_title('Monte Carlo Simulation of Convexity')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Convexity')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()