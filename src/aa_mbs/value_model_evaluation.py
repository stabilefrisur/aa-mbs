import numpy as np
import pandas as pd
from aa_mbs.mock_data import generate_training_data
from aa_mbs.value_model import (
    JointReversionModel,
    monte_carlo_simulation,
    estimate_parameters_ols,
    estimate_parameters_mle,
)


def resample_data(data: dict, freq: int) -> dict:
    """Resample data according to the specified frequency."""
    return {key: val[::freq] for key, val in data.items()}


def split_data(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    split_ratio: float = 0.8,
) -> tuple:
    """Split data into in-sample and out-of-sample."""
    split_index = int(len(oas_data) * split_ratio)
    in_sample_data = {
        'oas': oas_data[:split_index],
        'cvx': cvx_data[:split_index],
        'sigma_r': sigma_r_data[:split_index],
        'nu_r': nu_r_data[:split_index],
    }
    out_sample_data = {
        'oas': oas_data[split_index:],
        'cvx': cvx_data[split_index:],
        'sigma_r': sigma_r_data[split_index:],
        'nu_r': nu_r_data[split_index:],
    }
    return in_sample_data, out_sample_data


def estimate_parameters(data: dict, S_OAS_inf: float, C_CC: float, dt: float, complexity_params: dict):
    """Estimate model parameters using OLS and refine using MLE."""
    kappa_ols, gamma_ols, sigma_O_0_ols, delta_ols, lambda_ols, beta_ols, sigma_C_ols = estimate_parameters_ols(
        data['oas'].values,
        data['cvx'].values,
        data['sigma_r'].values,
        data['nu_r'].values,
        S_OAS_inf,
        C_CC,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
    )

    initial_guess = [
        kappa_ols,
        *gamma_ols,
        sigma_O_0_ols,
        delta_ols,
        lambda_ols,
        *beta_ols,
        sigma_C_ols,
    ]

    kappa_mle, gamma_mle, sigma_O_0_mle, delta_mle, lambda_mle, beta_mle, sigma_C_mle = estimate_parameters_mle(
        data['oas'].values,
        data['cvx'].values,
        data['sigma_r'].values,
        data['nu_r'].values,
        S_OAS_inf,
        C_CC,
        dt,
        initial_guess=initial_guess,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
    )

    return kappa_mle, gamma_mle, sigma_O_0_mle, delta_mle, lambda_mle, beta_mle, sigma_C_mle


def perform_simulation(data: dict, S_OAS_inf: float, C_CC: float, seed: int, num_paths: int, steps: int, complexity_params: dict):
    """Perform Monte Carlo simulation using the estimated parameters."""
    dt = 1 / steps
    kappa, gamma, sigma_O_0, delta, lambda_, beta, sigma_C = estimate_parameters(
        data, S_OAS_inf, C_CC, dt, complexity_params
    )

    model = JointReversionModel(
        kappa, lambda_, gamma, beta, sigma_O_0, delta, sigma_C, dt
    )

    S_OAS_init = float(data['oas'].iloc[-1])
    C_init = float(data['cvx'].iloc[-1])

    paths_OAS, paths_C, paths_sigma_O = monte_carlo_simulation(
        model,
        S_OAS_init,
        C_init,
        S_OAS_inf,
        C_CC,
        data['sigma_r'].values,
        data['nu_r'].values,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
        num_paths=num_paths,
        steps=steps,
        seed=seed,
    )

    return paths_OAS, paths_C, paths_sigma_O


def evaluate_frequency(data: dict, S_OAS_inf: float, C_CC: float, seed: int, num_paths: int, steps: int, complexity_params: dict) -> list:
    """Evaluate model at different frequencies."""
    frequencies = {'daily': 1, 'weekly': 5, 'monthly': 21}
    results = []

    for freq_name, freq in frequencies.items():
        resampled_data = resample_data(data, freq)
        paths_OAS, paths_C, paths_sigma_O = perform_simulation(
            resampled_data, S_OAS_inf, C_CC, seed, num_paths, steps, complexity_params
        )

        results.append(
            {
                'frequency': freq_name,
                'expected_value_OAS': np.mean(paths_OAS, axis=0)[-1],
                'expected_value_C': np.mean(paths_C, axis=0)[-1],
                'expected_value_sigma_O': np.mean(paths_sigma_O, axis=0)[-1],
            }
        )

    return results


def evaluate_complexity(data: dict, S_OAS_inf: float, C_CC: float, seed: int, num_paths: int, steps: int) -> list:
    """Evaluate model with different complexities."""
    complexities = {
        'mean_reversion_only': {'enable_convexity': False, 'enable_volatility': False},
        'include_convexity': {'enable_convexity': True, 'enable_volatility': False},
        'include_volatility': {'enable_convexity': False, 'enable_volatility': True},
        'full_model': {'enable_convexity': True, 'enable_volatility': True},
    }
    results = []

    for complexity_name, complexity_params in complexities.items():
        paths_OAS, paths_C, paths_sigma_O = perform_simulation(
            data, S_OAS_inf, C_CC, seed, num_paths, steps, complexity_params
        )

        results.append(
            {
                'complexity': complexity_name,
                'expected_value_OAS': np.mean(paths_OAS, axis=0)[-1],
                'expected_value_C': np.mean(paths_C, axis=0)[-1],
                'expected_value_sigma_O': np.mean(paths_sigma_O, axis=0)[-1],
            }
        )

    return results


def evaluate_in_vs_out_of_sample(in_sample_data: dict, out_sample_data: dict, S_OAS_inf: float, C_CC: float, seed: int, num_paths: int, steps: int, complexity_params: dict) -> dict:
    """Evaluate model in-sample and out-of-sample."""
    results = {}

    paths_OAS_in, paths_C_in, paths_sigma_O_in = perform_simulation(
        in_sample_data, S_OAS_inf, C_CC, seed, num_paths, steps, complexity_params
    )

    paths_OAS_out, paths_C_out, paths_sigma_O_out = perform_simulation(
        out_sample_data, S_OAS_inf, C_CC, seed, num_paths, steps, complexity_params
    )

    results['in_sample'] = {
        'expected_value_OAS': np.mean(paths_OAS_in, axis=0)[-1],
        'expected_value_C': np.mean(paths_C_in, axis=0)[-1],
        'expected_value_sigma_O': np.mean(paths_sigma_O_in, axis=0)[-1],
    }

    results['out_sample'] = {
        'expected_value_OAS': np.mean(paths_OAS_out, axis=0)[-1],
        'expected_value_C': np.mean(paths_C_out, axis=0)[-1],
        'expected_value_sigma_O': np.mean(paths_sigma_O_out, axis=0)[-1],
    }

    return results


def evaluate_ols_vs_mle(data: dict, S_OAS_inf: float, C_CC: float, seed: int, num_paths: int, steps: int, complexity_params: dict) -> dict:
    """Evaluate model using OLS vs OLS + MLE."""
    results = {}

    dt = 1 / steps
    kappa_ols, gamma_ols, sigma_O_0_ols, delta_ols, lambda_ols, beta_ols, sigma_C_ols = estimate_parameters_ols(
        data['oas'].values,
        data['cvx'].values,
        data['sigma_r'].values,
        data['nu_r'].values,
        S_OAS_inf,
        C_CC,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
    )

    model_ols = JointReversionModel(
        kappa_ols, lambda_ols, gamma_ols, beta_ols, sigma_O_0_ols, delta_ols, sigma_C_ols, dt
    )

    paths_OAS_ols, paths_C_ols, paths_sigma_O_ols = monte_carlo_simulation(
        model_ols,
        float(data['oas'].iloc[-1]),
        float(data['cvx'].iloc[-1]),
        S_OAS_inf,
        C_CC,
        data['sigma_r'].values,
        data['nu_r'].values,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
        num_paths=num_paths,
        steps=steps,
        seed=seed,
    )

    kappa_mle, gamma_mle, sigma_O_0_mle, delta_mle, lambda_mle, beta_mle, sigma_C_mle = estimate_parameters_mle(
        data['oas'].values,
        data['cvx'].values,
        data['sigma_r'].values,
        data['nu_r'].values,
        S_OAS_inf,
        C_CC,
        dt,
        initial_guess=(kappa_ols, *gamma_ols, sigma_O_0_ols, delta_ols, lambda_ols, *beta_ols, sigma_C_ols),
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
    )

    model_mle = JointReversionModel(
        kappa_mle, lambda_mle, gamma_mle, beta_mle, sigma_O_0_mle, delta_mle, sigma_C_mle, dt
    )

    paths_OAS_mle, paths_C_mle, paths_sigma_O_mle = monte_carlo_simulation(
        model_mle,
        float(data['oas'].iloc[-1]),
        float(data['cvx'].iloc[-1]),
        S_OAS_inf,
        C_CC,
        data['sigma_r'].values,
        data['nu_r'].values,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
        num_paths=num_paths,
        steps=steps,
        seed=seed,
    )

    results['ols'] = {
        'expected_value_OAS': np.mean(paths_OAS_ols, axis=0)[-1],
        'expected_value_C': np.mean(paths_C_ols, axis=0)[-1],
        'expected_value_sigma_O': np.mean(paths_sigma_O_ols, axis=0)[-1],
    }

    results['mle'] = {
        'expected_value_OAS': np.mean(paths_OAS_mle, axis=0)[-1],
        'expected_value_C': np.mean(paths_C_mle, axis=0)[-1],
        'expected_value_sigma_O': np.mean(paths_sigma_O_mle, axis=0)[-1],
    }

    return results


def evaluate_model(oas_data: pd.Series, cvx_data: pd.Series, sigma_r_data: pd.Series, nu_r_data: pd.Series, S_OAS_inf: float, C_CC: float, seed: int = 42, num_paths: int = 100, steps: int = 252):
    """Evaluate the model at different frequencies, complexities, and in vs out of sample."""
    in_sample_data, out_sample_data = split_data(oas_data, cvx_data, sigma_r_data, nu_r_data)

    frequency_results = evaluate_frequency(
        in_sample_data,
        S_OAS_inf,
        C_CC,
        seed,
        num_paths,
        steps,
        {'enable_convexity': True, 'enable_volatility': True},
    )

    complexity_results = evaluate_complexity(
        in_sample_data, S_OAS_inf, C_CC, seed, num_paths, steps
    )

    in_vs_out_sample_results = evaluate_in_vs_out_of_sample(
        in_sample_data,
        out_sample_data,
        S_OAS_inf,
        C_CC,
        seed,
        num_paths,
        steps,
        {'enable_convexity': True, 'enable_volatility': True},
    )

    ols_vs_mle_results = evaluate_ols_vs_mle(
        in_sample_data,
        S_OAS_inf,
        C_CC,
        seed,
        num_paths,
        steps,
        {'enable_convexity': True, 'enable_volatility': True},
    )

    return {
        'frequency_results': frequency_results,
        'complexity_results': complexity_results,
        'in_vs_out_sample_results': in_vs_out_sample_results,
        'ols_vs_mle_results': ols_vs_mle_results,
    }


if __name__ == '__main__':
    # Set seed for reproducibility
    seed = 42

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
    freq = 'B'

    # Generate training data
    zv_data, oas_data, sigma_r_data, nu_r_data = generate_training_data(
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

    # Estimate model parameters
    S_OAS_inf = float(np.mean(oas_data))
    C_CC = 0.004

    # Monte Carlo simulation parameters
    steps = project_num_days  # Assuming 252 trading days in a year
    num_paths = 1000  # Number of Monte Carlo paths

    # Perform model evaluation
    results = evaluate_model(
        oas_data, cvx_data, sigma_r_data, nu_r_data, S_OAS_inf, C_CC, seed, num_paths, steps
    )
    print(results)