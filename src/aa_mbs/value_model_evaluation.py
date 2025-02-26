import numpy as np
import pandas as pd
from aa_mbs.mock_data import generate_training_data
from aa_mbs.value_model import (
    JointReversionModel,
    monte_carlo_simulation,
    estimate_parameters,
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


def evaluate_frequency(
    data: dict,
    S_OAS_inf: float,
    C_CC: float,
    seed: int,
    num_paths: int,
    steps: int,
    complexity_params: dict,
) -> list:
    """Evaluate model at different frequencies."""
    frequencies = {'daily': 1, 'weekly': 5, 'monthly': 21}
    results = []

    for freq_name, freq in frequencies.items():
        resampled_data = resample_data(data, freq)

        # Estimate parameters using resampled data
        kappa, gamma, sigma_O_0, delta, lambda_, beta, sigma_C = estimate_parameters(
            resampled_data['oas'].values,
            resampled_data['cvx'].values,
            resampled_data['sigma_r'].values,
            resampled_data['nu_r'].values,
            S_OAS_inf,
            C_CC,
        )

        # Create model instance
        model = JointReversionModel(
            kappa, lambda_, gamma, beta, sigma_O_0, delta, sigma_C, dt=1 / 252
        )

        # Perform Monte Carlo simulation
        expected_value_OAS, expected_value_C, expected_value_sigma_O, _, _, _ = (
            monte_carlo_simulation(
                model,
                float(resampled_data['oas'].iloc[-1]),
                float(resampled_data['cvx'].iloc[-1]),
                S_OAS_inf,
                C_CC,
                resampled_data['sigma_r'].values,
                resampled_data['nu_r'].values,
                num_paths,
                steps,
                seed,
                **complexity_params,
            )
        )

        results.append(
            {
                'frequency': freq_name,
                'expected_value_OAS': expected_value_OAS,
                'expected_value_C': expected_value_C,
                'expected_value_sigma_O': expected_value_sigma_O,
            }
        )

    return results


def evaluate_complexity(
    data: dict, S_OAS_inf: float, C_CC: float, seed: int, num_paths: int, steps: int
) -> list:
    """Evaluate model with different complexities."""
    complexities = {
        'mean_reversion_only': {'enable_convexity': False, 'enable_volatility': False},
        'include_convexity': {'enable_convexity': True, 'enable_volatility': False},
        'full_model': {'enable_convexity': True, 'enable_volatility': True},
    }
    results = []

    for complexity_name, complexity_params in complexities.items():
        # Estimate parameters using data
        kappa, gamma, sigma_O_0, delta, lambda_, beta, sigma_C = estimate_parameters(
            data['oas'].values,
            data['cvx'].values,
            data['sigma_r'].values,
            data['nu_r'].values,
            S_OAS_inf,
            C_CC,
        )

        # Create model instance
        model = JointReversionModel(
            kappa, lambda_, gamma, beta, sigma_O_0, delta, sigma_C, dt=1 / 252
        )

        # Perform Monte Carlo simulation
        expected_value_OAS, expected_value_C, expected_value_sigma_O, _, _, _ = (
            monte_carlo_simulation(
                model,
                float(data['oas'].iloc[-1]),
                float(data['cvx'].iloc[-1]),
                S_OAS_inf,
                C_CC,
                data['sigma_r'].values,
                data['nu_r'].values,
                num_paths,
                steps,
                seed,
                **complexity_params,
            )
        )

        results.append(
            {
                'complexity': complexity_name,
                'expected_value_OAS': expected_value_OAS,
                'expected_value_C': expected_value_C,
                'expected_value_sigma_O': expected_value_sigma_O,
            }
        )

    return results


def evaluate_in_vs_out_of_sample(
    in_sample_data: dict,
    out_sample_data: dict,
    S_OAS_inf: float,
    C_CC: float,
    seed: int,
    num_paths: int,
    steps: int,
    complexity_params: dict,
) -> dict:
    """Evaluate model in-sample and out-of-sample."""
    results = {}

    # Estimate parameters using in-sample data
    kappa, gamma, sigma_O_0, delta, lambda_, beta, sigma_C = estimate_parameters(
        in_sample_data['oas'].values,
        in_sample_data['cvx'].values,
        in_sample_data['sigma_r'].values,
        in_sample_data['nu_r'].values,
        S_OAS_inf,
        C_CC,
    )

    # Create model instance
    model = JointReversionModel(
        kappa, lambda_, gamma, beta, sigma_O_0, delta, sigma_C, dt=1 / 252
    )

    # Perform Monte Carlo simulation using in-sample data
    expected_value_OAS_in, expected_value_C_in, expected_value_sigma_O_in, _, _, _ = (
        monte_carlo_simulation(
            model,
            float(in_sample_data['oas'].iloc[-1]),
            float(in_sample_data['cvx'].iloc[-1]),
            S_OAS_inf,
            C_CC,
            in_sample_data['sigma_r'].values,
            in_sample_data['nu_r'].values,
            num_paths,
            steps,
            seed,
            **complexity_params,
        )
    )

    # Perform Monte Carlo simulation using out-of-sample data
    (
        expected_value_OAS_out,
        expected_value_C_out,
        expected_value_sigma_O_out,
        _,
        _,
        _,
    ) = monte_carlo_simulation(
        model,
        float(out_sample_data['oas'].iloc[-1]),
        float(out_sample_data['cvx'].iloc[-1]),
        S_OAS_inf,
        C_CC,
        out_sample_data['sigma_r'].values,
        out_sample_data['nu_r'].values,
        num_paths,
        steps,
        seed,
        **complexity_params,
    )

    results['in_sample'] = {
        'expected_value_OAS': expected_value_OAS_in,
        'expected_value_C': expected_value_C_in,
        'expected_value_sigma_O': expected_value_sigma_O_in,
    }

    results['out_sample'] = {
        'expected_value_OAS': expected_value_OAS_out,
        'expected_value_C': expected_value_C_out,
        'expected_value_sigma_O': expected_value_sigma_O_out,
    }

    return results


def evaluate_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    S_OAS_inf: float,
    C_CC: float,
    seed: int = 42,
    num_paths: int = 100,
    steps: int = 252,
):
    """Evaluate the model at different frequencies, complexities, and in vs out of sample."""
    # Split data into in-sample and out-of-sample
    in_sample_data, out_sample_data = split_data(
        oas_data, cvx_data, sigma_r_data, nu_r_data
    )

    # Evaluate at different frequencies
    frequency_results = evaluate_frequency(
        in_sample_data,
        S_OAS_inf,
        C_CC,
        seed,
        num_paths,
        steps,
        {'enable_convexity': True, 'enable_volatility': True},
    )

    # Evaluate different complexities
    complexity_results = evaluate_complexity(
        in_sample_data, S_OAS_inf, C_CC, seed, num_paths, steps
    )

    # Evaluate in vs out of sample
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

    return {
        'frequency_results': frequency_results,
        'complexity_results': complexity_results,
        'in_vs_out_sample_results': in_vs_out_sample_results,
    }


if __name__ == '__main__':
    # Mock data simulation parameters
    train_start_date = '2013-01-01'
    train_end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    project_num_days = 252
    freq = 'B'

    # Model parameters
    zv_params = {'kappa': 0.3, 'gamma': 0.1, 'sigma_O_0': 0.01, 'delta': 0.1}
    oas_params = {'kappa': 0.3, 'gamma': 0.1, 'sigma_O_0': 0.01, 'delta': 0.1}
    zv_oas_rho = 0.5
    sigma_r_params = {'kappa': 0.3, 'gamma': 0.1, 'sigma_O_0': 0.01, 'delta': 0.1}
    nu_r_params = {'kappa': 0.3, 'gamma': 0.1, 'sigma_O_0': 0.01, 'delta': 0.1}
    seed = 42

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

    # Perform model evaluation
    results = evaluate_model(
        oas_data, cvx_data, sigma_r_data, nu_r_data, S_OAS_inf, C_CC
    )
    print(results)
