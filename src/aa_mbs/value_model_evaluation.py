import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from aa_mbs.mock_data import generate_historical_data
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
    data: dict,
    split_ratio: float = 0.8,
) -> tuple:
    """Split data into in-sample and out-of-sample."""
    split_index = int(len(oas_data) * split_ratio)
    in_sample_data = {k: v[:split_index] for k, v in data.items()}
    out_sample_data = {k: v[split_index:] for k, v in data.items()}
    return in_sample_data, out_sample_data


def estimate_parameters(
    data: dict, S_OAS_inf: float, C_CC: float, dt: float, complexity_params: dict
):
    """Estimate model parameters using OLS and refine using MLE."""
    (
        kappa_ols,
        gamma_ols,
        sigma_O_0_ols,
        delta_ols,
        lambda_ols,
        beta_ols,
        sigma_C_ols,
    ) = estimate_parameters_ols(
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

    (
        kappa_mle,
        gamma_mle,
        sigma_O_0_mle,
        delta_mle,
        lambda_mle,
        beta_mle,
        sigma_C_mle,
    ) = estimate_parameters_mle(
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

    return (
        kappa_mle,
        gamma_mle,
        sigma_O_0_mle,
        delta_mle,
        lambda_mle,
        beta_mle,
        sigma_C_mle,
    )


def perform_simulation(
    data: dict,
    S_OAS_init: float,
    C_init: float,
    sigma_r_forward: pd.Series,
    nu_r_forward: pd.Series,
    S_OAS_inf: float,
    C_CC: float,
    seed: int,
    num_paths: int,
    steps: int,
    complexity_params: dict,
):
    """Perform Monte Carlo simulation using the estimated parameters."""
    dt = 1 / steps
    kappa, gamma, sigma_O_0, delta, lambda_, beta, sigma_C = estimate_parameters(
        data, S_OAS_inf, C_CC, dt, complexity_params
    )

    model = JointReversionModel(
        kappa, lambda_, gamma, beta, sigma_O_0, delta, sigma_C, dt
    )

    paths_OAS, paths_C, paths_sigma_O = monte_carlo_simulation(
        model,
        S_OAS_init,
        C_init,
        S_OAS_inf,
        C_CC,
        sigma_r_forward.values,
        nu_r_forward.values,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
        num_paths=num_paths,
        steps=steps,
        seed=seed,
    )

    return paths_OAS, paths_C, paths_sigma_O


def evaluate_frequency(
    data: dict,
    S_OAS_init: float,
    C_init: float,
    sigma_r_forward: pd.Series,
    nu_r_forward: pd.Series,
    S_OAS_inf: float,
    C_CC: float,
    seed: int,
    num_paths: int,
    steps: int,
    complexity_params: dict,
) -> list:
    """Evaluate model at different frequencies."""
    # frequencies = {'daily': 1, 'weekly': 5, 'monthly': 21}
    frequencies = {'daily': 'B', 'weekly': 'W-FRI', 'monthly': 'BME'}
    results = {}

    for freq_name, freq in frequencies.items():
        resampled_data = {k: v.asfreq('D').ffill().asfreq(freq) for k, v in data.items()}
        # resampled_data = resample_data(data, freq)
        # adjusted_steps = (
        #     steps // freq_days
        # )  # Adjust the number of steps according to the frequency
        adjusted_steps = int(steps // (len(data['oas']) / len(resampled_data['oas'])))
        paths_OAS, paths_C, paths_sigma_O = perform_simulation(
            resampled_data,
            S_OAS_init,
            C_init,
            sigma_r_forward.asfreq('D').ffill().asfreq(freq),
            nu_r_forward.asfreq('D').ffill().asfreq(freq),
            S_OAS_inf,
            C_CC,
            seed,
            num_paths,
            adjusted_steps,
            complexity_params,
        )

        results[freq_name] = (
            paths_OAS,
            paths_C,
            paths_sigma_O,
        )

    return results


def evaluate_complexity(
    data: dict,
    S_OAS_init: float,
    C_init: float,
    sigma_r_forward: pd.Series,
    nu_r_forward: pd.Series,
    S_OAS_inf: float,
    C_CC: float,
    seed: int,
    num_paths: int,
    steps: int,
) -> list:
    """Evaluate model with different complexities."""
    complexities = {
        'mean_reversion_only': {'enable_convexity': False, 'enable_volatility': False},
        'include_convexity': {'enable_convexity': True, 'enable_volatility': False},
        'include_volatility': {'enable_convexity': False, 'enable_volatility': True},
        'full_model': {'enable_convexity': True, 'enable_volatility': True},
    }
    results = {}

    for complexity_name, complexity_params in complexities.items():
        paths_OAS, paths_C, paths_sigma_O = perform_simulation(
            data,
            S_OAS_init,
            C_init,
            sigma_r_forward,
            nu_r_forward,
            S_OAS_inf,
            C_CC,
            seed,
            num_paths,
            steps,
            complexity_params,
        )

        results[complexity_name] = (
            paths_OAS,
            paths_C,
            paths_sigma_O,
        )

    return results


def evaluate_ols_vs_mle(
    data: dict,
    S_OAS_init: float,
    C_init: float,
    sigma_r_forward: pd.Series,
    nu_r_forward: pd.Series,
    S_OAS_inf: float,
    C_CC: float,
    seed: int,
    num_paths: int,
    steps: int,
    complexity_params: dict,
) -> dict:
    """Evaluate model using OLS vs OLS + MLE."""
    results = {}

    dt = 1 / steps
    (
        kappa_ols,
        gamma_ols,
        sigma_O_0_ols,
        delta_ols,
        lambda_ols,
        beta_ols,
        sigma_C_ols,
    ) = estimate_parameters_ols(
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
        kappa_ols,
        lambda_ols,
        gamma_ols,
        beta_ols,
        sigma_O_0_ols,
        delta_ols,
        sigma_C_ols,
        dt,
    )

    paths_OAS_ols, paths_C_ols, paths_sigma_O_ols = monte_carlo_simulation(
        model_ols,
        S_OAS_init,
        C_init,
        S_OAS_inf,
        C_CC,
        sigma_r_forward.values,
        nu_r_forward.values,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
        num_paths=num_paths,
        steps=steps,
        seed=seed,
    )

    (
        kappa_mle,
        gamma_mle,
        sigma_O_0_mle,
        delta_mle,
        lambda_mle,
        beta_mle,
        sigma_C_mle,
    ) = estimate_parameters_mle(
        data['oas'].values,
        data['cvx'].values,
        data['sigma_r'].values,
        data['nu_r'].values,
        S_OAS_inf,
        C_CC,
        dt,
        initial_guess=(
            kappa_ols,
            *gamma_ols,
            sigma_O_0_ols,
            delta_ols,
            lambda_ols,
            *beta_ols,
            sigma_C_ols,
        ),
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
    )

    model_mle = JointReversionModel(
        kappa_mle,
        lambda_mle,
        gamma_mle,
        beta_mle,
        sigma_O_0_mle,
        delta_mle,
        sigma_C_mle,
        dt,
    )

    paths_OAS_mle, paths_C_mle, paths_sigma_O_mle = monte_carlo_simulation(
        model_mle,
        S_OAS_init,
        C_init,
        S_OAS_inf,
        C_CC,
        sigma_r_forward.values,
        nu_r_forward.values,
        enable_convexity=complexity_params['enable_convexity'],
        enable_volatility=complexity_params['enable_volatility'],
        num_paths=num_paths,
        steps=steps,
        seed=seed,
    )

    results['ols'] = (
        paths_OAS_ols,
        paths_C_ols,
        paths_sigma_O_ols,
    )

    results['mle'] = (
        paths_OAS_mle,
        paths_C_mle,
        paths_sigma_O_mle,
    )

    return results


def evaluate_criteria(simulated_paths, actual_data):
    """Evaluate the model using various criteria."""
    mse = mean_squared_error(actual_data, simulated_paths)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_data, simulated_paths)
    r2 = r2_score(actual_data, simulated_paths)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def evaluate_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    seed: int = 42,
    num_paths: int = 100,
    steps: int = 252,
):
    """Evaluate the model at different frequencies, complexities, and in vs out of sample."""
    sample_data = {
        'oas': oas_data,
        'cvx': cvx_data,
        'sigma_r': sigma_r_data,
        'nu_r': nu_r_data,
    }

    # Reserve forward steps for out-of-sample evaluation
    in_sample_data = {k: v[:-steps] for k, v in sample_data.items()}
    S_OAS_init = float(oas_data.iloc[-steps])
    C_init = float(cvx_data.iloc[-steps])
    sigma_r_forward = sigma_r_data.iloc[-steps:]
    nu_r_forward = nu_r_data.iloc[-steps:]

    # Equilibrium parameters
    S_OAS_inf = float(np.mean(oas_data[:-steps]))
    C_CC = float(np.mean(cvx_data[:-steps]))

    frequency_results = evaluate_frequency(
        in_sample_data,
        S_OAS_init,
        C_init,
        sigma_r_forward,
        nu_r_forward,
        S_OAS_inf,
        C_CC,
        seed,
        num_paths,
        steps,
        {'enable_convexity': True, 'enable_volatility': True},
    )

    complexity_results = evaluate_complexity(
        sample_data,
        S_OAS_init,
        C_init,
        sigma_r_forward,
        nu_r_forward,
        S_OAS_inf,
        C_CC,
        seed,
        num_paths,
        steps,
    )

    ols_vs_mle_results = evaluate_ols_vs_mle(
        sample_data,
        S_OAS_init,
        C_init,
        sigma_r_forward,
        nu_r_forward,
        S_OAS_inf,
        C_CC,
        seed,
        num_paths,
        steps,
        {'enable_convexity': True, 'enable_volatility': True},
    )

    # Evaluate criteria for each result set
    def extract_result(paths: list[np.ndarray]) -> pd.Series:
        expected_path = pd.DataFrame(paths).T.mean(axis=1)
        expected_path.index = out_index[::steps // len(expected_path)]
        return expected_path
    
    out_oas_data = oas_data.iloc[-steps:]
    out_cvx_data = cvx_data.iloc[-steps:]
    out_index = out_oas_data.index
    
    # frequency_oas = {k: extract_result(v[0]) for k, v in frequency_results.items()}
    complexity_oas = {k: extract_result(v[0]) for k, v in complexity_results.items()}
    ols_vs_mle_oas = {k: extract_result(v[0]) for k, v in ols_vs_mle_results.items()}
    oas_evaluation = {
        # 'frequency': {k: evaluate_criteria(v, out_oas_data) for k, v in frequency_oas.items()},
        'complexity': {k: evaluate_criteria(v, out_oas_data) for k, v in complexity_oas.items()},
        'ols_vs_mle': {k: evaluate_criteria(v, out_oas_data) for k, v in ols_vs_mle_oas.items()},
    }

    # frequency_cvx = {k: extract_result(v[1]) for k, v in frequency_results.items()}
    complexity_cvx = {k: extract_result(v[1]) for k, v in complexity_results.items()}
    ols_vs_mle_cvx = {k: extract_result(v[1]) for k, v in ols_vs_mle_results.items()}
    cvx_evaluation = {
        # 'frequency': {k: evaluate_criteria(v, out_cvx_data) for k, v in frequency_cvx.items()},
        'complexity': {k: evaluate_criteria(v, out_cvx_data) for k, v in complexity_cvx.items()},
        'ols_vs_mle': {k: evaluate_criteria(v, out_cvx_data) for k, v in ols_vs_mle_cvx.items()},
    }

    def flatten_evaluation(evaluation_dict):
        """Flatten the evaluation dictionary into a DataFrame."""
        flattened_data = []
        for topic, cases in evaluation_dict.items():
            for case, metrics in cases.items():
                flattened_entry = {'topic': topic, 'case': case}
                flattened_entry.update(metrics)
                flattened_data.append(flattened_entry)
        return pd.DataFrame(flattened_data)
    
    oas_evaluation_df = flatten_evaluation(oas_evaluation)
    cvx_evaluation_df = flatten_evaluation(cvx_evaluation)

    return {
        'frequency_results': frequency_results,
        'complexity_results': complexity_results,
        'ols_vs_mle_results': ols_vs_mle_results,
        'oas_evaluation': oas_evaluation_df,
        'cvx_evaluation': cvx_evaluation_df,
    }


if __name__ == '__main__':
    # Set seed for reproducibility
    seed = 42

    # Define mock data parameters
    zv_params = {'mu': 0.005, 'theta': 0.01, 'sigma': 0.002, 'X0': 0.008}
    oas_params = {'mu': 0.003, 'theta': 0.02, 'sigma': 0.001, 'X0': 0.002}
    zv_oas_rho = 0.8
    sigma_r_params = {'mu': 0.002, 'theta': 0.01, 'sigma': 0.001, 'X0': 0.002}
    nu_r_params = {'mu': 0.001, 'theta': 0.01, 'sigma': 0.001, 'X0': 0.001}

    # Mock data simulation parameters
    start_date = '2013-01-01'
    end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    freq = 'B'

    # Generate mock data
    zv_data, oas_data, sigma_r_data, nu_r_data = generate_historical_data(
        zv_params,
        oas_params,
        zv_oas_rho,
        sigma_r_params,
        nu_r_params,
        start_date,
        end_date,
        freq,
        seed,
    )
    cvx_data = zv_data - oas_data

    # Simulation parameters
    steps = 252  # Assuming 252 trading days in a year
    num_paths = 1000  # Number of Monte Carlo paths

    # Perform model evaluation
    results = evaluate_model(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        seed,
        num_paths,
        steps,
    )

    print(results)
