import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stochastic_processes import generate_mean_reversion, generate_correlated_mean_reversion
from typing import Tuple

def generate_series(
        params: dict[str, float], 
        start: str, 
        end: str, 
        freq: str = 'B', 
        seed: int = 42
) -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range(start=start, end=end, freq=freq)
    T = len(dates) - 1
    N = len(dates) - 1
    t, series = generate_mean_reversion(
        params['mu'], params['theta'], params['sigma'], params['X0'], T, N, seed
    )
    return pd.Series(series, index=dates)

def generate_correlated_series(
    params1: dict[str, float],
    params2: dict[str, float],
    rho: float,
    start: str,
    end: str,
    freq: str = 'B',
    seed: int = 42
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate two correlated mean reversion series using the Euler-Maruyama method.

    Parameters:
    - params1: Parameters for the first mean reversion process.
    - params2: Parameters for the second mean reversion process.
    - rho: Correlation coefficient between the two processes.
    - start: Start date for the series.
    - end: End date for the series.
    - freq: Frequency of the data (default: 'B' for business days).
    - seed: Random seed for reproducibility (default: 42).

    Returns:
    - Tuple containing two pandas Series.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start, end=end, freq=freq)
    T = len(dates) - 1
    N = len(dates) - 1
    t, series1, series2 = generate_correlated_mean_reversion(
        params1['mu'], params1['theta'], params1['sigma'], params1['X0'],
        params2['mu'], params2['theta'], params2['sigma'], params2['X0'],
        T, N, rho, seed
    )
    return pd.Series(series1, index=dates), pd.Series(series2, index=dates)

def generate_training_data(
    zv_params: dict[str, float],
    oas_params: dict[str, float],
    zv_oas_rho: float,
    sigma_r_params: dict[str, float],
    nu_r_params: dict[str, float],
    start: str = '2013-01-01',
    end: str | None = None,
    freq: str = 'B',
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate historical training data for MBS ZV, MBS OAS, Rates Vol, and Rates Vol of Vol.

    Parameters:
    - zv_params: Parameters for ZV mean reversion process.
    - oas_params: Parameters for OAS mean reversion process.
    - sigma_r_params: Parameters for sigma_r mean reversion process.
    - nu_r_params: Parameters for nu_r mean reversion process.
    - train_start: Start date for training data (default: '2013-01-01').
    - train_end: End date for training data (default: most recent business day).
    - freq: Frequency of the data (default: 'B' for business days).
    - seed: Random seed for reproducibility (default: 42).

    Returns:
    - Tuple containing four pandas Series: zv_data, oas_data, sigma_r_data, nu_r_data.
    """
    if end is None:
        end = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')

    # Generate past training spread data
    zv_data, oas_data = generate_correlated_series(zv_params, oas_params, zv_oas_rho, start, end, freq, seed)
    sigma_r_data = generate_series(sigma_r_params, start, end, freq, seed)
    nu_r_data = generate_series(nu_r_params, start, end, freq, seed)
    
    return zv_data, oas_data, sigma_r_data, nu_r_data

def generate_forward_data(
    sigma_r_params: dict[str, float],
    nu_r_params: dict[str, float],
    start: str,
    num_days: int = 252,
    freq: str = 'B',
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate forward data for Rates Vol and Rates Vol of Vol.

    Parameters:
    - sigma_r_params: Parameters for sigma_r mean reversion process.
    - nu_r_params: Parameters for nu_r mean reversion process.
    - start_date: Start date for forward data.
    - project_num_days: Number of days to project forward (default: 252).
    - freq: Frequency of the data (default: 'B' for business days).
    - seed: Random seed for reproducibility (default: 42).

    Returns:
    - Tuple containing two pandas Series: sigma_r_data, nu_r_data.
    """
    end = pd.date_range(start=start, periods=num_days, freq=freq)[-1]
    sigma_r_data = generate_series(sigma_r_params, start, end, freq, seed)
    nu_r_data = generate_series(nu_r_params, start, end, freq, seed)
    
    return sigma_r_data, nu_r_data

def plot_mock_data(data: pd.DataFrame) -> None:
    data.plot()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Define variable parameters
    zv_params = {'mu': 0.005, 'theta': 0.01, 'sigma': 0.008, 'X0': 0.005}
    oas_params = {'mu': 0.003, 'theta': 0.01, 'sigma': 0.005, 'X0': 0.003}
    zv_oas_rho = 0.8
    sigma_r_params = {'mu': 0.02, 'theta': 0.01, 'sigma': 0.002, 'X0': 0.02}
    nu_r_params = {'mu': 0.01, 'theta': 0.01, 'sigma': 0.002, 'X0': 0.01}

    # Define simulation parameters
    train_start_date = '2013-01-01'
    train_end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    fwd_num_days = 252
    freq = 'B'
    seed = 42

    # Generate training data
    zv_data, oas_data, sigma_r_data, nu_r_data = generate_training_data(
        zv_params, oas_params, zv_oas_rho, sigma_r_params, nu_r_params,
        train_start_date, train_end_date, freq, seed
    )

    # Update X0 for forward data
    sigma_r_params['X0'] = sigma_r_data[-1]
    nu_r_params['X0'] = nu_r_data[-1]

    # Generate forward data
    sigma_r_forward, nu_r_forward = generate_forward_data(
        sigma_r_params, nu_r_params, train_end_date, fwd_num_days, freq, seed
    )

    mock_data = pd.DataFrame(
        {'MBS ZV': zv_data, 'MBS OAS': oas_data, 'Rates Vol': sigma_r_data, 'Rates Vol of Vol': nu_r_data, 
         'Rates Vol Forward': sigma_r_forward, 'Rates Vol of Vol Forward': nu_r_forward}
    ).mul(1e4)  # Scale to basis points
    plot_mock_data(mock_data)