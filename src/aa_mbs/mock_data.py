import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
n_points = 1000  # Number of data points
dates = pd.date_range(start='2020-01-01', periods=n_points, freq='B')  # Business days


# Generate mock data
def generate_time_series(
    n_points, mean, std_dev, mean_reversion_speed, noise_std_dev, non_negative=False
):
    series = np.zeros(n_points)
    for t in range(1, n_points):
        series[t] = (
            series[t - 1]
            + mean_reversion_speed * (mean - series[t - 1])
            + np.random.normal(0, noise_std_dev)
        )
        if non_negative:
            series[t] = max(series[t], 0)
    return series


# ZV and OAS
def generate_zv_oas_data(n_points):
    zv_mean = 0.02
    oas_mean = 0.01
    std_dev = 0.005
    mean_reversion_speed = 0.01
    noise_std_dev = 0.001

    zv_data = (
        generate_time_series(
            n_points, zv_mean, std_dev, mean_reversion_speed, noise_std_dev
        )
        * 1e4
    )
    oas_data = (
        generate_time_series(
            n_points, oas_mean, std_dev, mean_reversion_speed, noise_std_dev
        )
        * 1e4
    )
    return zv_data, oas_data


# Interest rate volatility (sigma_r) and volatility of volatility (nu_r)
def generate_vol_data(n_points):
    sigma_r_mean = 0.02
    nu_r_mean = 0.01
    std_dev_vol = 0.002
    mean_reversion_speed_vol = 0.01
    noise_std_dev_vol = 0.0005

    sigma_r_data = (
        generate_time_series(
            n_points,
            sigma_r_mean,
            std_dev_vol,
            mean_reversion_speed_vol,
            noise_std_dev_vol,
            non_negative=True,
        )
        * 1e4
    )
    nu_r_data = (
        generate_time_series(
            n_points,
            nu_r_mean,
            std_dev_vol,
            mean_reversion_speed_vol,
            noise_std_dev_vol,
            non_negative=True,
        )
        * 1e4
    )
    return sigma_r_data, nu_r_data


# Generate all mock data
def generate_mock_data(start_date='2013-01-01', end_date=None, freq='B', seed=42):
    np.random.seed(seed)

    if end_date is None:
        end_date = (pd.Timestamp.today() - pd.offsets.BDay(0)).strftime('%Y-%m-%d')

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_points = len(dates)

    zv_data, oas_data = generate_zv_oas_data(n_points)
    sigma_r_data, nu_r_data = generate_vol_data(n_points)
    convexity_data = zv_data - oas_data

    data = pd.DataFrame(
        {
            'Date': dates,
            'ZV': zv_data,
            'OAS': oas_data,
            'Convexity': convexity_data,
            'Sigma_r': sigma_r_data,
            'Nu_r': nu_r_data,
        }
    )
    data.set_index('Date', inplace=True)
    return data
