import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller


class JointReversionModel:
    """Joint Reversion Model for OAS and Convexity.

    The model is defined by the following stochastic differential equations:
    dS_OAS = kappa * (S_OAS_inf - S_OAS) * dt + (gamma_0 * C + gamma_1 * sigma_r + gamma_2 * nu_r) * dt + sigma_O * dW_O
    dC = lambda * (C_CC - C) * dt + (beta_0 * S_OAS + beta_1 * sigma_r + beta_2 * nu_r) * dt + sigma_C * dW_C
    where:
    - S_OAS is the OAS spread.
    - C is the Convexity.
    - kappa and lambda are the reversion speeds for OAS and Convexity, respectively.
    - gamma and beta are the interest rate interaction coefficients for OAS and Convexity, respectively.
    - sigma_O is the volatility of OAS.
    - delta is the convexity volatility coefficient.
    - sigma_C is the volatility of Convexity.
    - dt is the time step.
    - dW_O and dW_C are Wiener processes for OAS and Convexity, respectively.

    The model is simulated using the Euler-Maruyama method with the following parameters:
    - S_OAS_init: Initial value of OAS spread.
    - C_init: Initial value of Convexity.
    - S_OAS_inf: Long-term mean of OAS spread.
    - C_CC: Convexity of the current coupon bond (or TBA).
    - sigma_r: Volatility of interest rates.
    - nu_r: Volatility of volatility of interest rates.
    - steps: Number of time steps.
    - seed: Random seed for reproducibility.

    For educational and illustrative purposes only. Not intended for trading or investment purposes.
    Use at your own risk. No warranty or guarantee of accuracy or reliability.
    """
    def __init__(
        self,
        kappa: float,
        lambda_: float,
        gamma: list[float],
        beta: list[float],
        sigma_O_0: float,
        delta: float,
        sigma_C: float,
        dt: float,
    ):
        """Initialize the Joint Reversion Model.

        Args:
            kappa (float): Reversion speed for OAS.
            lambda_ (float): Reversion speed for Convexity.
            gamma (list[float]): Interest rate interaction coefficients for OAS.
            beta (list[float]): Interest rate interaction coefficients for Convexity.
            sigma_O_0 (float): Initial volatility of OAS.
            delta (float): Convexity volatility coefficient.
            sigma_C (float): Volatility of Convexity.
            dt (float): Time step.
        """
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
        C_CC: float,
        sigma_r: np.ndarray,
        nu_r: np.ndarray,
        enable_convexity: bool = True,
        enable_volatility: bool = True,
        steps: int = 252,
        rng: np.random.Generator = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the joint reversion model for OAS and Convexity.

        Args:
            S_OAS_init (float): Initial value of OAS spread.
            C_init (float): Initial value of Convexity.
            S_OAS_inf (float): Long-term mean of OAS spread.
            C_CC (float): Convexity of the current coupon bond (or TBA).
            sigma_r (np.ndarray): Volatility of interest rates.
            nu_r (np.ndarray): Volatility of volatility of interest rates.
            enable_convexity (bool, optional): Enable convexity interaction. Defaults to True.
            enable_volatility (bool, optional): Enable interest rate interaction. Defaults to True.
            steps (int, optional): Number of time steps. Defaults to 252.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Simulated OAS, Convexity, and Volatility of OAS.
        """
        if rng is None:
            rng = np.random.default_rng()

        S_OAS = np.zeros(steps)
        C = np.zeros(steps)
        sigma_O = np.zeros(steps)
        S_OAS[0] = S_OAS_init
        C[0] = C_init
        sigma_O[0] = np.sqrt(self.sigma_O_0**2 + self.delta * C_init**2)

        for t in range(1, steps):
            Z_O = rng.normal()
            Z_C = rng.normal()

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
                + self.kappa * (S_OAS_inf - float(S_OAS[t - 1])) * self.dt
                + gamma_term * self.dt
                + sigma_O[t - 1] * np.sqrt(self.dt) * Z_O
            )

            C[t] = (
                float(C[t - 1])
                + self.lambda_ * (C_CC - float(C[t - 1])) * self.dt
                + beta_term * self.dt
                + self.sigma_C * np.sqrt(self.dt) * Z_C
            )

            sigma_O[t] = np.sqrt(self.sigma_O_0**2 + self.delta * C[t]**2)

        return S_OAS, C, sigma_O


def estimate_parameters_ols(
    S_OAS: np.ndarray,
    C: np.ndarray,
    sigma_r: np.ndarray,
    nu_r: np.ndarray,
    S_OAS_inf: float,
    C_CC: float,
    initial_guess: tuple[float, float] = (0.05, 0.2),
    enable_convexity: bool = True,
    enable_volatility: bool = True,
    verbose: bool = True,
) -> tuple[float, list[float], float, float, float, list[float], float]:
    """Estimate the parameters of the joint reversion model using OLS.

    Args:
        S_OAS (np.ndarray): OAS spread time series.
        C (np.ndarray): Convexity time series.
        sigma_r (np.ndarray): Volatility of interest rates time series.
        nu_r (np.ndarray): Volatility of volatility of interest rates time series.
        S_OAS_inf (float): Long-term mean of OAS spread.
        C_CC (float): Convexity of the current coupon bond (or TBA).
        initial_guess (tuple[float, float], optional): Initial guess for the residual variance parameters. Defaults to (0.05, 0.2).
        enable_convexity (bool, optional): Enable convexity interaction. Defaults to True.
        enable_volatility (bool, optional): Enable interest rate interaction. Defaults to True.
        verbose (bool, optional): Print estimated parameters. Defaults to True.

    Returns:
        tuple[float, list[float], float, float, float, list[float], float]: Estimated parameters.
    """
    # Mean Reversion Parameter Estimation using OLS
    X_OAS = [S_OAS_inf - S_OAS[:-1]]
    X_C = [C_CC - C[:-1]]

    if enable_convexity:
        X_OAS.append(C[:-1])
        X_C.append(S_OAS[:-1])
    if enable_volatility:
        X_OAS.extend([sigma_r[:-1], nu_r[:-1]])
        X_C.extend([sigma_r[:-1], nu_r[:-1]])

    X_OAS = np.vstack(X_OAS).T
    y_OAS = S_OAS[1:] - S_OAS[:-1]
    X_C = np.vstack(X_C).T
    y_C = C[1:] - C[:-1]

    # OLS regression for OAS
    ols_OAS = np.linalg.lstsq(X_OAS, y_OAS, rcond=None)[0]
    kappa = ols_OAS[0]
    if not enable_convexity and not enable_volatility:
        gamma = [0.0, 0.0, 0.0]
    elif enable_convexity and not enable_volatility:
        gamma = [ols_OAS[1]] + [0.0, 0.0]
    elif not enable_convexity and enable_volatility:
        gamma = [0.0] + list(ols_OAS[1:])
    else:
        gamma = list(ols_OAS[1:])

    # OLS regression for Convexity
    ols_C = np.linalg.lstsq(X_C, y_C, rcond=None)[0]
    lambda_ = ols_C[0]
    if not enable_convexity and not enable_volatility:
        beta = [0.0, 0.0, 0.0]
    elif enable_convexity and not enable_volatility:
        beta = [ols_C[1]] + [0.0, 0.0]
    elif not enable_convexity and enable_volatility:
        beta = [0.0] + list(ols_C[1:])
    else:
        beta = list(ols_C[1:])

    # Residuals for variance calibration
    residuals_OAS = y_OAS - X_OAS @ ols_OAS
    residuals_C = y_C - X_C @ ols_C

    # Residual Variance Calibration
    def variance_function(params):
        sigma_O_0, delta = params
        return np.sum((residuals_OAS**2 - (sigma_O_0**2 + delta * C[:-1] ** 2)) ** 2)

    if enable_convexity:
        sigma_O_0, delta = minimize(variance_function, initial_guess).x
    else:
        sigma_O_0 = np.std(residuals_OAS)
        delta = 0.0

    sigma_C = np.std(residuals_C)

    if verbose:
        print('\nEstimated Parameters (OLS):')
        print(f'kappa: {kappa:.4f}')
        print(f'gamma: {", ".join(f"{g:.4f}" for g in gamma)}')
        print(f'sigma_O_0: {sigma_O_0:.4f}')
        print(f'delta: {delta:.4f}')
        print(f'lambda: {lambda_:.4f}')
        print(f'beta: {", ".join(f"{b:.4f}" for b in beta)}')
        print(f'sigma_C: {sigma_C:.4f}')
        print()

    return (
        kappa,
        gamma,
        sigma_O_0,
        delta,
        lambda_,
        beta,
        sigma_C,
    )


def estimate_parameters_mle(
    S_OAS: np.ndarray,
    C: np.ndarray,
    sigma_r: np.ndarray,
    nu_r: np.ndarray,
    S_OAS_inf: float,
    C_CC: float,
    dt: float,
    initial_guess: tuple[float, ...] = None,
    enable_convexity: bool = True,
    enable_volatility: bool = True,
    verbose: bool = True,
) -> tuple[float, list[float], float, float, float, list[float], float]:
    """Estimate the parameters of the joint reversion model using MLE.

    Args:
        S_OAS (np.ndarray): OAS spread time series.
        C (np.ndarray): Convexity time series.
        sigma_r (np.ndarray): Volatility of interest rates time series.
        nu_r (np.ndarray): Volatility of volatility of interest rates time series.
        S_OAS_inf (float): Long-term mean of OAS spread.
        C_CC (float): Convexity of the current coupon bond (or TBA).
        dt (float): Time step size.
        initial_guess (tuple[float, ..., float], optional): Initial guess for the parameters. Defaults to (0.1, ..., 0.1).
        enable_convexity (bool, optional): Enable convexity interaction. Defaults to True.
        enable_volatility (bool, optional): Enable interest rate interaction. Defaults to True.
        verbose (bool, optional): Print estimated parameters. Defaults to True.

    Returns:
        tuple[float, list[float], float, float, float, list[float], float]: Estimated parameters.
    """
    if initial_guess is None:
        initial_guess = tuple([0.1] * 11)

    def log_likelihood(params):
        kappa = params[0]
        gamma = params[1:4]
        sigma_O_0 = params[4]
        delta = params[5]
        lambda_ = params[6]
        beta = params[7:10]
        sigma_C = params[10]

        n = len(S_OAS)
        log_likelihood = 0

        for t in range(1, n):
            S_OAS_prev = S_OAS[t - 1]
            C_prev = C[t - 1]
            sigma_r_prev = sigma_r[t - 1]
            nu_r_prev = nu_r[t - 1]

            S_OAS_mean = S_OAS_prev + kappa * (S_OAS_inf - S_OAS_prev) * dt
            if enable_convexity:
                S_OAS_mean += gamma[0] * C_prev * dt
            if enable_volatility:
                S_OAS_mean += (gamma[1] * sigma_r_prev + gamma[2] * nu_r_prev) * dt

            sigma_O = (
                np.sqrt(sigma_O_0**2 + delta * C_prev**2)
                if enable_convexity
                else sigma_O_0
            )

            C_mean = C_prev + lambda_ * (C_CC - C_prev) * dt
            if enable_convexity:
                C_mean += beta[0] * S_OAS_prev * dt
            if enable_volatility:
                C_mean += (beta[1] * sigma_r_prev + beta[2] * nu_r_prev) * dt

            log_likelihood += -0.5 * np.log(2 * np.pi * sigma_O**2 * dt) - 0.5 * (
                (S_OAS[t] - S_OAS_mean) ** 2 / (sigma_O**2 * dt)
            )
            log_likelihood += -0.5 * np.log(2 * np.pi * sigma_C**2 * dt) - 0.5 * (
                (C[t] - C_mean) ** 2 / (sigma_C**2 * dt)
            )

        return -log_likelihood

    # Optimize the log-likelihood function
    result = minimize(log_likelihood, initial_guess, method='L-BFGS-B')

    # Extract the estimated parameters
    kappa = result.x[0]
    gamma = result.x[1:4]
    sigma_O_0 = result.x[4]
    delta = result.x[5]
    lambda_ = result.x[6]
    beta = result.x[7:10]
    sigma_C = result.x[10]

    if verbose:
        print('\nEstimated Parameters (MLE):')
        print(f'kappa: {kappa:.4f}')
        print(f'gamma: {", ".join(f"{g:.4f}" for g in gamma)}')
        print(f'sigma_O_0: {sigma_O_0:.4f}')
        print(f'delta: {delta:.4f}')
        print(f'lambda: {lambda_:.4f}')
        print(f'beta: {", ".join(f"{b:.4f}" for b in beta)}')
        print(f'sigma_C: {sigma_C:.4f}')
        print()

    return (
        kappa,
        gamma,
        sigma_O_0,
        delta,
        lambda_,
        beta,
        sigma_C,
    )


def stationarity_tests(
    S_OAS: pd.Series,
    C: pd.Series,
    verbose: bool = True,
) -> tuple[
    tuple[float, float, int, int, dict, float],
    tuple[float, float, int, int, dict, float],
]:
    """Test the stationarity of the time series using the Augmented Dickey-Fuller test.

    Args:
        S_OAS (pd.Series): OAS spread time series.
        C (pd.Series): Convexity time series.

    Returns:
        tuple[ tuple[float, float, int, int, dict, float], tuple[float, float, int, int, dict, float], ]: ADF test results for OAS and Convexity.

    ADF test results:
    - Test statistic.
    - P-value.
    - Number of lags used.
    - Number of observations used.
    - Critical values.
    - Maximum information criterion.

    ADF test null hypothesis:
    - H0: The time series is non-stationary.

    ADF test alternative hypothesis:
    - H1: The time series is stationary.

    If the p-value is less than the significance level (e.g., 0.05), we reject the null hypothesis and conclude that the time series is stationary.

    Reference:
    - https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
    """
    adf_OAS = adfuller(S_OAS)
    adf_C = adfuller(C)

    if verbose:
        # Print ADF test results
        print('\nAugmented Dickey-Fuller Test for OAS:')
        print(f'Test Statistic: {adf_OAS[0]:.4f}')
        print(f'P-Value: {adf_OAS[1]:.4f}')
        print(f'Lags Used: {adf_OAS[2]:.0f}')
        print(f'Observations Used: {adf_OAS[3]:.0f}')
        print('Critical Values:')
        for key, value in adf_OAS[4].items():
            print(f'  {key}: {value:.4f}')
        print(f'Max Information Criteria: {adf_OAS[5]:.4f}')
        print()

        print('Augmented Dickey-Fuller Test for Convexity:')
        print(f'Test Statistic: {adf_C[0]:.4f}')
        print(f'P-Value: {adf_C[1]:.4f}')
        print(f'Lags Used: {adf_C[2]:.0f}')
        print(f'Observations Used: {adf_C[3]:.0f}')
        print('Critical Values:')
        for key, value in adf_C[4].items():
            print(f'  {key}: {value:.4f}')
        print(f'Max Information Criteria: {adf_C[5]:.4f}')
        print()

    return adf_OAS, adf_C


def monte_carlo_simulation(
    model: JointReversionModel,
    S_OAS_init: float,
    C_init: float,
    S_OAS_inf: float,
    C_CC: float,
    sigma_r: float | np.ndarray,
    nu_r: float | np.ndarray,
    enable_convexity: bool = True,
    enable_volatility: bool = True,
    num_paths: int = 1000,
    steps: int = 252,
    seed: int = None,
    verbose: bool = True,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Perform Monte Carlo simulation of the Joint Reversion Model.

    Args:
        model (JointReversionModel): Joint Reversion Model instance.
        S_OAS_init (float): Initial value of OAS spread.
        C_init (float): Initial value of Convexity.
        S_OAS_inf (float): Long-term mean of OAS spread.
        C_CC (float): Convexity of the current coupon bond (or TBA).
        sigma_r (float | np.ndarray): Volatility of interest rates.
        nu_r (float | np.ndarray): Volatility of volatility of interest rates.
        num_paths (int): Number of Monte Carlo paths.
        steps (int): Number of time steps.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple[list[np.ndarray], tuple[list[np.ndarray], list[np.ndarray]]: Monte Carlo paths for OAS, Convexity, and Volatility of OAS.

    The Monte Carlo simulation generates multiple paths of the OAS spread, Convexity, and Volatility of OAS using the Joint Reversion Model.
    The expected value of OAS, Convexity, and Volatility of OAS can be calculated as the average of the simulated paths.

    Reference:
    - https://en.wikipedia.org/wiki/Monte_Carlo_method

    Note:
    - The Monte Carlo simulation is a stochastic process that generates random paths.
    - The results may vary each time the simulation is run.
    - The results are based on the model assumptions and parameters.
    - The results are for illustrative and educational purposes only.
    - Use at your own risk. No warranty or guarantee of accuracy or reliability.
    """
    paths_OAS = []
    paths_C = []
    paths_sigma_O = []

    sigma_r_series = np.full(steps, sigma_r) if isinstance(sigma_r, float) else sigma_r
    nu_r_series = np.full(steps, nu_r) if isinstance(nu_r, float) else nu_r

    rng = np.random.default_rng(seed)

    for i in range(num_paths):
        S_OAS, C, sigma_O = model.simulate(
            S_OAS_init,
            C_init,
            S_OAS_inf,
            C_CC,
            sigma_r_series,
            nu_r_series,
            enable_convexity,
            enable_volatility,
            steps,
            rng,
        )
        paths_OAS.append(S_OAS)
        paths_C.append(C)
        paths_sigma_O.append(sigma_O)

    if verbose:
        print(
            f'Final expected value of OAS: {np.mean(paths_OAS, axis=0)[-1] * 1e4:.0f} bps'
        )
        print(
            f'Final expected value of Convexity: {np.mean(paths_C, axis=0)[-1] * 1e4:.0f} bps'
        )
        print(
            f'Final expected value of Sigma_O: {np.mean(paths_sigma_O, axis=0)[-1] * 1e4:.0f} bps'
        )

    return (
        paths_OAS,
        paths_C,
        paths_sigma_O,
    )
