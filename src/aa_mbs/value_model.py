import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller


class JointReversionModel:
    """Joint Reversion Model for OAS and Convexity.

    The model is defined by the following stochastic differential equations:
    dS_OAS = -kappa * (S_OAS - S_OAS_inf - gamma * C) * dt + sigma_O * dW_O
    dC = -lambda * (C - C_CC - beta * S_OAS) * dt + sigma_C * dW_C
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
        seed: int = None,
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
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Simulated OAS, Convexity, and Volatility of OAS.
        """
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
                - self.lambda_ * (float(C[t - 1]) - C_CC - beta_term) * self.dt
                + self.sigma_C * np.sqrt(self.dt) * Z_C
            )

            sigma_O[t] = np.sqrt(self.sigma_O_0**2 + self.delta * C[t]**2)

        return S_OAS, C, sigma_O


def estimate_parameters(
    S_OAS: np.ndarray,
    C: np.ndarray,
    sigma_r: np.ndarray,
    nu_r: np.ndarray,
    initial_guess: tuple[float, float] = (0.05, 0.2)
) -> tuple[float, float, list[float], list[float], float, float, float]:
    """Estimate the parameters of the joint reversion model using OLS.

    Args:
        S_OAS (np.ndarray): OAS spread time series.
        C (np.ndarray): Convexity time series.
        sigma_r (np.ndarray): Volatility of interest rates time series.
        nu_r (np.ndarray): Volatility of volatility of interest rates time series.
        initial_guess (tuple[float, float], optional): Initial guess for the residual variance parameters. Defaults to (0.05, 0.2).

    Returns:
        tuple[float, float, list[float], list[float], float, float, float]: Estimated parameters of the Joint Reversion Model.

        Estimated parameters:
        - kappa: Reversion speed for OAS.
        - lambda_: Reversion speed for Convexity.
        - gamma: Interest rate interaction coefficients for OAS.
        - beta: Interest rate interaction coefficients for Convexity.
        - sigma_O_0: Initial volatility of OAS.
        - delta: Convexity volatility coefficient.
        - sigma_C: Volatility of Convexity.

    The model is estimated using Ordinary Least Squares (OLS) regression.
    OLS minimizes the sum of squared differences between the observed values and the values predicted by the model.

    The residual variance is calibrated using the following function:
    residuals_OAS = y_OAS - kappa * (S_OAS - gamma * C - sigma_r - nu_r)
    residuals_C = y_C - lambda_ * (C - beta * S_OAS - sigma_r - nu_r)
    variance_function = sum((residuals_OAS**2 - (sigma_O_0**2 + delta * C**2))**2)

    The variance function is minimized using the Nelder-Mead method.

    Reference:
    - https://en.wikipedia.org/wiki/Ordinary_least_squares
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
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

    return adf_OAS, adf_C


def monte_carlo_simulation(
    model: JointReversionModel,
    S_OAS_init: float,
    C_init: float,
    S_OAS_inf: float,
    C_CC: float,
    sigma_r: float | np.ndarray,
    nu_r: float | np.ndarray,
    num_paths: int,
    steps: int,
    seed: int = None,
) -> tuple[float, float, float, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
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
        tuple[float, float, float, list[np.ndarray], list[np.ndarray], list[np.ndarray]]: Expected value of OAS, Convexity, and Volatility of OAS, and Monte Carlo paths for OAS, Convexity, and Volatility of OAS.

    The Monte Carlo simulation generates multiple paths of the OAS spread, Convexity, and Volatility of OAS using the Joint Reversion Model.
    The expected value of OAS, Convexity, and Volatility of OAS is calculated as the average of the simulated paths.

    Reference:
    - https://en.wikipedia.org/wiki/Monte_Carlo_method

    Note:
    - The Monte Carlo simulation is a stochastic process that generates random paths.
    - The results may vary each time the simulation is run.
    - The results are based on the model assumptions and parameters.
    - The results are for illustrative and educational purposes only.
    - Use at your own risk. No warranty or guarantee of accuracy or reliability.
    """
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
            C_CC,
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
