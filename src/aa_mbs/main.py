import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

class JointReversionModel:
    def __init__(self, kappa, lambda_, gamma, beta, sigma_O, sigma_C, delta, dt, steps):
        self.kappa = kappa
        self.lambda_ = lambda_
        self.gamma = gamma
        self.beta = beta
        self.sigma_O = sigma_O
        self.sigma_C = sigma_C
        self.delta = delta
        self.dt = dt
        self.steps = steps

    def simulate(self, S_OAS_init, C_init, S_OAS_inf, sigma_r, nu_r, enable_convexity=True, enable_volatility=True):
        S_OAS = np.zeros(self.steps)
        C = np.zeros(self.steps)
        S_OAS[0] = S_OAS_init
        C[0] = C_init

        for t in range(1, self.steps):
            Z_O = np.random.normal()
            Z_C = np.random.normal()

            if enable_convexity:
                gamma_term = self.gamma[0] * float(C[t-1])
                beta_term = self.beta[0] * float(S_OAS[t-1])
            else:
                gamma_term = 0
                beta_term = 0

            if enable_volatility:
                gamma_term += self.gamma[1] * float(sigma_r[t-1]) + self.gamma[2] * float(nu_r[t-1])
                beta_term += self.beta[1] * float(sigma_r[t-1]) + self.beta[2] * float(nu_r[t-1])

            S_OAS[t] = float(S_OAS[t-1] - self.kappa * (S_OAS[t-1] - S_OAS_inf - gamma_term) * self.dt + self.sigma_O * np.sqrt(self.dt) * Z_O)
            C[t] = float(C[t-1] - self.lambda_ * (C[t-1] - beta_term) * self.dt + self.sigma_C * np.sqrt(self.dt) * Z_C)

        return S_OAS, C

def estimate_parameters(data):
    # Extract historical data
    S_OAS = data['OAS']
    C = data['Convexity']
    sigma_r = data['Sigma_r']
    nu_r = data['Nu_r']

    # Mean Reversion Parameter Estimation using OLS
    X_OAS = np.column_stack([np.ones(len(S_OAS)-1), S_OAS[:-1], C[:-1], sigma_r[:-1], nu_r[:-1]])
    y_OAS = S_OAS[1:]
    kappa, gamma_0, gamma_1, gamma_2 = np.linalg.lstsq(X_OAS, y_OAS, rcond=None)[0][1:]

    X_C = np.column_stack([np.ones(len(C)-1), C[:-1], S_OAS[:-1], sigma_r[:-1], nu_r[:-1]])
    y_C = C[1:]
    lambda_, beta_0, beta_1, beta_2 = np.linalg.lstsq(X_C, y_C, rcond=None)[0][1:]

    # Residual Variance Calibration
    residuals_OAS = y_OAS - (kappa * (S_OAS[:-1] - gamma_0 * C[:-1] - gamma_1 * sigma_r[:-1] - gamma_2 * nu_r[:-1]))
    residuals_C = y_C - (lambda_ * (C[:-1] - beta_0 * S_OAS[:-1] - beta_1 * sigma_r[:-1] - beta_2 * nu_r[:-1]))

    def variance_function(params):
        sigma_O_0, delta = params
        return np.sum((residuals_OAS**2 - (sigma_O_0**2 + delta * C[:-1]**2))**2)

    sigma_O_0, delta = minimize(variance_function, [0.02, 0.1]).x

    sigma_O = np.sqrt(sigma_O_0**2 + delta * C**2)
    sigma_C = np.std(residuals_C)

    return kappa, lambda_, [gamma_0, gamma_1, gamma_2], [beta_0, beta_1, beta_2], sigma_O, sigma_C, delta

def stationarity_tests(data):
    S_OAS = data['OAS']
    C = data['Convexity']

    adf_OAS = adfuller(S_OAS)
    adf_C = adfuller(C)

    return adf_OAS, adf_C

def main():
    # Load historical data
    data = pd.read_csv('mock_data.csv')

    # Estimate parameters
    kappa, lambda_, gamma, beta, sigma_O, sigma_C, delta = estimate_parameters(data)

    # Stationarity tests
    adf_OAS, adf_C = stationarity_tests(data)
    print(f'ADF Test for OAS: {adf_OAS}')
    print(f'ADF Test for Convexity: {adf_C}')

    # Simulation parameters
    dt = 0.01
    steps = 1000
    S_OAS_init = float(data['OAS'].iloc[-1])
    C_init = float(data['Convexity'].iloc[-1])
    S_OAS_inf = float(np.mean(data['OAS']))
    sigma_r = np.random.normal(0, 0.01, steps).astype(float)
    nu_r = np.random.normal(0, 0.01, steps).astype(float)

    # Create model instance
    model = JointReversionModel(kappa, lambda_, gamma, beta, sigma_O, sigma_C, delta, dt, steps)

    # Simulate with all components enabled
    S_OAS, C = model.simulate(S_OAS_init, C_init, S_OAS_inf, sigma_r, nu_r, enable_convexity=True, enable_volatility=True)

    # Plot results
    plt.plot(S_OAS, label='S_OAS')
    plt.plot(C, label='C')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()