import numpy as np

# Parameters
kappa = 0.1
lambda_ = 0.1
gamma_0 = 0.05
gamma_1 = 0.02
gamma_2 = 0.01
beta_0 = 0.03
beta_1 = 0.02
beta_2 = 0.01
sigma_O = 0.02
sigma_C = 0.02
S_OAS_inf = 0.03
sigma_r = 0.01
nu_r = 0.005

# Initial values
S_OAS = 0.02
C = 0.01

# Time step and number of steps
Delta_t = 0.01
N = 1000

# Arrays to store results
S_OAS_values = np.zeros(N)
C_values = np.zeros(N)

# Simulation
for n in range(N):
    Z_O = np.random.normal(0, 1)
    Z_C = np.random.normal(0, 1)
    
    S_OAS += -kappa * (S_OAS - S_OAS_inf - gamma_0 * C - gamma_1 * sigma_r - gamma_2 * nu_r) * Delta_t + sigma_O * np.sqrt(Delta_t) * Z_O
    C += -lambda_ * (C - beta_0 * S_OAS - beta_1 * sigma_r - beta_2 * nu_r) * Delta_t + sigma_C * np.sqrt(Delta_t) * Z_C
    
    S_OAS_values[n] = S_OAS
    C_values[n] = C

# Results are stored in S_OAS_values and C_values