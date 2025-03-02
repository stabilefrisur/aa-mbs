# Simulation of the Joint Reversion Model

This document describes the process of simulating the stochastic processes for OAS and Convexity using the Joint Reversion Model. The simulation is performed using the Euler-Maruyama method, which approximates the continuous-time processes with discrete-time steps. Monte Carlo Simulation provides further insights into the expected value and distribution of the stochastic processes.

## Discretised Joint Reversion Model

The Joint Reversion Model consists of two interlinked stochastic processes: OAS (Option-Adjusted Spread) and Convexity. The model captures the mean-reverting behavior of these processes and their dependence on external factors such as interest rate volatility ($\sigma_r$) and volatility of volatility ($\nu_r$).

The discretized versions of the OAS and Convexity processes are given by:

1. **Discretized OAS Process**:
   $$
   S_{OAS}(t_{n+1}) = S_{OAS}(t_n) + \kappa \left(S_{OAS}^{\infty} - S_{OAS}(t_n)\right) \Delta t + \left(\gamma_0 C(t_n) + \gamma_1 \sigma_r(t_n) + \gamma_2 \nu_r(t_n)\right) \Delta t + \sigma_O(t_n) \sqrt{\Delta t} \cdot Z_O 
   $$
   where 
   $$
   \sigma_O(t_n)^2 = \sigma_{O,0}^2 + \delta C(t_n)^2
   $$

2. **Discretized Convexity Process**:
   $$
   C(t_{n+1}) = C(t_n) + \lambda \left(C^{CC}- C(t_n)\right) \Delta t + \left(\beta_0 S_{OAS}(t_n) + \beta_1 \sigma_r(t_n) + \beta_2 \nu_r(t_n)\right) \Delta t + \sigma_C \sqrt{\Delta t} \cdot Z_C 
   $$

Here, $Z_O$ and $Z_C$ are independent standard normal random variables (i.e., $Z_O, Z_C \sim \mathcal{N}(0, 1)$) representing the Wiener processes $dW_O$ and $dW_C$.

## Stochastic Process Simulation

1. **Set Initial Conditions**:
   - $S_{OAS}(0) = S_{OAS,\text{init}}$
   - $C(0) = C_{\text{init}}$

2. **Define Parameters**:
   - $\kappa, \lambda, \gamma_0, \gamma_1, \gamma_2, \beta_0, \beta_1, \beta_2, \sigma_{O,0}, \delta, \sigma_C$
   - Refer to the [parameter estimation documentation](./parameter_estimation.md).

3. **Generate Random Variables**:
   - For each time step $t_n$, generate $Z_O \sim \mathcal{N}(0, 1)$ and $Z_C \sim \mathcal{N}(0, 1)$.

4. **Update Processes**:
   - Update $S_{OAS}$ and $C$ using the discretized equations:
     $$
     S_{OAS}(t_{n+1}) = S_{OAS}(t_n) + \kappa \left(S_{OAS}^{\infty} - S_{OAS}(t_n)\right) \Delta t + \left(\gamma_0 C(t_n) + \gamma_1 \sigma_r(t_n) + \gamma_2 \nu_r(t_n)\right) \Delta t + \sigma_O(t_n) \sqrt{\Delta t} \cdot Z_O 
     $$
     $$
     C(t_{n+1}) = C(t_n) + \lambda \left(C^{CC}- C(t_n)\right) \Delta t + \left(\beta_0 S_{OAS}(t_n) + \beta_1 \sigma_r(t_n) + \beta_2 \nu_r(t_n)\right) \Delta t + \sigma_C \sqrt{\Delta t} \cdot Z_C 
     $$

5. **Repeat for All Time Steps**:
   - Repeat the process for all time steps to generate the simulated paths for $S_{OAS}$ and $C$.

## Monte Carlo Simulation

Monte Carlo simulation involves running multiple simulations (paths) to estimate the expected value and distribution of the stochastic processes. This approach provides insights into the variability and uncertainty of the model predictions.

1. **Set Simulation Parameters**:
   - Number of paths (e.g., 1000)
   - Number of time steps (e.g., 252 for one year of daily data)
   - Initial conditions and model parameters

2. **Run Simulations**:
   - For each path, run the stochastic process simulation to generate the time series for $S_{OAS}$ and $C$.

3. **Calculate Expected Values**:
   - Calculate the expected value of $S_{OAS}$ and $C$ at each time step by averaging the results across all paths.

4. **Analyze Results**:
   - Analyze the distribution of the simulated paths to understand the variability and uncertainty in the model predictions.
