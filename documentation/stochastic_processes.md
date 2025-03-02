# Stochastic Processes

## Introduction

Stochastic processes are widely used in financial modeling to represent the random behavior of various financial variables. Two commonly used stochastic processes are Geometric Brownian Motion (GBM) and the mean reversion process. This document provides an overview of these processes, their discretization using the Euler-Maruyama method, and a comparison of their properties and typical use cases.

## Geometric Brownian Motion (GBM)

### Continuous-Time Model

Geometric Brownian Motion is described by the following stochastic differential equation (SDE):

$$
dX_t = \mu X_t dt + \sigma X_t dW_t
$$

where:
- $ X_t $ is the state variable.
- $ \mu $ is the drift coefficient.
- $ \sigma $ is the volatility coefficient.
- $ W_t $ is a Wiener process (standard Brownian motion).

### Discretization

To discretize the SDE using the Euler-Maruyama method, we approximate the continuous-time process with discrete-time steps. Let $ \Delta t $ be the time step size, and let $ t_n = n \Delta t $ for $ n = 0, 1, 2, \ldots $. The discretized version of the SDE is given by:

$$
X_{t_{n+1}} = X_{t_n} + \mu X_{t_n} \Delta t + \sigma X_{t_n} \sqrt{\Delta t} Z_n
$$

where $ Z_n $ are independent standard normal random variables (i.e., $ Z_n \sim \mathcal{N}(0, 1) $) representing the increments of the Wiener process.

### Properties

- **Non-Stationarity**: GBM is a non-stationary process, meaning its statistical properties, such as mean and variance, change over time.
- **Log-Normal Distribution**: The logarithm of the state variable $ X_t $ follows a normal distribution.
- **Unbounded Growth**: GBM can grow without bounds, making it suitable for modeling variables that can increase indefinitely.

### Typical Use Cases

- **Stock Prices**: GBM is commonly used to model stock prices due to its ability to capture the continuous growth and volatility observed in financial markets.
- **Commodity Prices**: GBM can also be used to model the prices of commodities that exhibit similar growth and volatility characteristics.

## Mean Reversion Process

### Continuous-Time Model

A mean reversion process, also know as Ornstein-Uhlenbeck process, is described by the following SDE:

$$
dX_t = \theta (\mu - X_t) dt + \sigma dW_t
$$

where:
- $ X_t $ is the state variable.
- $ \theta $ is the speed of reversion to the mean.
- $ \mu $ is the long-term mean level.
- $ \sigma $ is the volatility coefficient.
- $ W_t $ is a Wiener process (standard Brownian motion).

### Discretization

To discretize the mean reversion process using the Euler-Maruyama method, we approximate the continuous-time process with discrete-time steps. Let $ \Delta t $ be the time step size, and let $ t_n = n \Delta t $ for $ n = 0, 1, 2, \ldots $. The discretized version of the SDE is given by:

$$
X_{t_{n+1}} = X_{t_n} + \theta (\mu - X_{t_n}) \Delta t + \sigma \sqrt{\Delta t} Z_n
$$

where $ Z_n $ are independent standard normal random variables (i.e., $ Z_n \sim \mathcal{N}(0, 1) $) representing the increments of the Wiener process.

### Properties

- **Stationarity**: The mean reversion process is stationary, meaning its statistical properties, such as mean and variance, remain constant over time.
- **Mean-Reverting Behavior**: The process tends to move towards a long-term mean level $ \mu $ over time.
- **Bounded Fluctuations**: The state variable $ X_t $ fluctuates around the mean level $ \mu $, making it suitable for modeling variables that exhibit cyclical behavior.

### Typical Use Cases

- **Interest Rates**: Mean reversion processes are commonly used to model interest rates, which tend to revert to a long-term average over time.
- **Credit Spreads**: Mean reversion processes can be used to model credit spreads, which tend to revert to a long-term average due to changes in credit risk and economic conditions.
- **Volatility**: The mean reversion process is also used to model volatility, which often exhibits mean-reverting behavior in financial markets.

## Comparison

| Property                | Geometric Brownian Motion (GBM) | Mean Reversion Process       |
|-------------------------|---------------------------------|------------------------------|
| **Stationarity**        | Non-Stationary                  | Stationary                   |
| **Growth Behavior**     | Unbounded Growth                | Bounded Fluctuations         |
| **Distribution**        | Log-Normal                      | Normal                       |
| **Typical Use Cases**   | Stock Prices, Commodity Prices  | Interest Rates, Volatility, Credit Spreads |
| **Mean-Reverting**      | No                              | Yes                          |

## Correlated Mean Reversion Processes

### Continuous-Time Model

Two correlated mean reversion processes can be described by the following system of stochastic differential equations (SDEs):

$$
\begin{aligned}
dX_t^1 &= \theta_1 (\mu_1 - X_t^1) dt + \sigma_1 dW_t^1, \\
dX_t^2 &= \theta_2 (\mu_2 - X_t^2) dt + \sigma_2 dW_t^2,
\end{aligned}
$$

where:
- $ X_t^1 $ and $ X_t^2 $ are the state variables of the two processes.
- $ \theta_1 $ and $ \theta_2 $ are the speeds of reversion to the means $ \mu_1 $ and $ \mu_2 $, respectively.
- $ \sigma_1 $ and $ \sigma_2 $ are the volatility coefficients.
- $ W_t^1 $ and $ W_t^2 $ are Wiener processes (standard Brownian motions) with a constant correlation $ \rho $.

### Discretization

To discretize the correlated mean reversion processes using the Euler-Maruyama method, we approximate the continuous-time processes with discrete-time steps. Let $ \Delta t $ be the time step size, and let $ t_n = n \Delta t $ for $ n = 0, 1, 2, \ldots $. The discretized version of the SDEs is given by:

$$
\begin{aligned}
X_{t_{n+1}}^1 &= X_{t_n}^1 + \theta_1 (\mu_1 - X_{t_n}^1) \Delta t + \sigma_1 \sqrt{\Delta t} Z_n^1, \\
X_{t_{n+1}}^2 &= X_{t_n}^2 + \theta_2 (\mu_2 - X_{t_n}^2) \Delta t + \sigma_2 \sqrt{\Delta t} Z_n^2,
\end{aligned}
$$

where $ Z_n^1 $ and $ Z_n^2 $ are correlated standard normal random variables with correlation $ \rho $. To generate these correlated normal variables, we use the Cholesky decomposition of the correlation matrix:

$$
\begin{aligned}
\begin{bmatrix}
Z_n^1 \\
Z_n^2
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \\
\rho & \sqrt{1 - \rho^2}
\end{bmatrix}
\begin{bmatrix}
\tilde{Z}_n^1 \\
\tilde{Z}_n^2
\end{bmatrix},
\end{aligned}
$$

where $ \tilde{Z}_n^1 $ and $ \tilde{Z}_n^2 $ are independent standard normal random variables.

### Properties

- **Stationarity**: The correlated mean reversion processes are stationary, meaning their statistical properties, such as mean and variance, remain constant over time.
- **Mean-Reverting Behavior**: Both processes tend to move towards their respective long-term mean levels $ \mu_1 $ and $ \mu_2 $ over time.
- **Bounded Fluctuations**: The state variables $ X_t^1 $ and $ X_t^2 $ fluctuate around their mean levels, making them suitable for modeling variables that exhibit cyclical behavior.
- **Correlation**: The processes exhibit a constant correlation $ \rho $, which can be used to model the relationship between two mean-reverting variables.

### Typical Use Cases

- **Interest Rates**: Correlated mean reversion processes can be used to model the relationship between different interest rates that tend to revert to their long-term averages.
- **Credit Spreads**: These processes can be used to model the relationship between different credit spreads, which tend to revert to their long-term averages due to changes in credit risk and economic conditions.
- **Volatility**: Correlated mean reversion processes can also be used to model the relationship between different volatilities, which often exhibit mean-reverting behavior in financial markets.
