# Stochastic Processes: Geometric Brownian Motion and Mean Reversion

## Introduction

Stochastic processes are widely used in financial modeling to represent the random behavior of various financial variables. Two commonly used stochastic processes are Geometric Brownian Motion (GBM) and the mean reversion process. This document provides an overview of these processes, their discretization using the Euler-Maruyama method, and a comparison of their properties and typical use cases.

## Geometric Brownian Motion (GBM)

### Continuous-Time Model

Geometric Brownian Motion is described by the following stochastic differential equation (SDE):

$$
dX_t = \mu X_t dt + \sigma X_t dW_t
$$

where:
- \( X_t \) is the state variable.
- \( \mu \) is the drift coefficient.
- \( \sigma \) is the volatility coefficient.
- \( W_t \) is a Wiener process (standard Brownian motion).

### Discretization Using Euler-Maruyama Method

To discretize the SDE using the Euler-Maruyama method, we approximate the continuous-time process with discrete-time steps. Let \( \Delta t \) be the time step size, and let \( t_n = n \Delta t \) for \( n = 0, 1, 2, \ldots \). The discretized version of the SDE is given by:

$$
X_{t_{n+1}} = X_{t_n} + \mu X_{t_n} \Delta t + \sigma X_{t_n} \sqrt{\Delta t} Z_n
$$

where \( Z_n \) are independent standard normal random variables (i.e., \( Z_n \sim \mathcal{N}(0, 1) \)) representing the increments of the Wiener process.

### Properties

- **Non-Stationarity**: GBM is a non-stationary process, meaning its statistical properties, such as mean and variance, change over time.
- **Log-Normal Distribution**: The logarithm of the state variable \( X_t \) follows a normal distribution.
- **Unbounded Growth**: GBM can grow without bounds, making it suitable for modeling variables that can increase indefinitely.

### Typical Use Cases

- **Stock Prices**: GBM is commonly used to model stock prices due to its ability to capture the continuous growth and volatility observed in financial markets.
- **Commodity Prices**: GBM can also be used to model the prices of commodities that exhibit similar growth and volatility characteristics.

## Mean Reversion Process

### Continuous-Time Model

A mean reversion process is described by the following SDE:

$$
dX_t = \theta (\mu - X_t) dt + \sigma dW_t
$$

where:
- \( X_t \) is the state variable.
- \( \theta \) is the speed of reversion to the mean.
- \( \mu \) is the long-term mean level.
- \( \sigma \) is the volatility coefficient.
- \( W_t \) is a Wiener process (standard Brownian motion).

### Discretization Using Euler-Maruyama Method

To discretize the mean reversion process using the Euler-Maruyama method, we approximate the continuous-time process with discrete-time steps. Let \( \Delta t \) be the time step size, and let \( t_n = n \Delta t \) for \( n = 0, 1, 2, \ldots \). The discretized version of the SDE is given by:

$$
X_{t_{n+1}} = X_{t_n} + \theta (\mu - X_{t_n}) \Delta t + \sigma \sqrt{\Delta t} Z_n
$$

where \( Z_n \) are independent standard normal random variables (i.e., \( Z_n \sim \mathcal{N}(0, 1) \)) representing the increments of the Wiener process.

### Properties

- **Stationarity**: The mean reversion process is stationary, meaning its statistical properties, such as mean and variance, remain constant over time.
- **Mean-Reverting Behavior**: The process tends to move towards a long-term mean level \( \mu \) over time.
- **Bounded Fluctuations**: The state variable \( X_t \) fluctuates around the mean level \( \mu \), making it suitable for modeling variables that exhibit cyclical behavior.

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
