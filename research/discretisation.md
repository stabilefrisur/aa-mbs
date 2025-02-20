# Discretisation of Stochastic Differential Equations using Euler-Maruyama Method

## Introduction

The Euler-Maruyama method is a numerical technique used to approximate solutions to stochastic differential equations (SDEs). It is an extension of the Euler method for ordinary differential equations (ODEs) to include stochastic (random) components.

## Continuous-Time Model

Consider a general SDE of the form:

$$
dX_t = f(X_t, t) dt + g(X_t, t) dW_t
$$

where:
- \( X_t \) is the state variable.
- \( f(X_t, t) \) is the drift term.
- \( g(X_t, t) \) is the diffusion term.
- \( W_t \) is a Wiener process (standard Brownian motion).

## Euler-Maruyama Discretisation

To discretize the SDE using the Euler-Maruyama method, we approximate the continuous-time process with discrete-time steps. Let \( \Delta t \) be the time step size, and let \( t_n = n \Delta t \) for \( n = 0, 1, 2, \ldots \). The discretized version of the SDE is given by:

$$
X_{t_{n+1}} = X_{t_n} + f(X_{t_n}, t_n) \Delta t + g(X_{t_n}, t_n) \sqrt{\Delta t} Z_n
$$

where \( Z_n \) are independent standard normal random variables (i.e., \( Z_n \sim \mathcal{N}(0, 1) \)) representing the increments of the Wiener process.

## Step-by-Step Implementation

1. **Define the SDE**: Specify the drift term \( f(X_t, t) \) and the diffusion term \( g(X_t, t) \).

2. **Set Parameters**: Choose the time step size \( \Delta t \) and the number of steps \( N \).

3. **Initialize Variables**: Set the initial value \( X_0 \) and initialize arrays to store the results.

4. **Iterate Over Time Steps**:
   - For each time step \( t_n \):
     - Generate a random variable \( Z_n \) from the standard normal distribution.
     - Update the state variable \( X_{t_{n+1}} \) using the Euler-Maruyama formula.

## Example: Geometric Brownian Motion

Consider the SDE for geometric Brownian motion (GBM):

$$
dX_t = \mu X_t dt + \sigma X_t dW_t
$$

where \( \mu \) is the drift coefficient and \( \sigma \) is the volatility coefficient. The Euler-Maruyama discretization for GBM is:

$$
X_{t_{n+1}} = X_{t_n} + \mu X_{t_n} \Delta t + \sigma X_{t_n} \sqrt{\Delta t} Z_n
$$

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.1
sigma = 0.2
X0 = 1.0
T = 1.0
N = 1000
dt = T / N

# Initialize arrays
t = np.linspace(0, T, N+1)
X = np.zeros(N+1)
X[0] = X0

# Euler-Maruyama method
for n in range(N):
    Z = np.random.normal()
    X[n+1] = X[n] + mu * X[n] * dt + sigma * X[n] * np.sqrt(dt) * Z

# Plot the results
plt.plot(t, X)
plt.xlabel('Time')
plt.ylabel('X')
plt.title('Geometric Brownian Motion using Euler-Maruyama Method')
plt.show()
```

