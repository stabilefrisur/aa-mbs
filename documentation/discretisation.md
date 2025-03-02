# Discretisation of Stochastic Differential Equations

The Euler-Maruyama method is a numerical technique used to approximate solutions to stochastic differential equations (SDEs). It is an extension of the Euler method for ordinary differential equations (ODEs) to include stochastic (random) components.

## Continuous-Time Model

Consider a general SDE of the form:

$$
dX_t = f(X_t, t) dt + g(X_t, t) dW_t
$$

where:
- $ X_t $ is the state variable.
- $ f(X_t, t) $ is the drift term.
- $ g(X_t, t) $ is the diffusion term.
- $ W_t $ is a Wiener process (standard Brownian motion).

## Euler-Maruyama Discretisation

To discretize the SDE using the Euler-Maruyama method, we approximate the continuous-time process with discrete-time steps. Let $ \Delta t $ be the time step size, and let $ t_n = n \Delta t $ for $ n = 0, 1, 2, \ldots $. The discretized version of the SDE is given by:

$$
X_{t_{n+1}} = X_{t_n} + f(X_{t_n}, t_n) \Delta t + g(X_{t_n}, t_n) \sqrt{\Delta t} Z_n
$$

where $ Z_n $ are independent standard normal random variables (i.e., $ Z_n \sim \mathcal{N}(0, 1) $) representing the increments of the Wiener process.
