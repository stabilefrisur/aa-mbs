# Estimation of the Joint Reversion Model

This document describes the process of estimating the parameters of the Joint Reversion Model for OAS and Convexity. The estimation process involves two steps: first, using Ordinary Least Squares (OLS) to obtain initial parameter estimates, and then refining these estimates using Maximum Likelihood Estimation (MLE). The final estimated parameters are used to simulate the stochastic processes.

## Ordinary Least Squares (OLS)

### Overview

Ordinary Least Squares (OLS) is a method for estimating the parameters of a linear regression model. The goal of OLS is to minimize the sum of the squared differences between the observed values and the values predicted by the model.

### Mathematical Formulation

Given a linear model:
$$
y = X\beta + \epsilon
$$
where:
- $ y $ is the vector of observed values.
- $ X $ is the matrix of explanatory variables.
- $ \beta $ is the vector of parameters to be estimated.
- $ \epsilon $ is the vector of errors.

The OLS estimator of $ \beta $ is given by:
$$
\hat{\beta} = (X^TX)^{-1}X^Ty
$$

### Application to the Joint Reversion Model

For the Joint Reversion Model, we have two processes: OAS and Convexity. The discretized versions of these processes are:

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

We perform OLS regression on the discretized equations to estimate the parameters $ \kappa, \lambda, \gamma, \beta $.

## Maximum Likelihood Estimation (MLE)

### Overview

Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a statistical model. The goal of MLE is to find the parameter values that maximize the likelihood function, which measures the probability of observing the given data under the model.

### Mathematical Formulation

Given a likelihood function $ L(\theta; y) $, where $ \theta $ is the vector of parameters and $ y $ is the observed data, the MLE estimator of $ \theta $ is given by:
$$
\hat{\theta} = \arg\max_{\theta} L(\theta; y)
$$

### Log-Likelihood Function for the Joint Reversion Model

For the Joint Reversion Model, the log-likelihood function is derived from the discretized equations. The log-likelihood function for the OAS and Convexity processes is given by:

1. **Log-Likelihood for OAS**:
   $$
   \log L_{OAS} = \sum_{t=1}^{n} \left[ -\frac{1}{2} \log(2 \pi \sigma_O(t)^2 \Delta t) - \frac{(S_{OAS}(t) - S_{OAS,\text{mean}}(t))^2}{2 \sigma_O(t)^2 \Delta t} \right]
   $$
   where
   $$
   S_{OAS,\text{mean}}(t) = S_{OAS}(t-1) + \kappa (S_{OAS}^{\infty} - S_{OAS}(t-1)) \Delta t + (\gamma_0 C(t-1) + \gamma_1 \sigma_r(t-1) + \gamma_2 \nu_r(t-1)) \Delta t
   $$

2. **Log-Likelihood for Convexity**:
   $$
   \log L_{C} = \sum_{t=1}^{n} \left[ -\frac{1}{2} \log(2 \pi \sigma_C^2 \Delta t) - \frac{(C(t) - C_{\text{mean}}(t))^2}{2 \sigma_C^2 \Delta t} \right]
   $$
   where
   $$
   C_{\text{mean}}(t) = C(t-1) + \lambda (C^{CC} - C(t-1)) \Delta t + (\beta_0 S_{OAS}(t-1) + \beta_1 \sigma_r(t-1) + \beta_2 \nu_r(t-1)) \Delta t
   $$

The total log-likelihood function is the sum of the log-likelihoods for OAS and Convexity:
$$
\log L = \log L_{OAS} + \log L_{C}
$$

### Estimation Process

1. **Initial Guess Using OLS**:
   - Perform OLS regression on the discretized equations to obtain initial estimates for the parameters $ \kappa, \lambda, \gamma, \beta $.
   - Use these OLS estimates as the initial guess for the MLE optimization.

2. **MLE Optimization**:
   - Define the log-likelihood function based on the discretized equations.
   - Use numerical optimization techniques (e.g., L-BFGS-B) to maximize the log-likelihood function and obtain the MLE estimates for the parameters.

## Using OLS Results as Initial Guesses for MLE

### Benefits

1. **Improved Convergence**:
   - MLE involves optimizing a potentially complex likelihood function, which can have multiple local maxima. Providing good initial guesses can help the optimization algorithm converge more quickly and reliably to the global maximum.

2. **Reduced Computational Cost**:
   - Starting with reasonable initial estimates can reduce the number of iterations required for the optimization algorithm to converge, thereby saving computational resources and time.

3. **Stability**:
   - Initial guesses that are close to the true parameter values can improve the numerical stability of the optimization process, reducing the risk of encountering issues such as non-convergence or divergence.

4. **Accuracy**:
   - OLS estimates, while not always the most efficient, are often unbiased and provide a good starting point. This can lead to more accurate MLE estimates, as the optimization process refines these initial values.

### Practical Considerations

1. **Model Complexity**:
   - For simple models, OLS estimates might be sufficient on their own. However, for more complex models, especially those involving non-linear relationships or stochastic processes, MLE can provide more efficient and accurate parameter estimates.

2. **Data Characteristics**:
   - The quality of the initial OLS estimates can depend on the characteristics of the data, such as the presence of autocorrelation, heteroscedasticity, or non-linearity. In such cases, MLE can better account for these complexities.

3. **Computational Resources**:
   - While MLE can be computationally intensive, the use of good initial guesses from OLS can mitigate this issue, making the process more feasible even with limited computational resources.
   