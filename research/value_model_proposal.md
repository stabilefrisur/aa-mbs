# Agency MBS Valuation Model Proposal

## 1. Model Derivation

### High-Level Concept

MBS exhibit negative convexity due to the prepayment option. As interest rates decline, the price appreciation of MBS is limited by the increasing likelihood of prepayments. As interest rates rise, the price depreciation is exacerbated by the decreasing likelihood of prepayments. This negative convexity makes MBS valuation more complex. Credit spreads are more linear in their response to changes in interest rates and credit risk.

An MBS valuation model should thus consider spread reversion, prepayment optionality and their confluence. Provided ZV and OAS spreads are available, we can think of the differential as the embedded optionality or Convexity.

$$
S_{ZV} = S_{OAS} + C
$$

- **Zero Volatility Spread (ZV)**: Spread over interest rates, inclusive of optionality.
- **Option-Adjusted Spread (OAS)**: Spread after removing the impact of embedded options, particularly prepayments.
- **Convexity**: Defined as the differential between the Z-Spread and OAS, capturing the optionality premium intrinsic to MBS.

 However, changes in OAS cannot be thought of as independent from Convexity. The dynamics of both OAS and Convexity are also likely influenced by interest rates. 

 I propose a **Joint Reversion Model** for OAS and Convexity, explicitly incorporating external factors such as interest rate volatility ($\sigma_r$) and volatility of volatility ($\nu_r$), while also accounting for model uncertainty inherent in OAS estimation.

### Model Components

#### 1. Joint Reversion Process for OAS and Convexity

- OAS follows a mean-reverting process toward a long-run equilibrium $S_{OAS}^{\infty}$, integrating convexity and external influences:
  
  $$
  dS_{OAS} = -\kappa \left(S_{OAS} - S_{OAS}^{\infty} - \gamma_0 C - \gamma_1 \sigma_r - \gamma_2 \nu_r\right) dt + \sigma_O dW_O.
  $$
  
- Convexity also follows an auto-regressive process, with its equilibrium level influenced by OAS and external market factors:
  
  $$
  dC = -\lambda \left(C - \beta_0 S_{OAS} - \beta_1 \sigma_r - \beta_2 \nu_r\right) dt + \sigma_C dW_C.
  $$

- $\kappa$ and $\lambda$ are parameters which describe the speed of reversion of the OAS and Convexity processes to their respective equilibriums.

- $W_O$ and $W_C$ are Wiener processes that introduce stochasticity into the OAS and Convexity processes. Scaled to the respective standard deviations of both processes, $\sigma_O dW_O$ and $\sigma_C dW_C$ capture those random fluctuations. 

#### 2. Incorporation of Interest Rate Volatility and Volatility of Volatility

- **Rates Vol ($\sigma_r$)**: Influences both OAS and Convexity by altering prepayment risk and uncertainty.
- **Rates Vol of Vol ($\nu_r$)**: Introduces regime-dependent fluctuations, affecting both the risk premium and convexity.

#### 3. Accounting for OAS Model Uncertainty

- OAS estimation depends on prepayment models, which become increasingly unreliable under heightened convexity conditions.
- We can introduce a convexity-dependent residual variance, ensuring greater dispersion in OAS innovations when convexity is elevated.
  
  $$
  \sigma_O^2 = \sigma_{O,0}^2 + \delta C^2
  $$

### Discretisation of the System of SDEs

To discretize the system of differential equations in the Joint Reversion Model for OAS and Convexity, we can use the Euler-Maruyama method, which is a straightforward approach for numerical integration of stochastic differential equations (SDEs). Here’s a step-by-step explanation of the process:

#### Continuous-Time Model
The continuous-time model for OAS and Convexity is given by:

1. **OAS Process**: $ dS_{OAS} = -\kappa \left(S_{OAS} - S_{OAS}^{\infty} - \gamma_0 C - \gamma_1 \sigma_r - \gamma_2 \nu_r\right) dt + \sigma_O dW_O $

2. **Convexity Process**: $ dC = -\lambda \left(C - \beta_0 S_{OAS} - \beta_1 \sigma_r - \beta_2 \nu_r\right) dt + \sigma_C dW_C $

#### Discretisation Using Euler-Maruyama Method
To discretise these equations, we approximate the continuous-time processes with discrete-time steps. Let $\Delta t$ be the time step size, and let $t_n = n \Delta t$ for $n = 0, 1, 2, \ldots$. The discretized versions of the SDEs are:

1. **Discretized OAS Process**: $ S_{OAS}(t_{n+1}) = S_{OAS}(t_n) - \kappa \left(S_{OAS}(t_n) - S_{OAS}^{\infty} - \gamma_0 C(t_n) - \gamma_1 \sigma_r(t_n) - \gamma_2 \nu_r(t_n)\right) \Delta t + \sigma_O \sqrt{\Delta t} \cdot Z_O $

2. **Discretized Convexity Process**: $ C(t_{n+1}) = C(t_n) - \lambda \left(C(t_n) - \beta_0 S_{OAS}(t_n) - \beta_1 \sigma_r(t_n) - \beta_2 \nu_r(t_n)\right) \Delta t + \sigma_C \sqrt{\Delta t} \cdot Z_C $

Here, $Z_O$ and $Z_C$ are independent standard normal random variables (i.e., $Z_O, Z_C \sim \mathcal{N}(0, 1) $) representing the Wiener processes $dW_O$ and $dW_C$.

### Predicted Market Behaviour

Recovery (High Growth + Falling Inflation):
- Interest Rates: Likely to rise as the economy grows, leading to elevated OAS.
- Credit Spreads: May tighten due to improved economic conditions.
- Prepayments: Likely to remain stable, reducing convexity.

Boom (High Growth + Rising Inflation):
- Interest Rates: Likely to rise significantly due to inflationary pressures, leading to increased OAS.
- Credit Spreads: May widen slightly due to inflation concerns.
- Prepayments: Likely to remain stable, reducing convexity.

Downturn (Low Growth + Falling Inflation):
- Interest Rates: Likely to fall as the central bank cuts rates to stimulate the economy, leading to declining OAS.
- Credit Spreads: Likely to widen due to increased credit risk.
- Prepayments: Likely to accelerate as borrowers refinance at lower rates, increasing convexity.

Stagflation (Low Growth + Rising Inflation):
- Interest Rates: Likely to rise due to inflationary pressures despite low growth, leading to increased OAS.
- Credit Spreads: Likely to widen significantly due to economic uncertainty.
- Prepayments: Behavior may be uncertain, potentially rising convexity due to mixed signals in the market.

## 2. Implementation and Estimation

### Step-by-Step Implementation

1. **Data Acquisition and Preprocessing**:
   - Collect historical time series for OAS, Z-Spread, interest rate vol ($\sigma_r$), and vol of vol ($\nu_r$).
   - Compute convexity as $C = S_Z - S_{OAS}$.
   - Standardise and clean data to remove structural breaks and extreme outliers.

2. **Mean Reversion Parameter Estimation**:
   - Estimate parameters $\kappa, \lambda, \gamma, \beta$ using historical data.
   - Perform Ordinary Least Squares (OLS) regression on mean-reverting OAS and convexity dynamics.
   - Conduct stationarity tests (e.g., Augmented Dickey-Fuller) to validate model assumptions.
   - Use the discretized equations to simulate the mean-reverting processes and compare with historical data for validation.

3. **Residual Variance Calibration**:
   - Fit the convexity-dependent variance function $\sigma_O^2 = \sigma_{O,0}^2 + \delta C^2$ via non-linear least squares.
   - Assess statistical significance of convexity-driven uncertainty effects.
   - Incorporate the discretized variance function into the simulation model to capture the impact of convexity on OAS innovations.

4. **Global Model Calibration**:
   - Use numerical optimisation (e.g., maximum likelihood estimation, Nelder-Mead) to calibrate parameters.
   - Validate performance using out-of-sample tests.
   - Implement the discretized model equations in the calibration process to ensure consistency between the continuous-time model and its discrete-time implementation.

5. **Discretization of the System of SDEs**:
   - Discretize the continuous-time differential equations using the Euler-Maruyama method.
   - Define the time step size $\Delta t$ and the number of steps $N$.
   - Initialize parameters and variables:
     - Set initial values for $S_{OAS}(0)$ and $C(0)$.
     - Define the parameters $\kappa, \lambda, \gamma_0, \gamma_1, \gamma_2, \beta_0, \beta_1, \beta_2, \sigma_O, \sigma_C$.
   - Iterate over time steps:
     - For each time step $t_n$:
       - Generate random variables $Z_O$ and $Z_C$.
       - Update $S_{OAS}$ and $C$ using the discretized equations:

6. **Sensitivity Analysis and Stress Testing**:
   - Simulate model responses to diverse economic scenarios.
   - Test extreme cases such as volatility spikes and rapid prepayment shifts.

7. **Operational Implementation**:
   - Automate parameter updates.
   - Incorporate real-time adjustments to account for shifts in rate vol and prepayment behaviour.

## 3. Alternative Approaches

| Methodology                               | Description                                | Strengths                          | Weaknesses                        | Optimal Use Case                     |
| ----------------------------------------- | ------------------------------------------ | ----------------------------------- | ---------------------------------- | ---------------------------------- |
| **Joint Reversion Model (Proposed)**     | Mean-reverting dynamics for OAS and Convexity | Captures structural dependencies, incorporates exogenous factors | Requires careful calibration       | Time-series valuation modelling       |
| **Panel Regression**                      | Cross-sectional analysis of MBS spreads    | Leverages large datasets            | Ignores time-series dependencies, requires large cross-sectional sample  | Index aggregated spread dynamics         |
| **State-Space Model (SSM)**               | Kalman filter-based latent factor modelling | Captures hidden states dynamically | Computationally demanding          | High-frequency filtering applications |
| **Markov-Switching Model (MSM)**          | Regime-switching framework                 | Models non-linear transitions      | Hard to estimate transition probabilities | Capturing regime-dependent spread shifts |

## 4. Accounting for Funding Costs in TBA vs. Pools

- **TBA Market Characteristics**:
  - Greater liquidity but introduces cheapest-to-deliver risk.
  - Subject to roll dynamics and funding market conditions.

- **Specified Pools**:
  - Lower liquidity, but offer cash flow precision.
  - Require adjustments for funding costs and potential roll financing.

- **Spread Adjustment Framework**:
  
  $$
  S_{TBA} = S_{Pools} - \text{Funding Adjustment}.
  $$
  
  - Funding costs are influenced by repo rates, collateral availability, and roll specialness.
  - Elevated repo costs induce widening spreads between TBAs and pools.
  - Empirical models for roll costs provide enhanced valuation insights.
  - Liquidity stress scenarios should be incorporated into spread forecasting.
  - Differential pricing should account for varying dollar roll specialness by settlement structure and security type.

