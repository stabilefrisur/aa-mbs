- [x] add discretisation.md describing Euler Maryuama in detail
- [ ] amend mock data assumptions to match actual MBS stats
- [x] check if mock data volatility is scaled correctly; may need only one parameter
- [x] add mock_data.md describing how mock data are generated
- [x] allow 2 options to provide sigma_r and nu_r: fixed value; simulated series of same lenght as steps
- [x] convexity as modelled is not mean reverting but still autoregressive; reconsider the factors that drive changes in convexity
- [x] reserve "xsam" package name on PyPi
- [ ] revise signs of parameters. Shouldn't have minus signs everywhere as that's hard to read. Should be 

$$
dS_{OAS} = \kappa \left(S_{OAS}^{\infty} - S_{OAS} + \gamma_0 C + \gamma_1 \sigma_r + \gamma_2 \nu_r\right) dt + \sigma_O dW_O.
$$

$$
dC = \lambda \left(C^{CC} - C + \beta_0 S_{OAS} + \beta_1 \sigma_r + \beta_2 \nu_r\right) dt + \sigma_C dW_C.
$$