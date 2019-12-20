Algorithms and examples for importance-weighted imbalanced regression / __regression under covariate shift__.

**Example 1: Static, nonlinear regression.** Uses the Monte Carlo method proposed by Cook & Nachtsheim (1994) for importance estimation.
Run file `static.m`.

**Example 2: Nonlinear dynamical system estimation.** Uses the online kernel density estimation proposed by Matej, Ales and Danijel (2011).
This example requires the "maggot" toolbox to be installed and available on the Matlab path, which is described [here](http://www.vicos.si/Research/Multivariate_Online_Kernel_Density_Estimation) available for Matlab [here](http://www.vicos.si/File:Maggot_v3.5.zip).
Run file `sysid.m`.

---
Eike Petersen and Philipp Rostalski, December 2019

Institute for Electrical Engineering in Medicine

Universität zu Lübeck

Germany

eike.petersen@uni-luebeck.de

---
References:
- R. Dennis Cook and Christopher J. Nachtsheim. Reweighting to achieve elliptically contoured
covariates in regression. Journal of the American Statistical Association, 89(426):592–599, jun 1994. doi: 10.1080/01621459.1994.10476784.
- Matej Kristan, Ales Leonardis, and Danijel Skocaj. Multivariate online kernel density estimation
with gaussian kernels. Pattern Recognition, 44(10-11):2630–2642, oct 2011. doi: 10.1016/j.
patcog.2011.03.019.
