import numpy as np
import pandas as pd
from statsmodels.nonparametric._kernel_base import gpke
from statsmodels.nonparametric.kernels import gaussian, aitchison_aitken
from statsmodels.nonparametric.kernel_regression import KernelReg


def _est_loc_linear_weight(self, bw, exog, data_predict, index):
    nobs, k_vars = exog.shape
    ker = gpke(
        bw,
        data=exog,
        data_predict=data_predict,
        var_type=self.var_type,
        ckertype=self.ckertype,
        ukertype=self.ukertype,
        okertype=self.okertype,
        tosum=False,
    ) / float(nobs)
    # Create the matrix on p.492 in [7], after the multiplication w/ K_h,ij
    # See also p. 38 in [2]
    # ix_cont = np.arange(self.k_vars)  # Use all vars instead of continuous only
    # Note: because ix_cont was defined here such that it selected all
    # columns, I removed the indexing with it from exog/data_predict.

    # Convert ker to a 2-D array to make matrix operations below work
    ker = ker[:, np.newaxis]

    e_i = np.zeros((nobs, 1))
    e_i[index, 0] = 1.0

    M12 = exog - data_predict
    M22 = np.dot(M12.T, M12 * ker)
    M12 = (M12 * ker).sum(axis=0)
    M = np.empty((k_vars + 1, k_vars + 1))
    M[0, 0] = ker.sum()
    M[0, 1:] = M12
    M[1:, 0] = M12
    M[1:, 1:] = M22

    ker_e = ker * e_i
    V = np.empty((k_vars + 1, 1))
    V[0, 0] = ker_e.sum()
    V[1:, 0] = ((exog - data_predict) * ker_e).sum(axis=0)

    mean_mfx = np.dot(np.linalg.pinv(M), V)
    mean = mean_mfx[0]
    return mean


KernelReg._est_loc_linear_weight = _est_loc_linear_weight

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

n = 1000
x = np.linspace(0, 100, n)
y = np.array([2.0 * np.sin(x_i * np.pi / 50) for x_i in x])

exog = x[:, None]
endog = y[:, None]

mod = sm.nonparametric.KernelReg(endog, exog, var_type="c", reg_type="ll", bw=[10.0])

# It's only in the case of LOO that the predicted data is the same as the index
wght_estimate = _est_loc_linear_weight(
    mod, bw=np.array([10.0]), exog=exog, data_predict=exog[0], index=0
)


def loo_analytical(self):
    # Thing is you still have to loop to get the pred
    pass
