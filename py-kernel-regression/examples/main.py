import numpy as np
import pandas as pd 
from statsmodels.nonparametric._kernel_base import gpke
from statsmodels.nonparametric.kernels import gaussian, aitchison_aitken
from py_kernel_regression import loc_constant_fit


bw = [1.0, 0.2]
x_train = np.array([1.0, 3.2, 2.5, 1.2, 4.3])[:, None]
u_train = np.array([1, 3, 2, 2, 1])[:, None]
X_train = np.concatenate([x_train, u_train], axis=1)
Y_train = np.array([9., 9., 10., 3., 4.])

x_new = np.array([[1.0, 2.0], [2.2, 3.0], [2.6, 2.0]])

def main():
    lc_output = loc_constant_fit(bw, Y_train, X_train, x_new, ["c", "u"])
    print(lc_output)

if __name__ == "__main__":
    main()
