#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:45:17 2021

@author: Amelie
"""


import numpy as np
import statsmodels.api as sm
import scipy as sp

data = sm.datasets.longley.load(as_pandas=False)
data.exog = sm.add_constant(data.exog)
model = sm.OLS(data.endog, data.exog).fit()
A = np.identity(len(model.params))
A = A[1:,:] # This test if all coefficient other than itercept (i.e. intercept = coeff[0,:]) are statistically different from zero
            # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.f_test.html
ftest = model.f_test(A)
# print(ftest.fvalue[0],ftest.pvalue)
F = ftest.fvalue[0][0]
p = ftest.pvalue
dfn = ftest.df_num
dfd = ftest.df_denom

pcrit = 0.05
Fcrit = sp.stats.f.ppf(q=1-pcrit/2, dfn=dfn, dfd=dfd) # Two-tailed test
print(F, Fcrit, F > Fcrit)
pcrit = 0.01
Fcrit = sp.stats.f.ppf(q=1-pcrit/2, dfn=dfn, dfd=dfd) # Two-tailed test
print(F, Fcrit, F > Fcrit)

# p = 1-scipy.stats.f.cdf(F, dfn=dfn, dfd=dfd)


def r_to_z(r):
    return np.log((1 + r) / (1 - r)) / 2.0

def z_to_r(z):
    e = np.exp(2 * z)
    return((e - 1) / (e + 1))

def r_confidence_interval(r, alpha, n, tailed = 'two'):
    z = r_to_z(r)
    se = 1.0 / np.sqrt(n - 3)
    if tailed == 'two': z_crit = sp.stats.norm.ppf(1 - alpha/2)  # 2-tailed z critical value
    if tailed == 'one': z_crit = sp.stats.norm.ppf(1 - alpha)  # 1-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    # Return a sequence
    return (z_to_r(lo), z_to_r(hi))





