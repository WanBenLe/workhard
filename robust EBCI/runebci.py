import pandas as pd
from numpy import mean,isnan,ones,hstack
from statsmodels.api import add_constant
from ebciest import ebci, robust_ebci, length_fct
from cvaest import cva, norm_ppf_f

dat = pd.read_csv('cz.csv')
print(dat.columns)
dat = dat[~isnan(dat.theta25)]


Y = dat.theta25.values.reshape(-1, 1)
sigma = dat.se25.values.reshape(-1, 1)
n = len(Y)
X = hstack((ones((n, 1)), dat.stayer25.values.reshape(-1, 1)))
weights = (1 / sigma ** 2).reshape(-1, 1)
alpha = 0.1
thetahat, ci, w_estim, normlng, mu2, kappa, delta = ebci(Y, X, sigma, alpha, weight=weights)
#print(thetahat, ci, w_estim, normlng, mu2, kappa, delta)
ci_lo = ci[:,0]
ci_up = ci[:,1]

print('Average length of unshrunk CIs relative to robust EBCIs')
print((2*norm_ppf_f(1-alpha/2)*mean(sigma))/mean(ci_up-ci_lo))



'''
from line_profiler import LineProfiler
p = LineProfiler()
p_wrap = p(ebci)
p_wrap(Y, X, sigma, alpha, weight=weights)
p.print_stats()
'''

print(1)
