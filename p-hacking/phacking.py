import sys

import numpy as np
from numpy import sqrt
from numpy.random import randn, randint
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF


# @jit()
def isomean(y, w):
    nn = len(y)
    if nn == 1:
        return y
    k = np.zeros(nn, dtype=int)
    gew = np.zeros(nn)
    ghat = np.zeros(nn)
    c = 0
    k[c] = 0
    gew[c] = w[0]
    ghat[c] = y[0]
    for j in range(nn - 1):
        c += 1
        k[c] = j + 1
        gew[c] = w[j + 1]
        ghat[c] = y[j + 1]
        while ghat[c - 1] >= ghat[c]:
            neu = gew[c] + gew[c - 1]
            ghat[c - 1] = ghat[c - 1] + (gew[c] / neu) * (ghat[c] - ghat[c - 1])
            gew[c - 1] = neu
            c -= 1
            if c == 0:
                break
    while nn >= 1:
        for j in range(k[c], nn, 1):
            ghat[j] = ghat[c]
        nn = k[c]
        c -= 1
    return ghat


# @jit()
def gcmlcm(x, y, typex='LCM'):
    if len(np.unique(x)) != len(x):
        sys.exit(-1)
    dx = np.diff(x)
    dy = np.diff(y)
    if (dx < 0).any():
        sys.exit(-1)
    rawslope = dy / dx
    noninf = rawslope[np.abs(rawslope) != np.inf]
    xmax = np.max(noninf)
    xmin = np.min(noninf)
    rawslope[rawslope == np.inf] = xmax
    rawslope[rawslope == -np.inf] = xmin
    if typex.lower() == 'gcm':
        slope = isomean(rawslope, dx)
    elif typex.lower() == 'lcm':
        slope = isomean(-rawslope, dx)
    temp, keep = np.unique(slope, return_index=True)
    xxx = keep.tolist().copy()
    xxx.append(len(x) - 1)
    xknots = x[xxx]
    dxknots = np.diff(xknots)
    slopeknots = slope[keep]
    yknots = y[1] + np.concatenate((np.zeros(1), np.cumsum(dxknots * slopeknots)))
    return xknots, yknots, slopeknots


# @jit()
def SimulateBrownianBridge(M):
    N = 10000
    x = (np.arange(N) + 1) / N
    BBsup = np.zeros((M, 1))
    for m in range(M):
        eps = randn(N) / N
        W = np.cumsum(eps)
        B = W - x * W[-1]
        C = np.concatenate((np.zeros(1), x))
        B = np.concatenate((np.zeros(1), B))
        xknots, yknots, slopeknots = gcmlcm(C, B, typex='LCM')
        f_lcm = interp1d(xknots, yknots)
        y = f_lcm(C)
        BBsup[m, 0] = np.max(np.abs(y - B))
    return BBsup


def LCM(P, p_min, p_max):
    P = P[(P <= p_max) & (P >= p_min)]
    nn = len(P)
    f = ECDF(P)
    x = (np.arange(0, 1000) + 1) / 1000
    y = f(x * (p_max - p_min) + p_min)
    xknots, yknots, slopeknots = gcmlcm(x, y, typex="lcm")
    ff_lcm = interp1d(xknots, yknots)
    z = ff_lcm(x)
    Test_stat = sqrt(nn) * np.max(np.abs(y - z))
    SBB_data = SimulateBrownianBridge(10)
    F_LCMsup = ECDF(SBB_data.reshape(-1))
    return 1 - F_LCMsup(Test_stat)


print(1)

import matplotlib.pyplot as plt
import pandas as pd

x = np.random.randint(0, 100, 10000) / 100
d1 = pd.DataFrame(np.round(x, 1).astype(float), columns=['x'])
d1['y'] = d1['x']
df = pd.pivot_table(d1, index=['x'], values=['y'], aggfunc=len)
df['x'] = df.index
df.index = range(len(df))

df.plot(x='x', y='y', kind='scatter')
# plt.scatter(x)
plt.show()

x = np.random.rand(1000)
x1 = LCM(P=x, p_min=np.min(x), p_max=np.max(x))
# print(slopeknots)
