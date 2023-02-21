'''
<iterGMM est code.>
<Ben Wan>
from
https://www.econometricsociety.org/publications/econometrica/2021/05/01/inference-iterated-gmm-under-misspecification
Bruce E. Hansen, Seojeong Lee
'''

import numpy as np
import pandas as pd
from numba import jit
from numpy import kron, ones, dot, zeros, sqrt, diag, max, min, sum, nan, hstack, vstack, isnan
from scipy.linalg import inv
from scipy.stats import chi2, t


def tmat(mat):
    return np.transpose(mat)


def repmat(a, b, c):
    return kron(ones((c, b)), a)


# arellano_bond
def arellano_bond(d, y, x, year, country):
    '''
    ARELLANO, M., AND S. BOND (1991): “Some Tests of Specification for Panel Data: Monte Carlo Evidence and
    an Application to Employment Equations,” Review of Economic Studies, 58 (2), 277–297. [1436]

    Blundell, Richard; Bond, Stephen (1998). "Initial conditions and moment restrictions in dynamic panel data models".
    Journal of Econometrics. 87(1): 115–143


    d         n0x1 vector of dependent variable
    y         n0x1 vector of main regressor
    x         n0xm matrix of covariates (if none set x=[])
    year      n0x1 vector of year (index j within cluster)
    country   n0x1 vector of country code (cluster membership)

    Outputs:
    Dd      nx1 vector of differenced dependent variable
    DX      nx(m+1) matrix of differenced main regressor and covariates
    Z       nxl matrix of instruments
    T is the largest (most recent in time) value across i (countries)
    For those with missing t replace the row with the row of zeros
    mem     nx1 vector of cluster membership
    n       1x1 scalar of the sample size
    G       1x1 scalar of the number of clusters
    ng      Gx1 vector of cluster size
    '''
    # d0=array([np.nan,0.2])

    d0 = d
    n0 = len(year)

    dimx = np.shape(x)[1]
    T = int(len(np.unique(year)))
    yr = tmat(repmat(year, 1, T) == tmat(repmat(np.unique(year), 1, n0)))
    d_lag = zeros((n0, 2))
    y_lag = zeros((n0, 2 * y.shape[1]))
    yr_lag = zeros((n0, T))
    x_lag = zeros((n0, 2 * dimx))
    rangex = int(n0 / T)
    for xaha in range(rangex):
        i = xaha + 1
        d_lag[T * (i - 1): (T * i), :] = lagx(d[(T * (i - 1)): (T * i)], [1, 2])
        for ix in range(int(y.shape[1])):
            y_lag[T * (i - 1): (T * i), 2 * ix:2 * (ix + 1)] = lagx(y[(T * (i - 1)): (T * i), ix].reshape(-1, 1),
                                                                    [1, 2])
        yr_lag[T * (i - 1): (T * i), :] = lagx(yr[(T * (i - 1)): (T * i), :], [1] * yr_lag.shape[1])

        if dimx > 0:
            for ix in range(int(x.shape[1])):
                x_lag[T * (i - 1): (T * i), 2 * ix:2 * (ix + 1)] = lagx(x[T * (i - 1): (T * i), ix].reshape(-1, 1),
                                                                        [1, 2])

    if dimx > 0:
        data = hstack((d.reshape(-1, 1), d_lag, y_lag, y, yr, yr_lag, year.reshape(-1, 1),
                       country.reshape(-1, 1), x, x_lag))
    else:
        data = hstack((d.reshape(-1, 1), d_lag, y_lag, y, yr, yr_lag, year.reshape(-1, 1),
                       country.reshape(-1, 1)))
    temp1 = hstack((d.reshape(-1, 1), d_lag, y_lag))
    temp = hstack((temp1, data[:, 2 * T + 8 + dimx: data.shape[1] - 1]))
    temp = tmat(1 - tmat(np.isnan(temp)))
    temp1 = zeros((len(temp)))
    for i in range(len(temp)):
        temp1[i] = min(temp[i, :])
    ind = temp1 == 1
    # del temp
    data = data[ind, :]
    code = data[:, - (dimx * 3 + 1)]
    cc, mem = np.unique(code, return_inverse=True)
    G = int(len(cc))
    n = len(code)
    g_ng = conutnum(code[:], cc)
    ng = g_ng[:, 1]
    mxng = int(np.max(ng))

    d = data[:, 0]
    d1 = data[:, 1]
    d2 = data[:, 2]

    y1 = data[:, 3:][:, np.arange(0, y.shape[1] * 2, 2)]
    y2 = data[:, 3:][:, np.arange(1, y.shape[1] * 2, 2)]

    yr = data[:, 18:18 + T]
    yr1 = data[:, 18 + T:18 + T + T]

    x1 = data[:, -dimx * 2:][:, np.arange(0, dimx * 2, 2)]
    x2 = data[:, -dimx * 2:][:, np.arange(1, dimx * 2, 2)]

    Dyr = yr - yr1

    Dd = d - d1
    if dimx > 0:
        DX = hstack(((d1 - d2).reshape(-1, 1), (y1 - y2), (x1 - x2), Dyr[:, int(T - mxng): Dyr.shape[1]]))
    else:
        DX = hstack(((d1 - d2).reshape(-1, 1), (y1 - y2), Dyr[:, int(T - mxng): Dyr.shape[1]]))

    d_ = zeros((T, G))
    dc = hstack((d0.reshape(-1, 1), country.reshape(-1, 1)))
    for i in range(G):
        d_[:, i] = d0[dc[:, 1] == cc[i]].reshape(-1)
    d_[isnan(d_)] = 0
    d_ = d_[int(T - 2 - mxng):(T - 2), :]

    Z = []

    for i in range(G):
        Zi = zeros((mxng, int(np.sum(np.arange(mxng + 1)))))
        for j in range(mxng):
            Zi[j, sum(np.arange(j + 1)): sum(np.arange(j + 2))] = tmat(d_[0: j + 1, i])
        row = data[:, 2 * T + 7] == cc[i]
        row = data[row, 2 * T + 6] - (max(year) - max(ng)) - 1
        row = row.astype(int)
        temp1 = Zi[row, :]
        a1 = int(sum(ng[0: i + 1]) - ng[i])
        a2 = int(sum(ng[0:i + 1]))
        temp2 = y2[a1:a2]
        Zi = hstack((temp1, temp2.reshape(-1, 1)))
        if dimx > 0:
            temp1 = x2[a1:a2]
            Zi = hstack((Zi, temp1))
        temp1 = Dyr[a1:a2, (T - mxng): (Dyr.shape[1])]

        Zi = hstack((Zi, temp1))
        if len(Z) == 0:
            Z = Zi
        else:
            Z = vstack((Z, Zi))
    return Dd, DX, Z, mem, n, G, ng


def iterated_gmm_cluster(y, x, z, mem):
    y = np.float32(y)
    y = np.float32(y)
    x = np.float32(x)
    z = np.float32(z)
    mem = np.float32(mem)
    '''
    WINDMEIJER, F. (2000): “A Finite Sample Correction for the Variance of Linear Two-Step GMM Estimators
    (No. W00/19)” Working paper, Institute for Fiscal Studies.
    WINDMEIJER, F. (2005): “A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators,”
    Journal of Econometrics, 126 (1), 25–51.
    Inputs:
    y       nx1 vector of observations
    x       nxk matrix of regressors
    z       nxl matrix of instruments, l>=k (includes exogenous components of x)
    mem     nx1 vector of cluster membership set mem=(1:n)' for iid
    
    Outputs:
    b       kx1 vector of coefficient estimates
    s       kx1 vector of misspecification-and-cluster-and-heteroskedasticity
            robust asymptotic standard errors
    V       kxk misspecification-and-cluster-and-heteroskedasticity
            robust asymptotic covariance matrix estimate
    sw      kx1 vector of Windmeijer-corrected cluster-and-heteroskedasticity
            robust asymptotic standard errors
    Vw      kxk Windmeijer-corrected cluster-and-heteroskedasticity
            robust asymptotic covariance matrix estimate
    s0      kx1 vector of classic cluster-and-heteroskedasticity robust
            asymptotic standard errors
    V0      kxk classic cluster-and-heteroskedasticity robust
            asymptotic covariance matrix estimate
    J       J-statistic for overidentifying restrictions 
    pv      Asymptotic chi-square p-value for J-statistic
    iter    Number of iterations until convergence
    Output variables = NaN if the iteration reaches maxit
    '''
    tolerance = 1e-5
    maxit = 1e+3

    n = np.shape(y)[0]
    k = np.shape(x)[1]
    l = np.shape(z)[1]
    G = len(np.unique(mem))
    zx = tmat(z) @ x
    zy = tmat(z) @ y
    w = tmat(z) @ z
    b1 = inv(tmat(zx) @ inv(w) @ zx) @ (tmat(zx) @ inv(w) @ zy)
    idx = tmat(repmat(mem, 1, G)) == kron(ones((n, 1)), np.unique(mem).reshape(1, -1))
    iter = 0
    for iterx in range(int(maxit)):
        iter += 1
        e = y - x @ b1
        w = zeros((l, l))

        if n == G:
            ze = dot(z, tmat(repmat(e, 1, l)))
            w = (tmat(ze) @ ze) / n
        else:
            for g in range(G):
                zg = z[idx[:, g], :]
                eg = e[idx[:, g]]
                zeg = tmat(zg) @ eg
                w = w + zeg @ tmat(zeg)
            w = w / n

        b = inv(tmat(zx) @ inv(w) @ zx) @ (tmat(zx) @ inv(w) @ zy)
        db = b - b1
        if np.linalg.norm(db) < tolerance:
            break

        b1 = b

        if iter == (maxit - 1):
            b = np.nan
            s = np.nan
            V = np.nan
            sw = np.nan
            Vw = np.nan
            s0 = np.nan
            V0 = np.nan
            J = np.nan
            pv = np.nan
            return

    e = y - x @ b
    ze = z * (e @ ones((1, l)))
    mu = np.mean(ze, axis=0).reshape(-1, 1)

    if l > k:
        J = (tmat(mu) @ inv(w) @ mu)[0][0] * n
        pv = 1 - chi2.cdf(J, l - k)
    else:
        J = 0
        pv = 1
    xTz = tmat(x) @ z
    zTx = tmat(z) @ x
    eTz = tmat(e) @ z
    if n == G:
        ezwze = dot(e, (z @ inv(w) @ tmat(z) @ e))
        H = (1 / n ** 2) * xTz @ inv(w) * zTx - (2 / n ** 3) * xTz @ inv(w) * (tmat(z) @ x * repmat(ezwze, 1, k))
        temp1 = repmat(e, 1, l)
        temp2 = repmat(tmat((tmat(e) @ z) @ inv(w) * tmat(z)), 1, k)
        temp3 = repmat(e ** 0.5, 1, 1)
        Psi = dot(-(1 / n) * z @ temp1 @ inv(w) @ zTx - (1 / n) * temp2, x) + (1 / n ** 2) * temp2 * (
                z * temp3 / w @ zTx)
    else:
        Hpart = zeros((l, k))
        Psi = zeros((G, k))
        for g in range(G):
            zg = z[idx[:, g], :]
            eg = e[idx[:, g]]
            xg = x[idx[:, g], :]
            zgTeg = tmat(zg) @ eg
            zgTxg = tmat(zg) @ xg
            zTe = tmat(z) @ e

            Hpart = Hpart + zgTeg @ eTz @ inv(w) @ zgTxg + zgTxg * (eTz @ inv(w) @ zgTeg)[0][0]
            # Q=xTz, m(X,theta)=zgTeg=v(X,theta), Q(X,theta)=xgTzg, mu=zTe, v(theta,X)=egTzg

            Psi[g, :] = tmat(-(1 / n) * xTz @ inv(w) @ zgTeg
                             - (1 / n) * (tmat(xg) @ zg) @ inv(w) @ zTe
                             + (1 / n ** 2) * xTz @ inv(w) @ zgTeg @ (tmat(eg) @ zg) @ inv(w) @ zTe)

        H = (1 / n ** 2) * xTz @ inv(w) @ zTx - (1 / n ** 3) * xTz @ inv(w) @ Hpart

    Om = (tmat(Psi) @ Psi) / n

    V = inv(H) @ Om / tmat(H)
    s = sqrt(diag(V / n))

    Q = -zx / n
    V0 = inv(tmat(Q) @ inv(w) @ Q)
    s0 = sqrt(diag(V0 / n))

    Vw = inv(H) @ (tmat(Q) @ inv(w) @ Q) @ inv(tmat(H))
    sw = sqrt(diag(Vw / n))

    return b, s, V, sw, Vw, s0, V0, J, pv, iter


def conutnum(a, b):
    temp = zeros((len(b), 2))
    temp[:, 0] = b
    for i in range(len(b)):
        if i == 0:
            temp[i, 1] = np.sum((a <= b[0]))
        elif i == len(b) - 1:
            temp[i, 1] = np.sum((a >= b[i]))
        else:
            temp[i, 1] = np.sum((a > b[i - 1]) & (a <= b[i]))
    return temp


def lagx(a, b):
    ax = a.shape[1]
    temp = zeros((len(a), len(b)))
    temp[:] = np.nan
    for i in range(len(b)):
        t1 = -b[i]
        if ax > 1:
            temp[b[i]:, i] = a[:t1, i].reshape(-1)
        else:
            temp[b[i]:, i] = a[:t1].reshape(-1)
    return temp


# main
def Iterated_GMM(y, x, z, cluser):
    '''
    ACEMOGLU, D., S. JOHNSON, J. A. ROBINSON, AND P. YARED (2008): “Income and Democracy,” American
    Economic Review, 98 (3), 808–842. [1419,1421,1436,1437]

    CERVELLATI, M., F. JUNG, U. SUNDE, AND T. VISCHER (2014): “Income and Democracy,” Comment. American
    Economic Review, 104 (2), 707–719. [1421,1439]
    :return:
    '''
    cc, mem = np.unique(cluser, return_inverse=True)
    b, sr, Vr, sw, Vw, s0, V0, J, pv, iter = iterated_gmm_cluster(y, x, z, mem)
    k = x.shape[1]
    n = x.shape[0]
    RbI = np.zeros((k, 1))
    RbI[0, 0] = b[1] / (1 - b[0]) ** 2
    RbI[1, 0] = 1 / (1 - b[0])

    V_cie = tmat(RbI) @ Vr @ RbI
    s_cie = sqrt(diag(V_cie / n))
    V0_cie = tmat(RbI) @ V0 @ RbI
    s0_cie = sqrt(diag(V0_cie / n))
    tstat = b.reshape(-1) / sr
    pvals = 2 - 2 * t.cdf(np.abs(tstat), x.shape[1])

    result = np.zeros((5, len(b) + 1), dtype=object)
    result[0, 0] = 'para'
    result[0, 1:] = b.reshape(-1)
    result[1, 0] = 'SE'
    result[1, 1:] = sr
    result[2, 0] = 't-stat'
    result[2, 1:] = tstat
    result[3, 0] = 'pvalue'
    result[3, 1:] = pvals
    print(pd.DataFrame(result).transpose())
    print('J-stat', J, 'p-value', pv)

