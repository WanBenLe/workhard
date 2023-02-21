'''
    <SLOPE est code.>
    Copyright (C) <2021>  <Ben Wan>https://github.com/WanBenLe/DS-SLOPE-iterGMM
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

import time

import numpy as np
import pandas as pd
from numba import jit, typeof
from numpy import zeros, mod, abs, argsort, max, sign, argwhere, isnan, arange, sqrt, power, ceil, cumsum, min, max, \
    sort, array
from numpy.linalg import norm
import gc


@jit(fastmath=True, cache=True)
def cumsumx(arrayx):
    cum = array([0.0] * len(arrayx))
    cum[0] = arrayx[0]
    for i in range(len(arrayx) - 1):
        cum[i + 1] = cum[i] + arrayx[i + 1]
    return cum


@jit(parallel=True, fastmath=True, cache=True)
def evaluateProx(y, lambdax):
    n = len(y)
    x = zeros((n))
    s = zeros((n))
    w = zeros((n))
    idx_i = zeros((n))
    idx_j = zeros((n))

    k = 0
    for i in range(n):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = y[i] - lambdax[i]
        w[k] = s[k]
        while (k > 0) & (w[k - 1] <= w[k]):
            k -= 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)
        k += 1

    for j in range(k):
        d = 0
        d = w[j]
        if d < 0:
            d = 0
        a1 = 0
        a1 = int(idx_j[j] + 1 - idx_i[j])
        a2 = 0
        a2 = int(idx_i[j])
        for i in range(a1):
            x[i + a2] = d
    return x


@jit(fastmath=True, cache=True)
def tmat(mat):
    return np.transpose(mat)


@jit(parallel=True, fastmath=True, cache=True)
def proxSortedL1(y, lambdax):
    k = array([0, 1])
    a = y.copy()
    sgn = sign(y)
    idx = argsort(-abs(a.reshape(-1)))
    y = -sort(-abs(a.reshape(-1))).reshape(-1, 1)
    try:
        k = argwhere(y > lambdax)[-1]
    except:
        print(1)
    n = y.shape[0] * y.shape[1]
    x = zeros((n, 1))

    if len(k) > 0:
        k = k[0]
        v1 = y[0: k]
        v2 = lambdax[0:k]
        v = evaluateProx(v1.reshape(-1), v2.reshape(-1)).reshape(-1, 1)
        x[idx[0: k].reshape(-1)] = v
        x = sgn * x
    return x


@jit(fastmath=True)
def SBc_FastProxSL(A, b, lambdax):
    # A:自变量
    # b:因变量
    # lambdax:正则项,要求递减且大于0
    # 优化更新迭代

    gradIter = 20
    optimIter = 10

    n = A.shape[1]
    x = np.random.randn(n, 1)
    x /= norm(x, 2)
    x = tmat(A) @ (A @ x)
    L = norm(x, 2)

    xInit = zeros((n, 1))
    t = 1
    eta = 2
    x = xInit
    y = x
    Ax = A @ x
    fPrev = 10 ** 99
    iter = 0
    tPrev = 0
    Aprods = 2
    ATprods = 1
    AxPrev = np.array([[0.0], [0.0]])
    # 迭代10次
    while 1:
        if mod(iter, gradIter) == 0:
            r = A @ y - b
            g = tmat(A) @ r
            f = (tmat(r) @ r) / 2
        else:
            r = (Ax + ((tPrev - 1) / t) * (Ax - AxPrev)) - b
            g = tmat(A) @ r
            f = (tmat(r) @ r) / 2
        iter += 1
        if mod(iter, optimIter) == 0:
            gs = g[argsort(-abs(g).reshape(-1))]
            ys = y[argsort(-abs(y).reshape(-1))]

            try:
                infeas = max(cumsumx((gs - lambdax).reshape(-1)))
            except:
                print(1)
            objPrimal = f + tmat(lambdax) @ ys

            objDual = -f - tmat(r) @ b
        if iter > 3000:
            break
        AxPrev = Ax
        xPrev = x
        fPrev = f
        tPrev = t
        while 1:
            x = proxSortedL1(y - (1 / L) * g, lambdax / L)
            d = x - y
            Ax = A @ x
            r = A @ x - b
            f = (tmat(r) @ r) / 2
            q = fPrev + tmat(d) @ g + (L / 2) * (tmat(d) @ d)
            Aprods = Aprods + 1
            if q >= f * (1 - 10 ** -12):
                break
            else:
                L = L * eta
        t = (1 + sqrt(1 + 4 * power(t, 2))) / 2
        y = x + ((tPrev - 1) / t) * (x - xPrev)
    x = y

    info = zeros((7, 1))

    info[1] = Aprods + ceil(iter / gradIter)
    info[2] = ATprods + iter
    info[3] = objPrimal
    info[4] = objDual
    info[5] = infeas
    info[6] = L
    return x, info
