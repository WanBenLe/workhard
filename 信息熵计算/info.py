
import numpy as np
from pickle import load, dump
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from numba import jit
from sklearn.metrics import log_loss
from numpy import sum, zeros, unique, abs, dstack, argsort, meshgrid, e, pi, ones, concatenate, min, log, log2, sort, \
    arange, ones_like, hstack, ndarray
import os
from polars import DataFrame
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import BallTree
from scipy.special import psi, gamma
from copy import deepcopy




def CUnique(A):
    a = DataFrame(A.reshape(-1, 1), columns=['a'])
    b = a.groupby(['a']).count()
    c = b.to_numpy()
    return c[:, 0], c[:, 1]


def CUnique2(x, y):
    a = DataFrame(concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1), columns=['a', 'b'])
    b = a.groupby(['a', 'b']).count()
    c = b.to_numpy()
    return c


@jit(forceobj=True)
def SU_c(X, Y, num1=0, num2=0):
    if num2 == 1:
        k = 3
        tree = BallTree(Y.reshape(-1, 1), metric='chebyshev')
        dis = tree.query(Y.reshape(-1, 1), k=k + 1)[0][:, -1]
        cD = log((pi *1/ 2) / gamma(1+1/2))
        Hy=psi(len(Y))-psi(k)+log(cD)+1*log(dis).mean()
    else:
        unY, cunY = CUnique(Y)
        pY = cunY / sum(cunY)
        Hy = -sum(pY * log2(pY))
    if num1 == 1:
        k = 3
        tree = BallTree(X.reshape(-1, 1), metric='chebyshev')
        dis = tree.query(X.reshape(-1, 1), k=k + 1)[0][:, -1]
        cD = log((pi *1/ 2) / gamma(1+1/2))
        Hx=psi(len(X))-psi(k)+log(cD)+1*log(dis).mean()
    else:
        unX, cunX = CUnique(X)
        pX = cunX / sum(cunX)
        Hx = -sum(pX * log2(pX))

    if (num1 == 1) and (num2 == 0):
        H_xy = mutual_info_classif(X.reshape(-1, 1), Y)
    elif (num1 == 0) and (num2 == 1):
        H_xy = mutual_info_classif(y.reshape(-1, 1), X)
    elif (num1 == 1) and (num2 == 1):
        H_xy = mutual_info_regression(y.reshape(-1, 1), X.reshape(-1, 1))
    else:

        unY, cunY = CUnique(Y)
        pY = cunY / sum(cunY)
        Hy = -sum(pY * log2(pY))
        XY_count = CUnique2(Y, X)
        H_xy = 0
        for ind_x, set_y in enumerate(unY):
            temp = XY_count[XY_count[:, 0] == set_y]
            pX_temp = temp[:, 2] / sum(temp[:, 2])
            H_xy -= pY[ind_x] * sum(pX_temp * log2(pX_temp))
    return 2 * (Hx - H_xy) / (Hx + Hy)



SU_c(x[:, i], y, num1=0, num2=0)