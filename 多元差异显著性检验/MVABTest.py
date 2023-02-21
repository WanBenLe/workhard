'''
Copyright <2021> <Ben Wan: wanbenfighting@gmail.com>
'''

import numpy as np
from numpy.linalg import inv
from scipy.stats import f, ranksums


def Hotelling_HT_test(X, Y):
    # 多元AB霍特林测试,对dis有Norm假设,带dof adj 的方差不齐Welch's t test的拓展
    p = X.shape[1]
    m = len(X)
    n = len(Y)
    N = m + n
    X_bar = np.mean(X, axis=0).reshape(1, -1)
    Y_bar = np.mean(Y, axis=0).reshape(1, -1)
    V1 = (X - X_bar).T @ (X - X_bar) / (m - 1)
    V2 = (Y - Y_bar).T @ (Y - Y_bar) / (m - 1)
    D = (X_bar - Y_bar).reshape(-1, 1)

    V = (1 / n + 1 / m) * ((m - 1) * V1 + (n - 1) * V2) / (N - 2)
    HT = D.transpose() @ inv(V) @ D
    HT_stat = (N - p - 1) / ((N - 2) * p) * HT
    HT_stat = HT_stat[0][0]
    dof1 = p
    dof2 = N - p - 1
    single_pvalue = f.sf(HT_stat, dof1, dof2)
    dobule_pvalue = f.sf(np.abs(HT_stat), dof1, dof2)

    return HT_stat, single_pvalue, dobule_pvalue


def Tippett_MuNonpara_test(X, Y, test_sample=5):
    '''
    Multivariate tests based on interpoint distances with application to magnetic resonance imaging, Marco Marozzi,
    Statistical Methods in Medical Research, 2014
    '''
    # 高维的多元AB测试,抽样的test_sample的for循环次数是阶乘级别的,切记不能太大
    X = X[np.random.choice(len(X), test_sample, replace=False)]
    Y = Y[np.random.choice(len(Y), test_sample, replace=False)]
    m = len(X)
    n = len(Y)
    N = m + n

    # 不用对数容易爆炸
    B = np.sum(np.log(np.arange(N) + 1)) - np.sum(np.log(np.arange(m) + 1)) - np.sum(np.log(np.arange(n) + 1))
    B = int(np.exp(B))
    PJK = np.zeros((B, m))
    for j in range(m):
        indx = np.random.choice(len(X), 1, replace=False)[0]
        x_first = X[indx]
        z_other = np.concatenate((X[:indx, :], X[(1 + indx):, :], Y), axis=0)
        z_len = len(z_other)
        L2 = (np.sum((x_first - z_other) ** 2, axis=1)) ** 0.5
        Sample1 = L2[:m]
        Sample2 = L2[m:]
        PJK0 = ranksums(Sample1, Sample2).pvalue
        for i in range(B):
            z = z_other[np.random.choice(z_len, z_len, replace=False)]
            L2_t = (np.sum((x_first - z) ** 2, axis=1)) ** 0.5
            PJK[i, j] = ranksums(L2_t[:m], L2_t[m:]).pvalue
    # Tippett adj
    X1 = np.max(1 - PJK[1:], axis=1)
    X2 = np.max(1 - PJK[0])
    QMJK = np.sum(X1 < X2) / B
    return QMJK


'''
# 样本量
test_n1 = 100
# 维度
test_n2 = 50
X = np.random.rand(test_n1, test_n2)
Y = np.random.rand(test_n1, test_n2) + 1000
print(np.mean(X, axis=0))
print(np.mean(Y, axis=0))

X = X - Y
print(X)
Y = np.zeros((5, 2))

# 样本量够大的时候就Hotelling,裂开来了才用下面的
print(Hotelling_HT_test(X, Y))
print(Tippett_MuNonpara_test(X, Y, test_sample=6))
print(1)
'''
