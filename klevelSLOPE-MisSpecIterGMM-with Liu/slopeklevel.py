'''
    <Klevel est code.>
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

import gc
import time
import numpy as np
import pandas as pd
from numpy import zeros, mod, abs, argsort, max, sign, argwhere, isnan, arange, sqrt, power, ceil, cumsum, min, max, \
    sort, array
from numpy.linalg import norm
from slopeest import SBc_FastProxSL


def DSSlope(A, b):
    doge1, info = slope3level(A, b)
    # 第一个非0集
    Set1 = doge1 != 0
    Set2x = np.zeros((len(Set1), A.shape[1]))
    index1 = np.argwhere(Set1 == True)[:, 0]
    for i in index1:
        print('a1', i)
        y_ds = A[:, i].reshape(-1, 1)
        selct = np.array(range(A.shape[1])) != i
        x_ds = A[:, selct]
        para_ds, y_ds = slope2level(x_ds, y_ds)
        Set2x[i, selct] = (para_ds != 0).reshape(-1)
    result_st = np.sum(Set2x, axis=0)
    result_sr = np.sum(Set2x, axis=0) / len(index1)
    print('DS select time:', result_st)
    print('DS select ratio:', result_sr)
    Set_2 = np.max(Set2x, axis=0)
    Set_all = np.max(np.concatenate((Set1.reshape(1, -1), Set_2.reshape(1, -1))), axis=0)
    return Set1, Set_2, Set_all, result_st, result_sr


def slope3level(A, b):
    p = A.shape[1]
    lam1 = 18
    tx = []
    msex = []
    timesx = 0
    for p12 in np.array([0.4, 0.4, 0.1]):
        gc.collect()
        print(p12)
        for p23 in np.array([0.4, 0.5, 0.6]):
            for s1 in np.array([1e-2, 1e-1, 1, 1e1]):
                for s2 in np.array([5e-3, 5e-2, 5e-1]):
                    for s3 in np.array([2.5e-3, 2.5e-3, 2.5e-1]):
                        for passx in [10000]:
                            timesx += 1
                            Lda = [lam1 * s1] * int(np.rint(p * p12))
                            Lda.extend([lam1 * s2] * (int(np.rint(p * p23)) - int(np.rint(p * p12))))
                            Lda.extend([lam1 * s3] * (p - int(np.rint(p * p23))))
                            para = np.array(Lda).reshape(-1, 1)
                            para = -np.sort(-para.reshape(-1)).reshape(-1, 1)
                            time_op = time.time()
                            doge1, doge2 = SBc_FastProxSL(A, b, para)
                            time_end = time.time()
                            doge2 = doge2.reshape(-1, 1)
                            doge2[0] = time_end - time_op
                            tx.append(doge2[0])
                            msex.append([p12, p23, s1, s2, s3, (np.sum((b - A @ doge1) ** 2) / len(A))])
    print('mean time: ', np.mean(tx))
    best = msex[np.argmin(np.array(msex)[:, -1])]
    print('best para', best)
    p12, p23, s1, s2, s3 = best[0], best[1], best[2], best[3], best[4]
    Lda = [lam1 * s1] * int(np.rint(p * p12))
    Lda.extend([lam1 * s2] * (int(np.rint(p * p23)) - int(np.rint(p * p12))))
    Lda.extend([lam1 * s3] * (p - int(np.rint(p * p23))))
    para = np.array(Lda).reshape(-1, 1)
    para = -np.sort(-para.reshape(-1)).reshape(-1, 1)
    doge1, doge2 = SBc_FastProxSL(A, b, para)
    doge2 = doge2.reshape(-1, 1)
    doge2[0] = time_end - time_op
    info = np.array([['runtime'], ['Aprods'], ['ATprods'], ['objPrimal'], ['objDual'], ['infeas'], ['L']])
    info = np.hstack((info, doge2))
    print(info)
    print(doge1)
    return doge1, info


def slope2level(A, b):
    p = A.shape[1]
    lam1 = 20
    tx = []
    msex = []
    timesx = 0
    for p12 in np.array([0.4, 0.3, 0.1]):
        gc.collect()
        for s1 in np.array([1e-2, 1e-1, 1, 1e1]):
            for s2 in np.array([5e-2, 5e-1, 5]):
                for passx in [10000]:
                    timesx += 1
                    Lda = [lam1 * s1] * int(np.rint(p * p12))
                    Lda.extend([lam1 * s2] * (p - int(np.rint(p * p12))))
                    para = np.array(Lda).reshape(-1, 1)
                    para = -np.sort(-para.reshape(-1)).reshape(-1, 1)
                    time_op = time.time()
                    doge1, doge2 = SBc_FastProxSL(A, b, para)
                    time_end = time.time()
                    doge2 = doge2.reshape(-1, 1)
                    doge2[0] = time_end - time_op
                    tx.append(doge2[0])
                    msex.append([p12, s1, s2, (np.sum((b - A @ doge1) ** 2) / len(A))])
    print('mean time: ', np.mean(tx))
    best = msex[np.argmin(np.array(msex)[:, -1])]
    print('best para', best)
    p12, s1, s2 = best[0], best[1], best[2]
    Lda = [lam1 * s1] * int(np.rint(p * p12))
    Lda.extend([lam1 * s2] * (p - int(np.rint(p * p12))))
    para = np.array(Lda).reshape(-1, 1)
    para = -np.sort(-para.reshape(-1)).reshape(-1, 1)
    doge1, doge2 = SBc_FastProxSL(A, b, para)
    doge2 = doge2.reshape(-1, 1)
    doge2[0] = time_end - time_op
    info = np.array([['runtime'], ['Aprods'], ['ATprods'], ['objPrimal'], ['objDual'], ['infeas'], ['L']])
    info = np.hstack((info, doge2))
    print(info)
    return doge1, info
