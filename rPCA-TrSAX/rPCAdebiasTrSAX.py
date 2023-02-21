# coding=utf-8
import numba as nb
import numpy as np
from numba import jit
from numpy import diag
from numpy.linalg import inv, qr, pinv
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.metrics import mutual_info_score, f1_score, recall_score, accuracy_score
from sklearn.preprocessing import scale
import gc
import scipy.io as sio
import os
import pandas as pd
import time
from sklearn.decomposition import PCA
from scipy import stats

since = time.time()

os.environ[
    "NUMBA_CPU_FEATURES"
] = "+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+avx512vnni,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+pku,+popcnt,+prfchw,+rdrnd,+rdseed,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave,+xsavec,+xsaveopt,+xsaves"


@jit()
def fun1(tseries, omega, jie_x, duan_k, duan, duan_m, duanshu):
    x1 = np.hstack(((np.arange(0, duanshu, 1.0) + 1.0).reshape(-1, 1), np.ones((duanshu, 1))))
    # x1 = (np.arange(0, duanshu, 1.0) + 1.0).reshape(-1, 1)

    dot1 = pinv(x1.T @ x1 + 0 * np.eye(2)) @ x1.T
    for i in range(len(tseries) - omega):
        jie_x[i] = jie_x_i = tseries[i: i + omega]
        for j in range(int(duan)):
            '''
            std_temp=np.nanstd(jie_x_i[j:j+1])
            if std_temp==0:
                duan_m[i,j]=0
            else:
                duan_m[i, j] = np.mean(jie_x_i[j:j+1])#/std_temp
            '''
            duan_m[i, j] = np.mean(jie_x_i[j:j + 1])
            y1 = jie_x_i[duanshu * j: (j + 1) * duanshu].reshape(-1, 1)
            duan_k[i, j] = (dot1 @ y1)[0][0]
            """
            if model == 1:
                weight1 = inv(diag(np.abs((y1 - (duan_k[i, j] * x1))).reshape(-1) ** 2))
                xx = x1.transpose() @ weight1
                duan_k[i, j] = (inv(xx @ x1) @ xx @ y1)[0][0]
            """

    for i in range(duan_m.shape[1]):
        duan_m[:, i] = (duan_m[:, i] - np.mean(duan_m[:, i])) / np.std(duan_m[:, i])

    return duan_k, duan_m


@jit()
def DuanCreat(tseries):
    omega = 3000
    jie_x = np.zeros((len(tseries) - omega, omega))
    duanshu = 100
    duan = omega // duanshu
    duan_m = np.zeros((len(tseries) - omega, duan))
    duan_k = np.zeros((len(tseries) - omega, duan))
    duan_k, duan_m = fun1(tseries, omega, jie_x, duan_k, duan, duan_m, duanshu)
    return duan_k, duan_m


# @jit()
def bacboot(samples, range_num):
    times = 10000
    np.random.seed(42)
    new_result = np.zeros((times, 10))
    # old_mean = np.mean(samples)
    old_20 = np.percentile(samples, 20)
    old_40 = np.percentile(samples, 40)
    old_60 = np.percentile(samples, 60)
    old_80 = np.percentile(samples, 80)
    r20 = np.zeros(10000)
    r40 = np.zeros(10000)
    r60 = np.zeros(10000)
    r80 = np.zeros(10000)
    for i in range(times):
        ind = np.random.choice(a=len(samples), size=len(samples))
        temp = samples[ind]
        if range_num == 4:
            r20[i] = np.percentile(temp, 20)
            r40[i] = np.percentile(temp, 40)
            r60[i] = np.percentile(temp, 60)
            r80[i] = np.percentile(temp, 80)
    if range_num == 4:
        r20.sort()
        r40.sort()
        r60.sort()
        r80.sort()
        percentile_x = stats.percentileofscore(r20, old_20, kind='rank') / 100
        p20 = np.percentile(samples, stats.norm.cdf(stats.norm.ppf(percentile_x) * 2) * 100)
        percentile_x = stats.percentileofscore(r40, old_40, kind='rank') / 100
        p40 = np.percentile(samples, stats.norm.cdf(stats.norm.ppf(percentile_x) * 2) * 100)
        percentile_x = stats.percentileofscore(r60, old_60, kind='rank') / 100
        p60 = np.percentile(samples, stats.norm.cdf(stats.norm.ppf(percentile_x) * 2) * 100)
        percentile_x = stats.percentileofscore(r80, old_80, kind='rank') / 100
        p80 = np.percentile(samples, stats.norm.cdf(stats.norm.ppf(percentile_x) * 2) * 100)
        result = np.array([p20, p40, p60, p80])
    else:
        result = np.array([0, 0])
    return result


@jit()
def rpca(data, S):
    for i in range(data.shape[1]):
        data[:, i] = data[:, i] - np.mean(data[:, i])
    n = data.shape[0]
    p = data.shape[1]
    U, Sigma, Vh = np.linalg.svd(data, full_matrices=True, compute_uv=True)
    V = Vh.T
    '''
    C = np.dot(data.T, data) / (data.shape[0] - 1)
    eig_vals, eig_vecs = np.linalg.eig(C)
    '''
    X_pca_svd = np.dot(data, Vh.T)
    U, Sigma, Vh = np.linalg.svd(data, full_matrices=True, compute_uv=True)
    sigma2 = np.sum(Sigma[-S:] ** 2) / (n * p - p - n * S - p * S + S ^ 2)
    lambda_shrinked = (Sigma[:S] ** 2 - n * (p / np.min([p, (n - 1)])) * sigma2) / Sigma[:S]
    if S == 1:
        result = np.dot(U[:, 0].reshape(-1, 1) * lambda_shrinked, Vh[:, 0].reshape(1, -1))
    else:
        result = U[:, :S] @ np.diag(lambda_shrinked) @ V[:, :S].T
    print(X_pca_svd)
    print(result)
    return result


@jit()
def TrSAXCreat(tseries, ols_range, sax_range):
    duan_k, duan_m = DuanCreat(tseries)
    zimu1 = np.zeros_like(duan_m)
    zimu2 = np.zeros_like(duan_m)
    '''
    for i in range(duan_m.shape[1]):
        duan_m[:, i] = (duan_m[:, i] - np.mean(duan_m[:, i])) / np.std(duan_m[:, i])

    for i in range(duan_k.shape[1]):
        duan_k[:, i] = (duan_k[:, i] - np.mean(duan_k[:, i])) / np.std(duan_k[:, i])
    '''
    for i in range(duan_k.shape[0]):
        for j in range(duan_k.shape[1]):
            temp = duan_k[i, j]
            # ols_range=[-1,-0,0,1]

            if temp > ols_range[3]:
                # if temp>-np.inf:
                zimu2[i, j] = 50.0
            elif (temp <= ols_range[3]) & (temp > ols_range[2]):
                zimu2[i, j] = 40.0
            elif (temp <= ols_range[2]) & (temp > ols_range[1]):
                zimu2[i, j] = 30.0
            elif (temp <= ols_range[1]) & (temp > ols_range[0]):
                zimu2[i, j] = 20.0
            else:
                zimu2[i, j] = 10.0

    for i in range(duan_m.shape[0]):
        for j in range(duan_m.shape[1]):
            temp = duan_m[i, j]
            sax_range = [-0.43, 0.43]
            # if temp>-np.inf:
            if temp > sax_range[1]:
                zimu1[i, j] = 3.0
            elif (temp <= sax_range[1]) & (temp > sax_range[0]):
                zimu1[i, j] = 2.0
            else:
                zimu1[i, j] = 1.0

    TrSAX = zimu2[:] + zimu1[:]
    TrSAX = TrSAX[:, 0] * 10000 + TrSAX[:, 1] * 100 + TrSAX[:, 2]

    temp = np.zeros_like(TrSAX)
    temp[0] = TrSAX[0]
    indx = 0
    for i in range(len(TrSAX) - 1):
        if temp[indx] != TrSAX[i + 1]:
            temp[indx + 1] = TrSAX[i + 1]
            indx += 1
    temp = temp[: indx + 1]
    return temp


@jit()
def TrSAXmodel(tseries1, unique_2, count_2, ols_range, sax_range):
    TrSAX1 = TrSAXCreat(tseries1, ols_range, sax_range)
    unique_1 = np.unique(TrSAX1)
    count_1 = np.zeros(len(unique_1))
    for indx, item in enumerate(unique_1):
        count_1[indx] = len(unique_1[unique_1 == item])
    union_all = np.unique(np.concatenate((unique_1, unique_2)))
    result = np.zeros((len(union_all), 3))
    result[:, 0] = union_all
    for indx, name in enumerate(union_all):
        indx1 = np.where(unique_1 == name)[0]
        if len(indx1) > 0:
            result[indx, 1] = count_1[indx1[0]]
        indx2 = np.where(unique_2 == name)[0]
        if len(indx2) > 0:
            result[indx, 2] = count_2[indx2[0]]
    Dis = np.sum((result[:, 1] - result[:, 2]) ** 2) ** 0.5
    # Dis=Dis+np.sum(np.abs(result[:,1]-result[:,2]))*100000
    # Dis = Dis + 0.01*(np.sum(result[:, 1]) - np.sum(result[:, 2]))**2
    # Dis=mutual_info_score(result[:,1],result[:,2])
    return Dis


load_plane = os.path.abspath("data") + "//"
# load_plane = '/home/datalab/ben_zeppelin/notebooks/MdisTrSAX/data/'

data_set = np.zeros((600, 2), dtype=object)
for i in range(600):
    mat_name = "TRAIN" + str(101 + i) + ".mat"
    data_set[i, 0] = sio.loadmat(load_plane + mat_name)["data"]
    data_set[i, 1] = "TRAIN" + str(101 + i)

df1 = pd.DataFrame(data_set, columns=["data", "dataset"])
# csv_path = "/home/datalab/ben_zeppelin/notebooks/MdisTrSAX/label.csv"
# df2 = pd.read_csv(csv_path)
df2 = pd.read_csv("label.csv")

data = df1.merge(df2, how="inner").values
datap = data[data[:, -1] == 1]
datan = data[data[:, -1] == 0]

del data, data_set, df1, df2

samples = 600
np.random.seed(42)
datap = datap[np.random.choice(len(datap), 300, replace=False)]
datap_sel = datap[0:1, 0][0]
datap_else = datap[1:]
datan = datan[np.random.choice(len(datan), 300, replace=False)]
datan_sel = datan[0:1, 0][0]
datan_else = datan[1:]
datax = np.vstack((datap_else, datan_else))[:, [0, 2]]
del datap_else, datan_else
gc.collect()

print(datax.shape)
result = np.zeros((len(datax), 3))
pca12 = PCA(n_components=12)
pca_method = 1
if pca_method == 1:
    result_temp = pca12.fit(datan_sel.T)
    n_keep = np.argwhere(np.cumsum(result_temp.explained_variance_ratio_) > 0.9)[0][0] + 1
    aaa_n = np.mean(rpca(datan_sel.T, S=n_keep), axis=1)
    '''
    pca = PCA(n_components=n_keep)
    aaa_n = np.mean(pca.fit_transform(datan_sel.T), axis=1)
    plt.plot(aaa_n)
    plt.show()
    plt.plot(aaa_n)
    plt.show()
    '''
    duan_k_n, duan_m_n = DuanCreat(aaa_n.reshape(-1))

    result_temp = pca12.fit(datap_sel.T)
    n_keep = np.argwhere(np.cumsum(result_temp.explained_variance_ratio_) > 0.9)[0][0] + 1
    aaa_p = np.mean(rpca(datap_sel.T, S=n_keep), axis=1)
    '''
    pca = PCA(n_components=n_keep)
    aaa_p = np.mean(pca.fit_transform(datap_sel.T), axis=1)
    '''
    duan_k_p, duan_m_p = DuanCreat(aaa_p.reshape(-1))

    ols_range = boost(np.concatenate((duan_k_n.reshape(-1), duan_k_p.reshape(-1))), range_num=4)
    # ols_range =boost(duan_k_p.reshape(-1), range_num=4)
    sax_range = boost(np.concatenate((duan_m_n.reshape(-1), duan_m_p.reshape(-1))), range_num=2)

    TrSAX2n = TrSAXCreat(aaa_n.reshape(-1), ols_range, sax_range)
    TrSAX2p = TrSAXCreat(aaa_p.reshape(-1), ols_range, sax_range)
    print(TrSAX2n.shape)
else:
    duan_k_n, duan_m_n = DuanCreat(np.mean(datan_sel, axis=0))
    duan_k_p, duan_m_p = DuanCreat(np.mean(datap_sel, axis=0))
    ols_range = boost(np.concatenate((duan_k_n.reshape(-1), duan_k_p.reshape(-1))), range_num=4)

    # ols_range = boost(duan_k_p.reshape(-1), range_num=4)
    sax_range = boost(np.concatenate((duan_m_n.reshape(-1), duan_m_p.reshape(-1))), range_num=2)
    TrSAX2n = TrSAXCreat(np.mean(datan_sel, axis=0), ols_range, sax_range)
    TrSAX2p = TrSAXCreat(np.mean(datap_sel, axis=0), ols_range, sax_range)
    print(ols_range)
    print(sax_range)
    print(1)
unique_2n = np.unique(TrSAX2n)
count_2n = np.zeros(len(unique_2n))
for indx, item in enumerate(unique_2n):
    count_2n[indx] = len(unique_2n[unique_2n == item])
unique_2p = np.unique(TrSAX2p)
count_2p = np.zeros(len(unique_2p))
for indx, item in enumerate(unique_2p):
    count_2p[indx] = len(unique_2p[unique_2p == item])
for i in range(len(datax)):
    # print(np.round(i / len(datax) * 100, 2))
    temp_data = datax[i, 0]
    label = datax[i, 1]
    dis = [0, 0, 0, 0]
    if pca_method == 1:
        result_temp = pca12.fit(temp_data.T)
        n_keep = np.argwhere(np.cumsum(result_temp.explained_variance_ratio_) > 0.9)[0][0] + 1
        aaa = np.mean(rpca(temp_data.T, S=n_keep), axis=1)
        # pca = PCA(n_components=n_keep)
        # aaa = np.mean(pca.fit_transform(temp_data.T), axis=1)
        dis[0] = TrSAXmodel(aaa.reshape(-1), unique_2n, count_2n, ols_range, sax_range)
        dis[1] = TrSAXmodel(aaa.reshape(-1), unique_2p, count_2p, ols_range, sax_range)
        # print(TrSAX2n.shape)
    else:
        dis[0] = TrSAXmodel(np.mean(temp_data, axis=0), unique_2n, count_2n, ols_range, sax_range)
        dis[1] = TrSAXmodel(np.mean(temp_data, axis=0), unique_2p, count_2p, ols_range, sax_range)

    if dis[0] > dis[1]:
        result[i, 0] = 1
    else:
        result[i, 0] = 0
    if dis[2] > dis[3]:
        result[i, 1] = 1
    else:
        result[i, 1] = 0
    result[i, 2] = label
df3 = pd.DataFrame(result, columns=["TrSAX", "new_TrSAX", "True"])
df3.to_csv("data.csv", index=False, encoding="utf_8_sig")
f1_1 = np.round(f1_score(df3['True'], df3['TrSAX']), 4)
recall_1 = np.round(recall_score(df3['True'], df3['TrSAX']), 4)
acc_1 = np.round(accuracy_score(df3['True'], df3['TrSAX']), 4)
print('acc', acc_1)
print('recall', recall_1)
print('f1', f1_1)
# 程序执行部分
time_elapsed = time.time() - since
print(
    "Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
)  # 打印出来时间
