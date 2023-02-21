'''Copyright <2021> <Ben Wan: wanbenfighting@gmail.com>'''
import numpy as np
from scipy.stats import norm
from copy import deepcopy

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except:
    pass
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import  LinearSVR


def DMLM(data, K=10):
    '''
    处理T,协变量X,中介变量M,还有结果Y
    T对M和Y产生影响,M也对Y产生影响
    X跟是否进入T和X有关
    业务例子:
    销售数据/产品属性X跟涨价不涨价T有关,可能也跟刷价失败成功与否M有关
    涨价不涨价T和刷价失败成功与否M都会对总利润Y产生影响

    2022年10月的这篇使用2018的DML和2014的有效得分函数对上面的中介效应进行了估计,得到了去偏的反事实的结果.假设检验和置信区间估计
    就是Y(T,M(1-T)),业务例子就是去偏的刷价,但是刷价失败的潜在Y值
    Causal Mediation Analysis with Double Machine Learning, Helmut Farbmacher, Econometrics Journal, 2022

    如果对假设区间置信区间不太感兴趣对造核弹感兴趣可以看看这篇曝光与否概率估计uplift的也是类似的思路
    Addressing Exposure Bias in Uplift Modeling for Large-scale Online Advertising, Wenwei Ke, ICDM 2021
    '''
    n = len(data)
    data = data[np.random.choice(np.arange(n), n)]
    nk = n // K
    data = data[:nk * K]
    ES_all = []
    for ia in range(K):
        subsample = data[(ia * nk):(ia + 1) * nk]
        # 补集估计模型
        subsamplec_ind = np.setdiff1d(np.arange(n), np.arange((ia * nk), (ia + 1) * nk))
        subsamplec = data[subsamplec_ind]
        # p_d(X)
        pdX = LogisticRegression().fit(subsamplec[:, 3:], subsamplec[:, 1])
        # p_d(M,X)
        pdMX = LogisticRegression().fit(subsamplec[:, 2:], subsamplec[:, 1])
        n_ssc = len(subsamplec)
        subsamplec = subsamplec[np.random.choice(np.arange(n_ssc), n)]
        ssc1 = subsamplec[:n_ssc // 2]
        ssc2 = subsamplec[n_ssc // 2:]
        # \mu(d,M,X)
        uMDX = LinearSVR().fit(ssc1[:, 1:], ssc1[:, 0])
        uMDX_Y = uMDX.predict(ssc2[:, 1:])

        temp = deepcopy(ssc2[:, 1:])
        temp[:, 0] = 1 - temp[:, 0]

        # \hat{\omega}(D,X)^{k}
        omega_antix = LinearSVR().fit(temp, uMDX_Y)

        # 估计完就该上有效得分了
        F_pdMX = pdMX.predict_proba(subsample[:, 2:])[:, 1]
        F_pdX = pdX.predict_proba(subsample[:, 3:])[:, 1]
        F_uMDX = uMDX.predict(subsample[:, 1:])
        temp = deepcopy(subsample[:, 1:])
        temp[:, 0] = 1 - temp[:, 0]
        F_omega_antix = omega_antix.predict(temp)
        p1 = (1 - F_pdMX) / (F_pdMX * (1 - F_pdX)) * (subsample[:, 0] - F_uMDX)
        p2 = 1 / (1 - F_pdX) * (F_uMDX - F_omega_antix)
        p3 = F_omega_antix
        ES = p1 + p2 + p3
        ES_all.extend(ES)

    ES_all = np.array(ES_all)
    fit_effect = np.mean(ES_all)
    var_est = np.mean(ES_all ** 2)
    stat = fit_effect * n ** 0.5 / var_est
    p = 2 * norm.sf(np.abs(stat))
    ci95 = 1.65 * var_est ** 0.5

    print('DML中介效应统计检验量和p值', stat, p)
    print('Y(T,M(1-T))的效应估计结果')
    print('DML中介效应95%CI', fit_effect - ci95, fit_effect, fit_effect + ci95)
    return fit_effect, p


data_len = 100000
K = 10

np.random.seed(42)
Y = np.random.rand(data_len, 1) * 10000
D = np.random.randint(0, 2, (data_len, 1))
M = np.random.randint(0, 2, (data_len, 1))
X = np.random.randn(data_len, 10) * 10000
print('处理变量和中介变量相关系数\n', np.corrcoef(np.concatenate((D.reshape(-1, 1), M.reshape(-1, 1)), axis=1).T))
# [Y,D,M,X]
data = np.concatenate((Y, D, M, X), axis=1)
data[:, 1] = data[:, 1].astype(int)
data[:, 2] = data[:, 2].astype(int)

DMLM(data, K)
print(1)
print(1)
