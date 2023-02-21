import numpy as np
import statsmodels.api as sm
from scipy import stats



def M_bcaboost(x1, z1, y1, statx):
    # 普通百分位boot +BC_a boot
    times = 1000
    alpha_beta_list = []
    ransize = x1.shape[0]
    for i in range(times):
        ind = np.random.choice(a=x1.shape[0], size=ransize)
        x = x1[ind]
        z = z1[ind]
        y = y1[ind]

        X2 = sm.add_constant(x)  # 加入常数项
        result2 = sm.OLS(z, X2).fit()  # 回归
        temp_x = np.hstack((x.reshape(-1, 1), z.reshape(-1, 1)))
        X3 = sm.add_constant(temp_x)  # 加入常数项
        result3 = sm.OLS(y, X3).fit()  # 回归
        alpha = result2.params[1]
        beta = result3.params[2]
        alpha_beta_list.append(alpha * beta)
    ab_array = np.array([alpha_beta_list])
    # 普通百分位boot
    # print(np.percentile(ab_array,2.5))
    # print(np.percentile(ab_array, 97.5))
    # 后面是BC_a Boot
    # 计算原估计在boot序列的百分位
    percentile_x = stats.percentileofscore(ab_array.T, statx, kind='rank') / 100
    # 获取正态分布中对应的z
    ppf_x = stats.norm.ppf(percentile_x)
    up = 2 * ppf_x + stats.norm.ppf(0.05 / 2)
    low = 2 * ppf_x - stats.norm.ppf(0.05 / 2)
    up_cdf = stats.norm.cdf(up)
    low_cdf = stats.norm.cdf(low)
    upx = np.percentile(ab_array, low_cdf)
    lowx = np.percentile(ab_array, up_cdf)

    return lowx, upx

