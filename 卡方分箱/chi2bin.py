import numpy as np
from numpy import min, sum
import pandas as pd
from distributed import Client
from numba import jit
import ray
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, minmax_scale
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


# import modin.pandas as pd
# ray.init()


@jit(fastmath=True)
def Chi2(df, overallRate):
    bad = df[:, 2]
    expected = df[:, 1] * overallRate
    chi2 = np.sum((expected - bad) ** 2 / expected)
    return chi2


@jit(forceobj=True, fastmath=True)
def while_done(regroup, groupIntervals, max_interval, groupNum, overallRate):
    while (len(groupIntervals) > max_interval):
        chisqList = []
        for interval in groupIntervals:
            df2 = regroup[np.isin(regroup[:, 0], interval)]
            chisq = Chi2(df2, overallRate)
            chisqList.append(chisq)
        min_position = chisqList.index(min(chisqList))
        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position - 1
        else:
            if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
        # 合并箱体
        groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
        # print(len(groupIntervals))
    return groupIntervals


# 最大分箱数分箱
def ChiMerge_MaxInterval_Original(df, col, target, max_interval=5):
    colLevels = np.unique(df[col].values).tolist()
    N_distinct = len(colLevels)
    print(N_distinct)
    if N_distinct <= max_interval:
        print("the row is cann't be less than interval numbers")
        return colLevels[:-1]
    else:
        total = df.groupby([col])[target].count()
        total = pd.DataFrame({'total': total})
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        # print(regroup.columns)
        # ['all_y_hat1', 'total', 'bad']
        regroup = regroup.values
        N = sum(regroup[:, 1])
        B = sum(regroup[:, 2])
        overallRate = B * 1.0 / N
        groupIntervals = [[i] for i in colLevels]
        groupNum = len(groupIntervals)
        groupIntervals = while_done(regroup, groupIntervals, max_interval, groupNum, overallRate)
        groupIntervals = [sorted(i) for i in groupIntervals]
        # print(groupIntervals)
        cutOffPoints = [i[-1] for i in groupIntervals[:-1]]
        return cutOffPoints


def Rank_qcut(vector, K):
    quantile = np.array([float(i) / K for i in range(K + 1)])  # Quantile: K+1 values
    funBounder = lambda x: (quantile >= x).argmax()
    return vector.rank(pct=True).apply(funBounder)


def Discretization_EqualFrequency(K, Datas, FeatureNumber):
    DisDatas = np.zeros_like(Datas)
    # w = [float(i) / K for i in range(K + 1)]
    for i in range(FeatureNumber):
        DisOneFeature = Rank_qcut(pd.Series(Datas.iloc[:, i]), K)
        DisDatas[:, i] = DisOneFeature

    return DisDatas

