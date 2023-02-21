'''
Copyright <2021> <Ben Wan: wanbenfighting@gmail.com>
'''
import numpy as np
from numpy.linalg import inv
from numba import jit
import time
from numpy.random import choice
from copy import deepcopy
from numba.typed import List


@jit()
def ZScoreCatCon(data, rank):
    xx = 0
    for i in range(data.shape[1]):
        col_temp = data[:, i]
        sd = np.std(col_temp)
        if sd != 0:
            temp = ((col_temp - np.mean(col_temp)) / np.std(col_temp) * rank[i]).reshape(-1, 1)
        else:
            temp = ((col_temp - np.mean(col_temp)) * rank[i]).reshape(-1, 1)
        if xx == 0:
            result = temp
            xx = 1
        else:
            result = np.hstack((result, temp))
    return result
    
@jit(fastmath=True, parallel=True)
# LDL分解求逆(对称正定矩阵)
def LDL_decompose_inverse(matrix):
    w = matrix.shape[0]
    L = np.zeros((w, w))
    for i in range(w):
        L[i, i] = 1
    D = np.zeros((w, w))
    for i in range(w):
        D[i, i] = matrix[i, i] - np.dot(np.dot(L[i, :i], D[:i, :i]), L[i, :i].T)
        for j in range(i + 1, w):
            L[j, i] = (matrix[j, i] - np.dot(np.dot(L[j, :i], D[:i, :i]), L[i, :i].T)) / D[i, i]
    Ax = inv(L.transpose()) @ inv(D) @ inv(L)
    return Ax


@jit(fastmath=True, parallel=True)
# LU分解求逆(正定矩阵)
def LU_decompose_inverse(A):
    n = len(A)
    L = np.zeros((n, n))
    for base in range(n - 1):
        for i in range(base + 1, n):
            L[i, base] = A[i, base] / A[base, base]
            A[i] = A[i] - L[i, base] * A[base]
    for i in range(n):  # range(n) 范围：[0，n-1]
        L[i, i] = 1
    U = A
    Ax = inv(L) @ inv(U)
    return Ax

@jit(fastmath=True)
def meanAxis(matrix):
    result = np.zeros((matrix.shape[1]))
    for i in range(matrix.shape[1]):
        result[i] = np.mean(matrix[:, i])
    return result


# 协方差并行计算
@jit(fastmath=True)
def cov_parallel(cov):
    if cov.shape[0] < 3:
        all_cov = np.cov(cov.T, ddof=0)
    else:
        # Erich Schubert, 2018. Numerically Stable Parallel Computation of (Co-)Variance, SSDBM
        n = len(cov) // 2
        shapex = cov.shape[1]
        all_cov = np.zeros((shapex, shapex))
        V_x = np.zeros((2, shapex))
        v_x = np.zeros((2, shapex))
        # 类似这种的是因为numba加速不支持2-D的运算(axis=~)
        meanx = meanAxis(cov)
        for i in range(shapex):
            cov[:, i] -= meanx[i]
        A = cov[0:n]
        B = cov[n:]
        og_A = len(A) * 0.5
        og_B = len(B) * 0.5
        og_AB = og_A + og_B
        for i in range(shapex):
            V_x[0, i] = np.sum(A[:, i] * 0.5)
            V_x[1, i] = np.sum(B[:, i] * 0.5)
        v_x[0:] = V_x[0:] / og_A
        v_x[1:] = V_x[1:] / og_B

        for i in range(shapex):
            for j in range(shapex):
                VW_A = np.sum(0.5 * ((A[:, i]) - (1 / og_A) * V_x[0, i]) * ((A[:, j]) - (1 / og_A) * V_x[0, j]))
                VW_B = np.sum(0.5 * ((B[:, i]) - (1 / og_B) * V_x[1, i]) * ((B[:, j]) - (1 / og_B) * V_x[1, j]))
                all_cov[i, j] = (VW_A + VW_B + (og_A * og_B) / og_AB * (v_x[0, i] - v_x[1, i]) * (
                        v_x[0, j] - v_x[1, j])) / (og_AB - 1)
    return all_cov


@jit(fastmath=True)
# 组欧式距离
def Edis(mat):
    shape = mat.shape[0]
    dis_mat = np.zeros((shape, shape))
    for i in range(shape):
        for j in range(shape):
            dis_mat[i, j] = (np.mean((mat[i, :] - mat[j, :]) ** 2)) ** 0.5
    return dis_mat


# 求马氏距离
@jit()
def mdis(data_new, data_merge):
    meanx = meanAxis(data_merge)
    data_new -= meanx
    A = np.cov(data_merge.T)
    try:
        B = np.linalg.inv(A)
        D2 = (data_new @ B @ data_new.transpose()) / 100
    except:
        B = np.linalg.pinv(A)
        D2 = (data_new @ B @ data_new.transpose()) / 100
    # except这部分,不属于马氏距离的,一般来说要去掉
    if D2 < 0:
        B = np.linalg.pinv(A)
        D2 = (data_new @ B @ data_new.transpose()) / 100
    return D2


# 根据批中心的马氏距离做比较优势抽样
@jit()
def Mdis_Group(data: np.ndarray, group: int, times_all: int):
    '''
    我也不知道这有没有别人写过,没有就当做是我设计的model吧doge,因此请勿用于学术用途
    1.给定data,抽样的组数,重要性权重和抽样次数将data根据马氏距离尽可能均分成抽样的组数,可用于控制AB测试的测试组和对照组
    2.截断样本至整分的,归一后然后根据重要性加权,分类用onehot
    3.随机取一个batch作为初始分组
    4.然后每次取一个batch计算质心和所有组样本的马氏距离,如果inv裂开就用pinv
    5.取距离最小的组分过去
    '''
    alter = 0
    alter = data.shape[0]
    n_sample = alter // (group * times_all)
    # 对样本取整方便计算
    # data_x预分配了内存,方便抽样,基于同样的原因该部分代码写在并行外面
    data_x = np.zeros((group * n_sample * times_all))
    set_n = [[np.float(x) for x in range(0)] for _ in range(group)]
    # 先随机分配样本給每个组
    sample = choice(np.argwhere(data_x == 0).reshape(-1), group * n_sample, replace=False). \
        reshape(group, n_sample).tolist()
    for i in range(group):
        data_x[sample[i]] = i + 1
        set_n[i].extend(np.copy(sample[i]).tolist())

    for i in range(times_all - 1):
        print('times', i)
        # 随机抽2个样本,分别拼接到AB,计算出4个马氏距离(的平方)A1,B1,A2,B2
        # replace=False是不放回抽样
        Mdis_mat = np.zeros((group, group))

        sample = choice(np.argwhere(data_x == 0).reshape(-1), group * n_sample, replace=False). \
            reshape(group, n_sample).tolist()
        # 第j组样本
        for j in range(group):
            # 第k个数据集
            for k in range(group):
                # 因为set1,set2是动态的所以要用copy
                # 批样本质心
                cluster = meanAxis(data[sample[j]]) - meanAxis(data[data_x == k])
                temp1 = np.copy(set_n[k]).tolist()
                temp1.extend(sample[j])

                Mdis_mat[j, k] = mdis(cluster, data[temp1])

        Mdis_mat = np.round(Mdis_mat, 6)
        resultx = np.zeros((group))
        for j in range(group):
            indx = np.argmin(Mdis_mat[j])
            resultx[j] = indx
            Mdis_mat[:, indx] = 10 ** 99
        for j in range(group):
            set_n[int(resultx[j])].extend(np.copy(sample[j]).tolist())
            data_x[sample[j]] = resultx[j] + 1
    result = np.hstack((data, data_x.reshape(-1, 1)))
    return result, set_n


@jit()
def absdiff(data):
    groupnum = np.unique(data[:, -1])
    result = np.zeros((len(groupnum), len(groupnum)))
    for i in range(len(groupnum)):
        temp1 = data[data[:, -1] == groupnum[i], :-1]
        for j in range(len(groupnum)):
            temp2 = data[data[:, -1] == groupnum[j], :-1]
            result[i, j] = np.sum(np.abs(temp1 - temp2))
    return result


# @jit()
def run_grouping(data: np.ndarray, rank_rate, group: int, times_all: int):
    '''
    # 随机样本量
    alter = 1000 l
    # 特征数
    features = 6
    # 组数
    group = 2
    # 抽样次数
    times_all = 10
    # 随机生成样本
    data = np.random.randint(high=13, low=10, size=(alter, features)).astype(float)
    # 为分类样本的列
    cat = []
    rank_rate = [1] * features
    print(1)
    '''
    n_sample = data.shape[0] // (group * times_all) * group * times_all
    datax = data[:n_sample, :]
    print('score')
    deal_data = ZScoreCatCon(datax, rank_rate)
    print('score1')
    # init_index = np.arange(features).tolist()
    # data=data[]

    t1 = time.time()
    print('start')
    result, setx = Mdis_Group(deal_data, group, times_all)
    # print(Mdis_Group.inspect_types())
    t2 = time.time()
    print('运行时间:', t2 - t1)
    check_series = np.zeros((group, result.shape[1] - 1))
    for i in range(group):
        check_series[i, :] = meanAxis(result[result[:, -1] == i + 1][:, :-1])

    emat = Edis(check_series)
    print('重要性加权标化后各组欧氏距离矩阵\n', emat)
    print('重要性加权标化后各组各特征欧氏距离均值\n', np.mean(emat, axis=1))

    # print(std)
    # print(inter)
    print('abs diff:\n', absdiff(result))
    return setx


# @jit()
def Mdis_parafor(data1: np.ndarray, data2: np.ndarray):
    setx = []
    result = []
    test_cluster = meanAxis(data1)
    temp = data2 - test_cluster

    indx = int(np.random.randint(0, len(data2) - 1, 1))
    result.append(data2[indx].tolist())
    setx.append(indx)
    temp[indx] = 10 ** 99
    for iter in range(len(data1) - 1):

        mdis_all = np.zeros((len(data2)))
        for i in range(len(data2)):
            if len(result) > 0:
                mdis_all[i] = mdis(meanAxis(np.vstack((np.array(result) - test_cluster, temp[i]))), data1)
            else:
                mdis_all[i] = mdis(temp[i], data1)
        indx = np.argmin(mdis_all)
        result.append(data2[indx].tolist())
        temp[indx] = 10 ** 99
        setx.append(indx)

    return result, setx


# @jit()
def run_parafor(data1: np.ndarray, data2: np.ndarray, rank_rate):
    '''
    样本切记不能太大O(mn),将对照组扔进全量备选对照组做Z-score(当然也是共同支撑假设~),然后贪心算法干最小的.
    data1:测试组,数量为m
    data2:需要采样的全量对照组,数量为n
    rank_rate:特征权重
    '''
    alldata = np.vstack((data1, data2))
    alldata = ZScoreCatCon(alldata, rank_rate)
    data1 = alldata[:len(data1)]
    data2 = alldata[len(data1):]

    t1 = time.time()
    print('start')
    result, setx = Mdis_parafor(data1, data2)
    # print(Mdis_Group.inspect_types())
    t2 = time.time()
    print('运行时间:', t2 - t1)

    return setx
