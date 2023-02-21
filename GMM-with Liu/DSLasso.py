from sklearn.linear_model import  LassoCV
import numpy as np
def dslasso(A, b):
    doge1 = LassoCV(cv=5).fit(A, b).coef_
    # 第一个非0集
    Set1 = doge1 != 0
    Set2x = np.zeros((len(Set1), A.shape[1]))
    index1 = np.argwhere(Set1 == True)[:, 0]
    for i in index1:
        print('a1', i)
        y_ds = A[:, i].reshape(-1, 1)
        selct = np.array(range(A.shape[1])) != i
        x_ds = A[:, selct]
        para_ds = LassoCV(cv=5).fit(x_ds, y_ds).coef_
        Set2x[i, selct] = (para_ds != 0).reshape(-1)
    result_st = np.sum(Set2x, axis=0)
    result_sr = np.sum(Set2x, axis=0) / len(index1)
    print('DS Lasso select time:', result_st)
    print('DS Lasso ratio:', result_sr)
    Set_2 = np.max(Set2x, axis=0)
    Set_all = np.max(np.concatenate((Set1.reshape(1, -1), Set_2.reshape(1, -1))), axis=0)
    return Set1, Set_2, Set_all, result_st, result_sr
