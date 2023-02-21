# 导入相关库
import copy
import pandas as pd
import numpy as np
from QCdec import DataDecAnalyst
import itertools
from numba import jit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, auc, accuracy_score, recall_score, roc_curve,confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('data.csv')

label_need = ['Guarantee_Type_Ind', 'Account_Count', 'Type_Count', 'Changing_Amount', 'Credit_Limit', 'Balance',
              'Type_Dw', 'Normal_State_Ratio_Ind', 'Changing_Months', 'Loancard_Count', 'Min_Credit_Limit_Per_Org',
              'Other_Loan_Count', 'Max_Duration', 'Finance_Corp_Count', 'Payment_Cyc_In',
              'Scheduled_Payment_Amount_Ind', 'Finance_Org_Count', 'Normal_Class5_Ratio_In',
              'Share_Credit_Limit_Amount', 'Latest_6m_Used_Avg_Amount', 'Standard_Loancard_Count',
              'Remain_Payment_Cyc_In', 'House_Loan_Count', 'Max_Credit_Limit_Per_Org', 'Used_Credit_Limit_Amount',
              'Scheduled_Payment_Amount_In', 'Agent', 'Curr_Overdue_Cyc_Ind', 'Amount', 'Last_Months', 'Announce_Count',
              'Curr_Overdue_Amount_In', 'Curr_Overdue_Amount_Ind', 'Dissent_Count', 'Count_Dw', 'Months',
              'Highest_Oa_Per_Mon', 'Curr_Overdue_Cyc_In', 'Commercial_Loan_Count', 'Is_Local', 'Sex',
              'Has_Fund', 'Work_Province', 'Marry_Status', 'Edu_Level', 'Salary', 'Actual_Payment_Amount_In',
              'Payment_Rating_In', 'Guarantee_Type_In', 'Normal_State_Ratio_In', 'Currency_Ind',
              'Actual_Payment_Amount_Ind', 'Currency_In', 'Used_Highest_Amount']

print(label_need)
data1 = data[label_need].values
# print(data1)

# 指标正向    化处理后数据为data2
data2 = data1
# print(data2)

# 越小越优指标位置,注意python是从0开始计数，对应位置也要相应减1
index = np.arange(39).tolist()
for i in range(0, len(index)):
    data2[:, index[i]] = max(data1[:, index[i]]) - data1[:, index[i]]
# print(data2)

# 0.002~1区间归一化
[m, n] = data2.shape
data3 = copy.deepcopy(data2)
ymin = 0.002
ymax = 1
for j in range(0, n):
    d_max = max(data2[:, j])
    d_min = min(data2[:, j])
    data3[:, j] = (ymax - ymin) * (data2[:, j] - d_min) / (d_max - d_min) + ymin
# 计算信息熵
p = copy.deepcopy(data3)
for j in range(0, n):
    p[:, j] = data3[:, j] / sum(data3[:, j])
E = copy.deepcopy(data3[0, :])
for j in range(0, n):
    E[j] = -1 / np.log(m) * sum(p[:, j] * np.log(p[:, j]))

# 计算权重
w = (1 - E) / sum(1 - E)
# print(w)

s = np.dot(data3, w)
Score = 100 * s / max(s)
data['S_Score'] = Score
data['S_Score'] = data['S_Score'].astype(float)
d1 = pd.pivot_table(data, values='S_Score', aggfunc='mean', index=['Report_Id']). \
    reset_index().sort_values(by='S_Score', ascending=False)

# d1.to_csv('评分结果.csv', encoding='ansi')

# 获取因子权重
d2 = pd.DataFrame(np.hstack((np.array(label_need).reshape(-1, 1), w.reshape(-1, 1))),
                  columns=['因子', '权重']).sort_values(by='权重', ascending=False)
# d2.to_csv('权重结果.csv', encoding='ansi')

num_name = ['Curr_Overdue_Cyc_Ind', 'Amount', 'Last_Months', 'Announce_Count', 'Curr_Overdue_Amount_In',
            'Curr_Overdue_Amount_Ind', 'Dissent_Count', 'Count_Dw', 'Months', 'Highest_Oa_Per_Mon',
            'Curr_Overdue_Cyc_In', 'Commercial_Loan_Count', 'Actual_Payment_Amount_In', 'Balance', 'Loancard_Count',
            'Payment_Rating_In', 'Normal_State_Ratio_Ind', 'Changing_Months', 'Scheduled_Payment_Amount_Ind',
            'Finance_Org_Count', 'Remain_Payment_Cyc_In', 'House_Loan_Count', 'Max_Credit_Limit_Per_Org',
            'Used_Credit_Limit_Amount', 'Scheduled_Payment_Amount_In', 'Normal_Class5_Ratio_In',
            'Latest_6m_Used_Avg_Amount', 'Standard_Loancard_Count',
            'Normal_State_Ratio_In', 'Min_Credit_Limit_Per_Org', 'Other_Loan_Count', 'Max_Duration',
            'Finance_Corp_Count', 'Payment_Cyc_In', 'Account_Count', 'Actual_Payment_Amount_Ind', 'Type_Count',
            'Changing_Amount', 'Credit_Limit', 'Used_Highest_Amount']

str_name = ['Is_Local', 'Sex', 'Has_Fund', 'Agent', 'Work_Province', 'Marry_Status', 'Edu_Level', 'Salary',
            'Guarantee_Type_Ind', 'Type_Dw', 'Guarantee_Type_In', 'Currency_Ind', 'Currency_In',
            'Share_Credit_Limit_Amount']
DataDecAnalyst(str_name=str_name, num_name=num_name, data=data, step='')

Y = data['Y']

bin_cut_df = copy.deepcopy(
    data[['Y', 'Is_Local', 'Sex', 'Has_Fund', 'Agent', 'Work_Province', 'Marry_Status', 'Edu_Level', 'Salary',
          'Guarantee_Type_Ind', 'Type_Dw', 'Guarantee_Type_In', 'Currency_Ind', 'Currency_In',
          'Share_Credit_Limit_Amount']])
print(num_name)


def compute_woe_iv(df, var_list, Y_flag):
    var_num = len(var_list)
    totalG_B = df.groupby([Y_flag])[Y_flag].count()  # 计算正负样本多少个
    G = totalG_B[1]
    B = totalG_B[0]
    woe_all = np.zeros((1, 8))
    var_iv = np.zeros((var_num))
    data_index = []
    for k in range(0, var_num):

        var1 = df.groupby([var_list[k]])[Y_flag].count()  # 计算col每个分组中的组的个数

        var_class = var1.shape[0]
        woe = np.zeros((var_class, 8))
        woe_pre = pd.DataFrame(data={'x1': [], 'ifgood': [], 'values': []})
        total = df.groupby([var_list[k], Y_flag])[Y_flag].count()  # 计算该变量下每个分组响应个数
        total1 = pd.DataFrame({'total': total})
        mu = []
        for u, group in df.groupby([var_list[k], Y_flag])[Y_flag]:
            mu.append(list(u))
        for lab1 in total.index.levels[0]:
            for lab2 in total.index.levels[1]:
                # temporary = pd.DataFrame(data={'x1': [lab1], 'ifgood': [lab2], 'values': [1]})
                if [lab1, lab2] not in mu:
                    temporary = pd.DataFrame(data={'x1': [lab1], 'ifgood': [lab2], 'values': [0]})
                else:
                    temporary = pd.DataFrame(
                        data={'x1': [lab1], 'ifgood': [lab2], 'values': [total1.xs((lab1, lab2)).values[0]]})
                woe_pre = pd.concat([woe_pre, temporary])
            # print(woe_pre)
        woe_pre.set_index(['x1', 'ifgood'], inplace=True)

        # 计算 WOE
        for i in range(0, var_class):  # var_class
            woe[i, 0] = woe_pre.values[2 * i + 1]
            woe[i, 1] = woe_pre.values[2 * i]
            woe[i, 2] = woe[i, 0] + woe[i, 1]
            woe[i, 3] = woe_pre.values[2 * i + 1] / G  # pyi
            woe[i, 4] = woe_pre.values[2 * i] / B  # pni
            abb = lambda i: (np.log(woe[i, 3] / woe[i, 4])) if woe[i, 3] != 0 else 0  # 防止 ln 函数值域报错
            woe[i, 5] = abb(i)
            woe[np.isinf(woe)] = 0  # 将无穷大替换为0，参与计算 woe 计算

            woe[i, 6] = (woe[i, 3] - woe[i, 4]) * woe[i, 5]  # iv_part
            var_iv[k] += woe[i, 6]
        iv_signal = np.zeros((1, 8))
        iv_signal[0, 7] = var_iv[k]
        woe_all = np.r_[woe_all, woe, iv_signal]
        index_var = df.groupby([var_list[k]])[Y_flag].count()
        u = index_var.index.values.tolist()
        data_index += u
        data_index += [var_list[k]]
    woe_all = np.delete(woe_all, 0, axis=0)
    result = pd.DataFrame(data=woe_all, columns=['good', 'bad', 'class_sum', 'pyi', 'pni', 'woe', 'iv_part', 'iv'])
    result.index = data_index
    result = result.reset_index()
    result = result[result['iv'] != 0][['index', 'iv']]
    print(result)
    return result


var_name_all = copy.deepcopy(str_name)
var_name_all.extend(num_name)

all_name = copy.deepcopy(var_name_all)
all_name.append('Y')
bin_cut_df = data[all_name]

iv_result = compute_woe_iv(df=bin_cut_df, var_list=str_name, Y_flag='Y').values
for ind, row in enumerate(iv_result):
    if row[1] < 0.02:
        print('del', row[0])
        del bin_cut_df[row[0]]
print(list(bin_cut_df.columns))

x = scale(bin_cut_df[['Curr_Overdue_Cyc_Ind', 'Amount', 'Last_Months', 'Announce_Count', 'Curr_Overdue_Amount_In',
                      'Curr_Overdue_Amount_Ind', 'Dissent_Count', 'Count_Dw', 'Months', 'Highest_Oa_Per_Mon',
                      'Curr_Overdue_Cyc_In', 'Commercial_Loan_Count', 'Actual_Payment_Amount_In', 'Balance',
                      'Loancard_Count',
                      'Payment_Rating_In', 'Normal_State_Ratio_Ind', 'Changing_Months', 'Scheduled_Payment_Amount_Ind',
                      'Finance_Org_Count', 'Remain_Payment_Cyc_In', 'House_Loan_Count', 'Max_Credit_Limit_Per_Org',
                      'Used_Credit_Limit_Amount', 'Scheduled_Payment_Amount_In', 'Normal_Class5_Ratio_In',
                      'Latest_6m_Used_Avg_Amount', 'Standard_Loancard_Count', 'Normal_State_Ratio_In',
                      'Min_Credit_Limit_Per_Org', 'Other_Loan_Count', 'Max_Duration', 'Finance_Corp_Count',
                      'Payment_Cyc_In',
                      'Account_Count', 'Actual_Payment_Amount_Ind', 'Type_Count', 'Changing_Amount', 'Credit_Limit',
                      'Used_Highest_Amount']].values)
y = bin_cut_df[['Y']].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
mdl1 = RandomForestClassifier(bootstrap=False,random_state=41,n_estimators=13,max_leaf_nodes=4).fit(X_train, y_train)
pred1 = mdl1.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, pred1, pos_label=1)
from  sklearn.metrics import plot_roc_curve

print(auc(fpr, tpr))
print(f1_score(y_test, pred1))
print(accuracy_score(y_test, pred1))
print(confusion_matrix(y_test, pred1))

a=[]
b=[]
for i in range(10):
    print(i)
    a.append(0.9+i*0.05)
    mdl2 = SVC(probability=True, random_state=42,C=0.9+i*0.05).fit(X_train, y_train.reshape(-1))
    pred2 = mdl2.predict(X_test)
    #print(auc(fpr, tpr))
    b.append(f1_score(y_test, pred2))
    print(f1_score(y_test, pred2))
    #print(accuracy_score(y_test, pred2))
    #print(confusion_matrix(y_test, pred2))

import matplotlib.pyplot as plt
plt.plot(a,b)

plt.ylabel('F1-score')
plt.xlabel('C')
plt.title('F1-score of SVM C ')
plt.show()
mdl2 = SVC(probability=True,random_state=42).fit(X_train, y_train)
pred2 = mdl2.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, pred2, pos_label=1)
print(auc(fpr, tpr))
print(f1_score(y_test, pred2))
print(accuracy_score(y_test, pred2))
print(confusion_matrix(y_test, pred2))
import matplotlib.pyplot as plt
from  sklearn.metrics import plot_roc_curve
plot_roc_curve(mdl2, X_test, y_test)
plt.show()
plot_roc_curve(mdl1, X_test, y_test)
plt.show()
# 处理报告
def score(xbeta):
    score = 1000 - 500 * (np.log2(1 - xbeta) / xbeta)
    return score


evl = pd.DataFrame(mdl2.predict_proba(X_test)[:, 1], columns=['xbeta'])
evl['score'] = evl.apply(lambda x: score(x.xbeta), axis=1)

row_num, col_num = 0, 0
bins = 20
Y_predict = evl['score']
Y = y_test
nrows = Y.shape[0]
lis = [(Y_predict[i], Y[i]) for i in range(nrows)]
ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)
bin_num = int(nrows / bins + 1)
bad = np.sum([1 for (p, y) in ks_lis if y > 0.5])
good = np.sum([1 for (p, y) in ks_lis if y <= 0.5])
bad_cnt, good_cnt = 0, 0
KS = []
BAD = []
GOOD = []
BAD_CNT = []
GOOD_CNT = []
BAD_PCTG = []
BADRATE = []
dct_report = {}
for j in range(bins):
    ds = ks_lis[j * bin_num: min((j + 1) * bin_num, nrows)]
    bad1 = np.sum([1 for (p, y) in ds if y > 0.5])
    good1 = np.sum([1 for (p, y) in ds if y <= 0.5])
    bad_cnt += bad1
    good_cnt += good1
    bad_pctg = np.round(bad_cnt / np.sum(Y), 3)
    badrate = np.round(bad1 / (bad1 + good1), 3)
    ks = np.round(np.fabs((bad_cnt / bad) - (good_cnt / good)), 3)
    KS.append(ks)
    BAD.append(bad1)
    GOOD.append(good1)
    BAD_CNT.append(bad_cnt)
    GOOD_CNT.append(good_cnt)
    BAD_PCTG.append(bad_pctg)
    BADRATE.append(badrate)
    dct_report['KS'] = KS
    dct_report['BAD'] = BAD
    dct_report['GOOD'] = GOOD
    dct_report['BAD_CNT'] = BAD_CNT
    dct_report['GOOD_CNT'] = GOOD_CNT
    dct_report['BAD_PCTG'] = BAD_PCTG
    dct_report['BADRATE'] = BADRATE
val_repot = pd.DataFrame(dct_report)


print(1)
