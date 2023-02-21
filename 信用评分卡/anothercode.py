# !/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import shap
from var_deal import RunBase, WOE
import pickle
from multiprocessing import cpu_count
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('application_train.csv')
y_data = data['TARGET']
del data['TARGET']
print(data.head(5))
print(data.isnull().sum(axis=0))
data = data.fillna(0)
str_var_name = []
num_var_name = []
cols = data.columns
for col in cols:
    # print(data[col].dtype)
    if 'float' in str(data[col].dtype):
        num_var_name.append(col)
    else:
        str_var_name.append(col)
data_str = data[str_var_name]
data_num = data[num_var_name]
#描述性统计,matplotlib/seaborn可以画图可视化,必要的可用KDE估计分布
Str_View_data = RunBase(data=data_str, VarNames=np.array(str_var_name), kind='Str')
Num_View_data = RunBase(data=data_num, VarNames=np.array(num_var_name), kind='Num')
Str_View_data.to_csv('Str_View_data.csv', index=False, encoding='ansi')
Num_View_data.to_csv('Num_View_data.csv', index=False, encoding='ansi')
print(Str_View_data)
print(Num_View_data)

# 写入数据的WOE,IV为Excel
WOE(data_str, varList=str_var_name, type0='CAT', y_data=y_data, encoding='ansi')
WOE(data_num, varList=data_num, type0='CON', y_data=y_data, encoding='ansi')
print(pd.read_csv('X_CON2.csv'))
print(pd.read_csv('X_CAT1.csv'))
ont_hotMdl = OneHotEncoder(sparse=False).fit(data[str_var_name].values)
# 保存模型
pickle.dump(file=open('OneHotMdl.pkl', 'wb'), obj=ont_hotMdl)
data_new = np.concatenate((data_num.values, ont_hotMdl.transform(data_str.values)), axis=1)

# 可以分割用于做CV,消费贷需要跨时间做验证
X_train, X_test, y_train, y_test = train_test_split(data_new, y_data.values, test_size=0.3)


'''
#LR
#odds:p/1-p
#MLE估计求解
#建议特征根据WOE分箱因LR是线性模型

#DT
#ID3用信息增益
#ID4.5用信息增益比(惩罚数量)
#剪枝加正则回退父结点算剪枝后的损失函数是否更优

#GBDT
#与RF的抽样后学习弱学习器然后投票多数决定类别不同,GBDT对上次误差进行继续学习
#误差用对特征的负梯度计算(加速)
#用了CART的回归树形式通过构造损失函数获取最佳切分点,分类用基尼系数
#贪心算法算所有的获取最大特征

#XGBoost
# 对目标函数进行了麦克劳林的泰勒展开,用到了二阶导数,并增加正则项工程上加快模型收敛和控制overfit
# 每棵树用衰减的学习权重加权使得不会学的太少(也是控制overfit)
# 分裂结点用上述目标函数贪心算法求解(此处算法跟GBDT一致,用目标函数替代信息增益/基尼系数等)
# 控制overfit:有可能分裂的时候随机选择部分特征来决定最优分割点
# 加速:对特征分桶(bin),使得只需要遍历每个特征的分位点而非所有特征的unique值加速
# 缺失值:先不处理,最后遇到的时候同时计算左右节点的值然后算增益大的

#LightGBM
#连续特征也分桶加速,这部分精度会下降但弱学习器精度不是特别重要而且还能控制overfit
#分桶后构建直方图,用差方法计算兄弟节点直方图加速
#Goss:前a%大梯度样本,后面b%小梯度样本抽样合并加速并保证分布
#Left-wise:在叶子就计算增益并分裂提高精度,但需要控制叶子最大深度控制overfit
#互斥特征合并加速
#非零数量降序,满足条件就拓展值域引入偏移合并特征加速
'''
model = lgb.LGBMClassifier(boosting_type='goss', n_jobs=cpu_count() - 1, )
model.fit(X_train, y_train)
print(model.n_features_)

# 模型贡献度放在feture中
feature = pd.DataFrame(
    {'name': model.booster_.feature_name(),
     'importance': model.feature_importances_
     }).sort_values(by=['importance'], ascending=False)

# 计算训练集、测试集、验证集上的KS和AUC

y_pred_train_lgb = model.predict_proba(X_train)[:, 1]
y_pred_test_lgb = model.predict_proba(X_test)[:, 1]
train_fpr_lgb, train_tpr_lgb, _ = roc_curve(y_train, y_pred_train_lgb)
test_fpr_lgb, test_tpr_lgb, _ = roc_curve(y_test, y_pred_test_lgb)
train_ks = abs(train_fpr_lgb - train_tpr_lgb).max()
test_ks = abs(test_fpr_lgb - test_tpr_lgb).max()
train_auc = auc(train_fpr_lgb, train_tpr_lgb)
test_auc = auc(test_fpr_lgb, test_tpr_lgb)
#KS大于0.25-0.5可用
print('train_ks: ', train_ks)
print('test_ks: ', test_ks)

'''
另外的指标
PSI=sum((Ai-Ei)*ln(Ai/Ei))
#PSI大于0.25IV小于0.02的指标可以剔除,VIF>10或corrcoef>0.7的可以剔除,也可根据chi2和信息熵F.Lasso去筛选
CSI=sum((A%-E%)*WOE)
Lift=(TP/(TP+FP))/((TP+FN)/(TP+FN+FP+TN))
concordant =(TP+FP) / (TP+FN+FP+TN)
disconcordant = (FN+TN) / (TP+FN+FP+TN)
gamma = (concordant - disconcordant) / (concordant + disconcordant)
somersd = concordant - disconcordant
tau_a = (concordant - disconcordant) / ((TP+FN+FP+TN) * ((TP+FN+FP+TN) - 1)) / 2)
c = (somersd + 1) / 2
'''

# ROC曲线,大于0.7最好
plt.plot(train_fpr_lgb, train_tpr_lgb, label='train LR')
plt.plot(test_fpr_lgb, test_tpr_lgb, label='test LR')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# SHAP
explainer = shap.TreeExplainer(model)
shap.initjs()
shap_values = explainer.shap_values(X_train)
# 每个特征的贡献
shap.summary_plot(shap_values, X_train)