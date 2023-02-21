import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
import warnings
from scipy.linalg import norm
from numba import jit

warnings.filterwarnings('ignore')


def cut_decile(df, col='y_hat', target_cols=[]):
    df['decile'] = pd.qcut(df[col], q=10, labels=[i for i in range(10)])
    df['decile'] = pd.qcut(df[col], q=10, labels=[i for i in range(10)])
    first_cut_q = df[['product_id', 'decile'] + target_cols].groupby('decile').agg({
        "product_id": "count",
        "target": "mean",
        "ProfitF": "sum",
        "SaleCntF": "sum"
    }).reset_index().rename(columns={"product_id": "n"})
    first_cut_q['profit_avg'] = first_cut_q['ProfitF'] / first_cut_q['SaleCntF']
    first_cut_q['salecnt_avg'] = first_cut_q['SaleCntF'] / first_cut_q['n']
    first_cut_q['n_pct'] = first_cut_q['n'] / first_cut_q.n.sum()
    first_cut_q['profit_pct'] = first_cut_q['ProfitF'] / first_cut_q.ProfitF.sum()
    first_cut_q['GINI'] = first_cut_q['profit_pct'] / first_cut_q['n_pct']
    first_cut_q = first_cut_q.sort_index(ascending=False)
    first_cut_q['cum_profit_pct'] = first_cut_q['profit_pct'].cumsum()
    first_cut_q['cum_n_pct'] = first_cut_q['n_pct'].cumsum()
    return first_cut_q


def plot_q(d2x, title, prob='prob', save_fig=True):
    decile_res = cut_decile(d2x, col=prob, target_cols=['target', 'ProfitF', 'SaleCntF'])
    plt.figure(figsize=(12, 8))
    sns.lineplot(np.linspace(0, 1, 11), [0] + list(decile_res['cum_profit_pct']))
    sns.lineplot(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    # print(d2.prob.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    plt.title(title)
    if save_fig:
        plt.savefig(title + '.png')
    plt.show()


def model_show(model, xdata, ydata, X_test, y_test, d2):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    all_pred_prob = model.predict_proba(xdata)[:, 1]

    y_pred = model.predict(X_test)
    all_pred = model.predict(xdata)

    print('all_roc_auc', np.round(roc_auc_score(ydata, all_pred_prob), 4))
    print('all-F1', np.round(f1_score(ydata, all_pred), 4))
    print('all-recall', np.round(recall_score(ydata, all_pred), 4))
    print('all-pre', np.round(precision_score(ydata, all_pred), 4))
    print('all-acc', np.round(accuracy_score(ydata, all_pred), 4))
    print('test_roc_auc', np.round(roc_auc_score(y_test, y_pred_prob), 4))
    print('test-F1', np.round(f1_score(y_test, y_pred), 4))
    print('test-recall', np.round(recall_score(y_test, y_pred), 4))
    print('test-pre', np.round(precision_score(y_test, y_pred), 4))
    print('test-acc', np.round(accuracy_score(y_test, y_pred), 4))
    d2['prob'] = all_pred_prob
    # d2.to_csv('result.csv', encoding='ansi')
    # print(d2.columns)

    sns.distplot(y_pred_prob)
    plt.show()
    sns.distplot(all_pred_prob)
    plt.show()
    plot_q(d2, 'all', 'prob', False)
    plot_q(d2[(d2['SaleAmtL12'] == 0) & (d2['live_days'] < 90)], 'live_less_90', 'prob', False)
    plot_q(d2[(d2['SaleAmtL12'] == 0) & (d2['live_days'] < 90)], 'live_less_90', 'prob', False)
    print(len(d2[(d2['SaleAmtL12'] == 0) & (d2['live_days'] < 90)]))


d1 = pd.read_csv('plproduct.csv')
print(list(d1.columns))
d1 = d1[['target', 'SaleCntL1', 'SaleAmtL1', 'ProfitL1', 'UserUniCntL1', 'SaleCntL3', 'SaleAmtL3', 'ProfitL3',
         'UserUniCntL3', 'SaleCntL6', 'SaleAmtL6', 'ProfitL6', 'UserUniCntL6', 'SaleCntL9', 'SaleAmtL9', 'ProfitL9',
         'UserUniCntL9', 'SaleCntL12', 'SaleAmtL12', 'ProfitL12', 'UserUniCntL12', 'recuent_one_sort_in_r12m',
         'saled_montht_in_r12m', 'product_cost', 'volume', 'live_days', 'pl_id', 'is_consumable', 'is_valuable',
         'is_BPP', 'is_breakable', 'express_no', 'over_volume', 'over_weight', 'is_batter', 'is_metal', 'include_disc',
         'wish_forbid', 'yahoo_forbid', 'amazon_forbid', 'is_led', 'goods_shape', 'tool_sharp_product', 'live450',
         'live360', 'live90', 'live30', 'liveLess30']]

catboost_mdl = 1

all_col = list(d1.columns)
num_col = deepcopy(all_col[1:26])
cat_col = deepcopy(all_col[26:])

d1[cat_col] = d1[cat_col].fillna('nan').astype(str)
for col in cat_col:
    d1[col] = LabelEncoder().fit_transform(d1[col]).astype(int)
d1[num_col] = d1[num_col].fillna(0).astype(float)

print(len(d1))

del_pl = [6, 7, 0, 9, 5]
for i in del_pl:
    d1 = d1[d1.pl_id != str(i)]
print(len(d1))

xdata = d1.iloc[:, 1:].values
ydata = d1.iloc[:, 0].values

xdata = xdata.astype(object)
cat_feature = np.arange(len(all_col) - 1)[25:].tolist()
xdata[:, cat_feature] = xdata[:, cat_feature].astype(int).astype(str)
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.33, random_state=42)
# 获取xdata里面的非销售数据部分
X_item_train = X_train[:, 22:]
X_item_X_test = X_test[:, 22:]
x_item_data = xdata[:, 22:]

# 22后面是产品属性,里面前3列是数值product_cost,volume,live_days
item_cat_feature = np.arange(len(list(d1.iloc[:, 1:].columns)[22:]))[3:]

# 全量产品模型
try:
    mdl_all_fea = pickle.load(open('catboost_allV0.pkl', 'rb'))
    print('all_model_loaded!')
except:
    mdl_all_fea = CatBoostClassifier(verbose=False, random_seed=42). \
        fit(X_train, y_train, cat_features=cat_feature)
    pickle.dump(mdl_all_fea, open('catboost_allV0.pkl', 'wb'))

d2 = pd.read_csv('plproduct.csv')
print('CatBoost ALL')
model_show(mdl_all_fea, xdata, ydata, X_test, y_test, d2)

# 全量特征模型
try:
    mdl_item_fea = pickle.load(open('catboost_itemV0.pkl', 'rb'))
    print('item_model_loaded!')
except:
    mdl_item_fea = CatBoostClassifier(verbose=False, random_seed=42). \
        fit(X_item_train, y_train, cat_features=item_cat_feature)
    pickle.dump(mdl_item_fea, open('catboost_itemV0.pkl', 'wb'))

print('CatBoost item')
d2 = pd.read_csv('plproduct.csv')
model_show(mdl_item_fea, x_item_data, ydata, X_item_X_test, y_test, d2)

# 随机丢弃的概率为0.01
# N=20滚20次得到不确定性
dropout_p = 0.01
un_std_N = 20
all_tree_depth = mdl_all_fea.get_all_params()['depth']
drop_all_prob = 1 - (1 - dropout_p) ** all_tree_depth

std_all = np.zeros(un_std_N)

model_all_train_prob = np.mean(mdl_all_fea.predict_proba(X_train)[:, 1])
model_all_test_prob = mdl_all_fea.predict_proba(X_test)[:, 1]
model_item_test_prob = mdl_item_fea.predict_proba(X_item_X_test)[:, 1]

model_all_prob = mdl_all_fea.predict_proba(xdata)[:, 1]


@jit(forceobj=True)
def std_calc(un_std_N, X_train, drop_all_prob):
    for i in range(un_std_N):
        rand_x_train = np.random.rand(len(X_train))
        rand_x_train[rand_x_train < drop_all_prob] = 1
        X_rand_train_data = X_train
        X_rand_train_data[rand_x_train == 1, :22] = 0
        model_drop_prob = mdl_all_fea.predict_proba(X_rand_train_data)[:, 1]
        std_all[i] = np.sqrt(np.sum((model_drop_prob - model_all_train_prob) ** 2 / len(model_drop_prob)))
    std = np.mean(std_all)
    return std


std = std_calc(un_std_N, X_train, drop_all_prob)
# t步梯度更新t_step=2,超参数adj_lambda=0.001,不确定性std*更新梯度iter_grad*超参数adj_lambda=扰动的x值
t_step = 1
adj_lambda = 1
shock = np.zeros((22, X_test.shape[1]), dtype=object)

print('grad_do')


@jit(forceobj=True)
def adj_prob(adj_X_test, t_step, mdl_all_fea, adj_lambda, std):
    for i in range(len(adj_X_test)):
        iter_grad = 0
        for k in range(t_step):
            prob_iter=mdl_all_fea.predict_proba(adj_X_test[i])[1]
            for j in range(22):
                shock[j, :] = adj_X_test[i]
                shock[j, j] += 1
            shock_prob = mdl_all_fea.predict_proba(shock)[:, 1]
            iter_grad = iter_grad + shock_prob - prob_iter
            iter_grad /= norm(iter_grad, 2)
        adj_X_test[i, :22] += iter_grad * std * adj_lambda


    return adj_X_test

# adj_X_test = deepcopy(X_test)
adj_X_test = adj_prob(xdata, t_step, mdl_all_fea, adj_lambda, std)
adj_prob = mdl_all_fea.predict_proba(adj_X_test)[:, 1]
class_adj = deepcopy(adj_prob)
class_adj[class_adj >= 0.5] = 1
class_adj[class_adj < 0.5] = 0
print(f1_score(xdata, class_adj))

# 模型初次预测的结果
d2 = pd.read_csv('plproduct.csv')
d2['all_prob'] = mdl_all_fea.predict_proba(xdata)[:, 1]
d2['item_prob'] = mdl_item_fea.predict_proba(x_item_data)[:, 1]
d2['adj_prob'] = adj_prob

d2 = pd.read_csv('plproduct.csv')
plot_q(d2, 'all', 'all_prob', False)
plot_q(d2[(d2['SaleAmtL12'] == 0) & (d2['live_days'] < 90)], 'live_less_90', 'all_prob', False)

plot_q(d2, 'all', 'item_prob', False)
plot_q(d2[(d2['SaleAmtL12'] == 0) & (d2['live_days'] < 90)], 'live_less_90', 'item_prob', False)

plot_q(d2, 'all', 'adj_prob', False)
plot_q(d2[(d2['SaleAmtL12'] == 0) & (d2['live_days'] < 90)], 'live_less_90', 'adj_prob', False)

d2.to_csv('data_with_2prob.csv', index=False, encoding='ansi')
print(1)
