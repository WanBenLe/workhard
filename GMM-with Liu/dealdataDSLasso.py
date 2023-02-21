'''
    <DS-klevel-SLOPE-iterGMM main code.>
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

import pandas as pd
import numpy as np
from DSLasso import dslasso
from iterGMMest import Iterated_GMM
from linearmodels import IVGMM, IV2SLS
from statsmodels.api import OLS, add_constant
import pickle

d1 = pd.read_csv('stock.csv')
d1['ym'] = np.floor(d1['date'].values / 100)
d2 = pd.read_csv('index.csv')
d2['ym'] = np.floor(d2['public_date'].values / 100)

d3 = pd.merge(d1, d2, how='left', left_on=['PERMNO', 'ym'], right_on=['permno', 'ym'])
listx = ['RETX', 'CAPEI', 'bm',
         'evm', 'pe_op_basic', 'pe_op_dil', 'pe_exi', 'pe_inc', 'ps', 'pcf',
         'dpr', 'npm', 'opmbd', 'opmad', 'gpm', 'ptpm', 'cfm', 'roa', 'roe',
         'roce', 'efftax', 'aftret_eq', 'aftret_invcapx', 'aftret_equity',
         'pretret_noa', 'pretret_earnat', 'GProf', 'equity_invcap',
         'debt_invcap', 'totdebt_invcap', 'capital_ratio', 'int_debt',
         'int_totdebt', 'cash_lt', 'invt_act', 'rect_act', 'debt_at',
         'debt_ebitda', 'short_debt', 'curr_debt', 'lt_debt', 'profit_lct',
         'ocf_lct', 'cash_debt', 'fcf_ocf', 'lt_ppent', 'dltt_be', 'debt_assets',
         'debt_capital', 'de_ratio', 'intcov', 'intcov_ratio', 'cash_ratio',
         'quick_ratio', 'curr_ratio', 'cash_conversion', 'at_turn',
         'rect_turn', 'sale_invcap', 'sale_equity', 'sale_nwc', 'accrual',
         'ptb', 'PEG_trailing', 'divyield%', 'PEG_1yrforward', 'PEG_ltgforward']
list1 = ['permno', 'ym']
list1.extend(listx)
d3 = d3[list1]
temp = d3['RETX'].values.copy()
temp[temp == 'B'] = 0
temp[temp == 'C'] = 0
temp[temp == 'nan'] = 0
d3['RETX'] = temp
for i in listx:
    d3[i] = d3[i].astype(float)

d3 = d3.fillna(0).drop_duplicates(subset=['permno', 'ym'])
# print(d3.columns)
d3.to_excel('look.xlsx')
date_un = np.unique(d3['ym'].values)[:-12]
fit_data = np.zeros((len(date_un), len(listx)))
for i in range(len(date_un)):
    # 去掉代码和年月,剩下的是收益率,后面的是指标
    temp = d3[d3['ym'] == date_un[i]].values[:, 2:].copy()
    # 收益率和对各个因子的协方差
    retx_temp = np.mean(temp[:, 0].copy())
    coef = np.cov(np.transpose(temp))[0][1:]
    fit_data[i, 0] = retx_temp
    fit_data[i, 1:] = coef
print('有数据的日期', date_un)
fit_data[np.isnan(fit_data)] = 0
# Set_1, Set_2, Set_all, st, sr = DSSlope(fit_data[:, 1:], fit_data[:, 0].reshape(-1, 1))

Set_1, Set_2, Set_all, st, sr = dslasso(fit_data[:, 1:], fit_data[:, 0].reshape(-1, 1))

print(sum(Set_1))
print(sum(Set_2))

# 这里是新数据....
d1 = d3.values
lookup = d1[:, 0:2]
date_un = np.unique(d1[:, 1])
no_un = np.unique(d1[:, 0])
fill_data = np.zeros((20000, 69))
fillx = 0

for i in no_un:
    t1 = np.argwhere(i == lookup[:, 0])
    for j in date_un:
        t2 = np.argwhere(j == lookup[:, 1])
        if len(np.intersect1d(t1, t2)) == 0:
            fill_data[fillx, 0] = i
            fill_data[fillx, 1] = j
            fillx += 1
fill_data = fill_data[fill_data[:, 1] != 0]

print(len(no_un) * len(date_un))
d1 = np.vstack((d1, fill_data))

set1 = np.argwhere(Set_1 == True)[:, 0]
set2 = np.argwhere(sr > 0)
set2 = np.setdiff1d(set2.reshape(-1), set1)
data = d1[:, 3:]
date = d1[:, 1]
cluser = d1[:, 0]
x = data[:, set1]
z = data[:, set2]
# pd.DataFrame(np.corrcoef(x.transpose())).to_csv('corrx.csv', index=False)
# pd.DataFrame(np.corrcoef(z.transpose())).to_csv('corrz.csv', index=False)

y = d1[:, 2].reshape(-1, 1)

# Iterated_GMM(y * 100, x, z, cluser)
X = add_constant(np.hstack((x, z)))
ols = OLS(X, y).fit()
print(ols.summary())
print(1)
