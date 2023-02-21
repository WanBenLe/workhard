# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import os
import codecs
import chardet
import sys


def Numeric_Var(data, VarNames):
    if len(VarNames) > 0:
        data1 = data[VarNames]
        data2 = data1.values
        Numeric_Var_Dec = np.empty((22, len(VarNames)), dtype=np.object)
        for i in range(len(VarNames)):
            print(VarNames[i])
            temp_data = data2[:, i]
            try:
                temp_data = temp_data.astype(float)
            except:
                print(np.unique(temp_data))
                print(1)
            Numeric_Var_Dec[0, i] = VarNames[i]
            Numeric_Var_Dec[1, i] = len(temp_data)
            Numeric_Var_Dec[2, i] = np.sum(np.isnan(temp_data))
            temp_data = temp_data[~np.isnan(temp_data)]
            Numeric_Var_Dec[3, i] = np.sum(temp_data < 0)
            Numeric_Var_Dec[4, i] = np.sum(temp_data == 0)
            Numeric_Var_Dec[5, i] = np.sum(temp_data > 0)
            Numeric_Var_Dec[6, i] = Numeric_Var_Dec[2, i] / Numeric_Var_Dec[1, i]
            Numeric_Var_Dec[7, i] = Numeric_Var_Dec[3, i] / Numeric_Var_Dec[1, i]
            Numeric_Var_Dec[8, i] = Numeric_Var_Dec[4, i] / Numeric_Var_Dec[1, i]

            Numeric_Var_Dec[9, i] = np.min(temp_data)
            Numeric_Var_Dec[10, i] = np.max(temp_data)
            Numeric_Var_Dec[11, i] = np.mean(temp_data)
            Numeric_Var_Dec[12, i] = np.std(temp_data)
            Numeric_Var_Dec[13, i] = np.percentile(temp_data, 1)
            Numeric_Var_Dec[14, i] = np.percentile(temp_data, 5)
            Numeric_Var_Dec[15, i] = np.percentile(temp_data, 10)
            Numeric_Var_Dec[16, i] = np.percentile(temp_data, 25)
            Numeric_Var_Dec[17, i] = np.percentile(temp_data, 50)
            Numeric_Var_Dec[18, i] = np.percentile(temp_data, 75)
            Numeric_Var_Dec[19, i] = np.percentile(temp_data, 90)
            Numeric_Var_Dec[20, i] = np.percentile(temp_data, 95)
            Numeric_Var_Dec[21, i] = np.percentile(temp_data, 99)
        stat_name = ['VarName', 'Obs', 'Missobs', 'NegaObs', ' ZeroObs', 'PosiObs', 'MissRatio', 'NegaRatio',
                     'ZeroRatio',
                     'MinValue', 'MaxValue', 'MeanValue', 'StdDev', 'P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90',
                     'P95', 'P99']
        print(Numeric_Var_Dec)
        View_data = pd.DataFrame(Numeric_Var_Dec.T, columns=stat_name)
        return View_data


def Str_Var(data, VarNames):
    if len(VarNames) > 0:
        data1 = data[VarNames]
        data2 = data1.values
        for i in range(len(VarNames)):
            print(VarNames[i])
            temp_data = data2[:, i]
            class_dum = np.unique(temp_data)
            len_class = len(temp_data)
            for j in range(len(class_dum)):
                class_obs = np.sum(temp_data == class_dum[j])
                class_ratio = class_obs / len_class
                if i == 0 & j == 0:
                    Numeric_Var_Dec = np.array([VarNames[i], class_dum[j], class_obs, class_ratio])
                else:
                    temp = np.array([VarNames[i], class_dum[j], class_obs, class_ratio])
                    Numeric_Var_Dec = np.vstack((Numeric_Var_Dec, temp))
    View_data = pd.DataFrame(Numeric_Var_Dec, columns=['VarName', 'ClassName', 'ClassObs', 'ClassRatio'])
    return View_data


def RunBase(data, VarNames, kind='Str'):
    VarNames=np.array(VarNames)
    if kind.lower() == 'str':
        View_data = Str_Var(data, VarNames)
    else:
        View_data = Numeric_Var(data, VarNames)
    return View_data



# WOE(路径,变量,类型,目标变量.编码(可以尝试'auto'))
def WOE(data, varList, type0='Con', target_id='y'):

    '''
    # 对分类变量直接进行分组统计并进行WOE、IV值 计算
    # 对连续型变量进行分组（default:10）后进行WOE、IV值 计算
    # 最终返回分组统计结果DataFrame格式，并生成xlsx文件
    '''

    for var in varList:
        print(var)

        if type0.upper() == "CON".upper():
            data[var] = data[var].fillna(-999999)
            df, retbins = pd.qcut(data[var], q=10, retbins=True, duplicates="drop")

            stat = np.zeros((len(retbins) - 1, 3)).astype(object)
            for i in range(len(retbins) - 1):
                stat[i, :] = np.array([retbins[i], '-', retbins[i + 1]])
        elif type0.upper() == "CAT".upper():
            print(data[var])
            data[var] = data[var].fillna(-999999)
            df = data[var]
            print(df.values)
            temp = np.unique(df.values)

            stat = np.zeros((temp.shape[0], 3)).astype(object)
            stat[:, 0] = temp
            stat[:, 2] = temp

        else:
            print('ERROR!!!')
        try:
            res = pd.crosstab(df, data[target_id], margins=True)
        except:
            print(1)
        bendata = res.values.astype(float)
        for i in range(bendata.shape[1]):
            bendata[:, i] = bendata[:, i] / bendata[-1, i]

        # if type0.upper() == "CON".upper():

        cool1 = res['All'].values.reshape(-1, 1)[0:-1, :]

        stat = np.hstack((stat, cool1))
        temp = res['All'].values.reshape(-1, 1)
        tempser = temp[0:-1, 0] / temp[-1, 0]

        stat = np.hstack((stat, tempser.reshape(-1, 1), res.values[0:-1, 2].reshape(-1, 1),
                          bendata[0:-1, 1].reshape(-1, 1), res.values[0:-1, 1].reshape(-1, 1),
                          bendata[0:-1, 0].reshape(-1, 1)))
        tempser1 = (bendata[0:-1, 1] - bendata[0:-1, 0]).reshape(-1, 1)
        stat = np.hstack((stat, tempser1, tempser1))
        a = bendata[0:-1, 1]
        b = bendata[0:-1, 0]
        c = a / b
        c[c == 0] = 1
        tempser2 = np.log(c).reshape(-1, 1)
        stat = np.hstack((stat, tempser2))
        stat = np.hstack((stat, tempser2 * tempser1))
        tempser1 = np.array([0, '', 0, np.sum(stat[:, 3]), 0, np.sum(stat[:, 5]), 0, np.sum(stat[:, 7]), 0, 0, 0, 0,
                             np.sum(stat[:, 12])])
        stat = np.vstack((stat, tempser1))
        tempser1 = np.zeros((1, 13)).astype(object)
        tempser1[:, :] = ''
        tempser1[0, 1] = var
        stat = np.vstack((stat, tempser1))
        stat = pd.DataFrame(stat, columns=['rank_start', 'null_var', '	rank_end', 'total_num', 'total_percent',
                                           'good_num', 'good_percent', 'bad_num', 'bad_percent', 'GBGOOD1', 'GBGOOD2',
                                           'WOE', 'IV'])
        stat.to_csv(str(var) + '_woe.csv', index=False)
        print(str(var) + '_woe.csv write')

    return stat


def applyWOE(X_data, X_map, var_list, id_cols_list=None, flag_y=None):
    if flag_y:
        bin_df = X_data[[flag_y]]
    else:
        bin_df = pd.DataFrame(X_data.index)
    for var in var_list:
        x = X_data[var]
        bin_map = X_map[X_map['name'] == var]
        bin_res = np.array([0] * x.shape[-1], dtype=float)
        for i in bin_map.index:
            upper = bin_map['high'][i]
            lower = bin_map['low'][i]
            if lower == upper:
                # print var,'==============',lower
                x1 = x[np.where(x == lower)[0]]
            else:
                # print var, '<<<<<<<<<<<<<<<<<<<<',lower, upper
                if i == bin_map.index.min():
                    x1 = x[np.where((x <= upper))[0]]  # 会去筛选矩阵里面符合条件的值
                elif i == bin_map.index.max():
                    x1 = x[np.where((x > lower))[0]]  # 会去筛选矩阵里面符合条件的值
                else:
                    x1 = x[np.where((x > lower) & (x <= upper))[0]]  # 会去筛选矩阵里面符合条件的值
            # mask = np.in1d(x, x1)  # 用于测试一个数组中的值在另一个数组中的成员资格,返回布尔型数组
            mask = np.in1d(x, x1)  # 用于测试一个数组中的值在另一个数组中的成员资格,返回布尔型数组
            bin_res[mask] = bin_map['WOE'][i]  # 将Ture的数据替换掉
        bin_res = pd.Series(bin_res, index=x.index)
        bin_res.name = x.name
        bin_df = pd.merge(bin_df, pd.DataFrame(bin_res), left_index=True, right_index=True)
    if id_cols_list:
        bin_df = pd.merge(bin_df, X_data[id_cols_list], left_index=True, right_index=True)
    return bin_df
