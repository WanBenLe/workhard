import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as st
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

'''
双样本均值差异显著性检验(AB测试)
双样本均值差异显著性检验是用于检验两个样本均值是否在统计学上有显著差异的检验方法.
在实际应用中通常用于比较方案实施前后的目标指标效果是否发生明显变化,进而推断新方案的优劣性.

一个完整的双样本均值差异显著性检验步骤如下:

1.先做Levene's 方差齐性检验.
Levene's 方差齐性检验的原假设为双样本方差相等,备择假设为双样本方差不等.
对于同方差的双样本,可以直接做双样本均值显著性t检验,但若方差不等,需要做带自由度调整的的Welch t检验.
Levene's 方差齐性检验步骤如下:
a. 对双样本分别计算均值,然后样本每个值减去均值,然后取绝对值.
b. 构建服从为k-1和N-k的F分布的W统计检验量,并进行假设检验.

2.进行双样本均值差异显著性检验,此处分两种情况：
a.无法拒绝Levene's 方差齐性检验的原假设：使用普通的双样本均值差异显著性检验.
b.拒绝Levene's 方差齐性检验的原假设：使用带自由度调整的Welsh t检验.
两种假设检验的原假设均为双样本均值相等,备择假设为双样本均值不等. 

3.计算出均值差异95%置信区间和Cohen’s D效应量,后者用于测量两个均值之间差异的效应大小.
4.最后我们给出了基于小样本对Cohen's D效应量进行调整Hedges's g效应量
{BEN:公式块在这}

首先我们给出了双样本的KDE核密度估计分布图和描述性的统计,后者包括了双样本的样本量,均值,标准差,最小值,25%分位数,中位数,75%分位数和最大值.
{图}
{描述性统计表}
其次我们进行了Levene's 方差齐性检验
'''

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

d2 = pd.read_excel('abtest.xlsx', sheet_name='Sheet2')[['对照组', '测试组']]

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10, 5)
sns.kdeplot(data=d2['对照组'].values, legend=False, ax=axes[0])
sns.kdeplot(data=d2['测试组'].values, legend=False, ax=axes[1])

axes[0].set(title='对照组数据集分布')
axes[1].set(title='测试组数据集分布')
plt.subplots_adjust(wspace=0.2)
plt.show()
print(d2.describe())

aaaa = d2['对照组'][pd.notna(d2['对照组'])]
bbbb = d2['测试组'][pd.notna(d2['测试组'])]

W, levene_P = stats.levene(aaaa, bbbb, center='median')
levene_P = np.round(levene_P, 4)
x1 = "Levene W的值为:" + str(np.round(W, 4)) + ',p值为' + str(levene_P)

if levene_P > 0.1:
    x2 = '.由于p值大于0.1,无法拒绝原假设,双样本方差相等.'
elif levene_P > 0.05:
    x2 = '.p值在0.1的显著性水平下拒绝了原假设,双样本方差不相等.'
elif levene_P > 0.01:
    x2 = '.p值在0.05的显著性水平下拒绝了原假设,双样本方差不相等.'
elif levene_P >= 0.00:
    x2 = '.p值在0.01的显著性水平下拒绝了原假设,双样本方差不相等.'

if levene_P > 0.1:
    x3 = '由于Levene\'s 方差齐性检验无法拒绝原假设,均值差异显著性检验使用标准的学生t检验.'
    t, p_two, df = st.ttest_ind(bbbb, aaaa)
else:
    x3 = '由于Levene\'s 方差齐性检验拒绝原假设,均值差异显著性检验使用Welsh t检验.'
    t, p_two, df = st.ttest_ind(bbbb, aaaa, usevar='unequal')
x4 = '均值差异显著性t的统计检验量为' + str(np.round(t, 4)) + ', p值为' + str(np.round(p_two, 4)) + ',自由度为:' + str(
    np.round(df, 4))

if p_two > 0.1:
    x5 = '.由于p值大于0.1,无法拒绝原假设,双样本均值相等.'
elif p_two > 0.05:
    x5 = '.p值在0.1的显著性水平下拒绝了原假设,双样本均值不相等.'
elif p_two > 0.01:
    x5 = '.p值在0.05的显著性水平下拒绝了原假设,双样本均值不相等.'
elif p_two > 0.00:
    x5 = '.p值在0.01的显著性水平下拒绝了原假设,双样本均值不相等.'

# A组均值与标准差
n1_mean = aaaa.mean()
n1_std = aaaa.std()

# B组均值与标准差
n2_mean = bbbb.mean()
n2_std = bbbb.std()

t_cv = 1.667
n1 = len(aaaa)
n2 = len(bbbb)

# 合并方差
pooled_s2 = ((n1 - 1) * np.square(n1_std) +
             (n2 - 1) * np.square(n2_std)) / (n1 + n2 - 2)

# 标准误差值
se_diff = np.sqrt(pooled_s2 * (1 / n1 + 1 / n2))

# 置信区间
x = np.round(n2_mean - n1_mean, 4)
a = np.round((n2_mean - n1_mean) - t_cv * se_diff, 4)
b = np.round((n2_mean - n1_mean) + t_cv * se_diff, 4)

x6 = '均值差异的95%%置信区间和差异均值为:(%f,%f,%f)' % (a, x, b)

# 合并标准差计算
pooled_std = np.sqrt(pooled_s2)

d = np.abs(np.round((n1_mean - n2_mean) / pooled_std, 4))
x7 = ',Cohen\'s D效应量为' + str(d)
g = np.round(d * (1 - 3 / (4 * (n1 + n2 - 2) - 1)), 4)
x8 = ',Hedges\'s g效应量为' + str(g)
z = (n2_mean - n1_mean) / pooled_std * np.sqrt(1 / n1 + 1 / n2)
power = 1 - stats.norm.cdf(z - stats.norm.ppf(1 - float(p_two) / 2)) - \
        stats.norm.cdf(-z - stats.norm.ppf(1 - float(p_two) / 2))

x9 = 'power为' + str(np.round(power, 4))
text_all = x1 + '\n' + x2 + '\n' + x3 + '\n' + x4 + '\n' + x5 + '\n' + x6 + '\n' \
           + x7 + '\n' + x8 + '\n' + x9

text_all = x1 + x2 + x3 + '\n' + x4 + x5 + x9 + '.\n' + x6 + x7 + x8
print(text_all)
print(1)
