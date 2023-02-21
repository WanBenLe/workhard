'''
    <iterGMM est code.>
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
DS-SLOPE-iterGMM
该模型是万善乐与刘鹏StormstoutLau的合作

https://github.com/WanBenLe

https://github.com/StormstoutLau

This model was proposed by Ben Wan and Peng Liu together

该代码执行AGPL 3.0协议.

This code is use the GNU AFFERO GENERAL PUBLIC LICENSE Version 3.

请为任何学术目的进行必要的引用.

Please cite as necessary for any academic purposes.

DS-Post-Lasso的工作:

1.对y和高维变量x进行第一阶段估计,得到非0的变量集S1(Lasso筛选).

2.对于S1的每个变量作为因变量,使用其余变量进行回归得到非0变量集S2(与非0变量相关可能被遗漏的变量).

3.使用OLS对S1和S2的并集进行回归得到(在满足高斯-马尔科夫假设的前提下)BLUE估计

我们对于DS-Post-Lasso进行改进并提出DS-klevelSLOPE-iterGMM,以解决以下方面的问题:

1.Lasso对于FDR和第一类错误的冲突和使得第二类错误表现不佳的问题

2.OLS无法解决时序相关.内生性.集群性以及错误识别的问题

SLOPE通过对不同大小的系数给予不同的惩罚系数,改善第二类错误的同时也在某种程度上控制了第一类错误,与此同时,由于SLOPE罚函数为凸函数的优良性质使得求解成本与Lasso近似,在参数优化上,我们使用了~的K-level优化方法.

对于同时存在多重计量问题的高维股票数据,我们使用了~的错误识别迭代GMM模型,该模型在面对轻微至中等的错误识别中均有着较优的结果,进一步的,我们讨论了模型中的工具变量的具体情况(待完善).

DS-klevelSLOPE-iterGMM的具体步骤为:

1.对y和高维变量x进行第一阶段估计,得到非0的变量集S1(3-level SLOPE筛选).

2.对于S1的每个变量作为因变量,使用其余变量进行回归得到非0变量集S2(与非0变量相关可能被遗漏的变量,2-level SLOPE筛选).

3.使用Y和S1的变量进行iterGMM回归,并使用S2的变量作为工具变量.

dealdata.py:运行主代码(带SDF)

dealdata_DS-Lasso.py:对比用的DS-Post-Lasso(带SDF)

slopeklevel.py承载了参数优化和主回归的工作

slopeest.py实现了基于numba优化的近似算子参数估计

iterGMMest.py实现了迭代GMM相关的估计工作