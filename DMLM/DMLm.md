Causal Mediation Analysis with Double Machine Learning, Helmut Farbmacher, Econometrics Journal, 2022

$Y结果.D处理.M潜在中介变量,X协变量$

![image-20221114170444737](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20221114170444737.png)

例如说D是涨价,M是刷价成功/失败,X是协变量(价格.毛利率.成本.销售数据之类的),Y销量利润

$ATE=E[Y(1,M(1))-Y(0,M(0))]$

$\theta(d)=E[Y(1,M(d))-Y(0,M(d))]纯粹的直接效应$

$\sigma(d)=E[Y(d,M(1))-Y(d,M(0))]纯粹的间接效应$

$\gamma(m)=E[Y(1,m))-Y(0,m)]受控直接效应$



反事实识别$E[Y(d,M(1-d))]=E[\psi^{*}_{d}]$

有效得分函数

$\psi^{*}_{d}=\frac{I\{D=d\}(1-P_{d}(M,X))}{P_{d}(M,X)(1-P_{d}(X))}[Y-\mu(d,M,X)]\\+\frac{I\{D=1-d\}}{1-P_{d}(X)}[\mu(d,M,X)-E[\mu(d,M,X)|D=1-d,X]]\\+E[\mu(d,M,X)|D=1-d,X]$

1.将$\mathcal{W}$拆成$K$份子样本$k$,后者样本量为$n_{k}$,$\mathcal{W}_k$是观测值,$\mathcal{W}_k^{C}$是前者补集

2.对每份子样本$k$用$\mathcal{W}_k^{C}$估计$p_{d}(X)$和$p_{d}(M,X)$,将$\mathcal{W}_k^{C}$分成2份不重叠子样本,用第一个子样本估计条件均值$\mu(d,M,X)$并用另一个子样本估计$\omega(1-d,X)=E[\mu(d,M,X)|D=1-d,X]$,进而预测$\mathcal{W}_k$的扰动参数,称为$\hat{P_{d}^{k}}(X),\hat{P_{d}^{k}}(M,X),\hat{\mu}^{k}(D,M,X),\hat{\omega}(D,X)^{k}$

3.于是可以算出来$\mathcal{W}_k$中每个样本$i$的有效得分函数

$\hat{\psi_{d,i}^{*k}}=\frac{I\{D_{i}=d\}(1-\hat{p}^{k}_{d}(M_{i},X_{i}))}{\hat{p}^{k}_{d}(M_{i},X_{i})(1-\hat{p}^{k}_{d}(X_{i}))}[Y-\hat{\mu}^{k}(d,M_{i},X_{i})]\\+\frac{I\{D_{i}=1-d\}}{1-\hat{p}^{k}_{d}(X_{i})}[\hat{\mu}^{k}(d,M_{i},X_{i})-\hat{\omega}(1-d,X_{i})^{k}]\\+\hat{\omega}(1-d,X_{i})^{k}$

4.对所有$K$个子样本中的所有$\hat{\psi_{d,i}^{*k}}$求平均，以获得总样本中$\psi_{d0}=E[Y(d,M(1-d))]$的估计值，用$\hat{\psi_{d}^{*}}=\frac{1}{n}\sum^{K}_{k=1}\sum^{n_{k}}_{i=1}\hat{\psi_{d,i}^{*k}}$表示

另外一摞假设下

$n^{1/2}(\hat{\psi_{d}^{*}}-\hat{\psi_{d0}^{*}})\rightarrow N(0,\sigma^{2}_{\psi_{d}^{*}})$

其中$\sigma^{2}_{\psi_{d}^{*}}=E[(\hat{\psi_{d}^{*}}-\hat{\psi_{d0}^{*}})^{2}]$

