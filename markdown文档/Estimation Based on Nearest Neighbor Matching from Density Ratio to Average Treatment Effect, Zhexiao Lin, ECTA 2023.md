#### Estimation Based on Nearest Neighbor Matching: from Density Ratio to Average Treatment Effect, Zhexiao Lin, ECTA 2023

匹配方法通过最小化观察到的协变量的组差异来平衡不同组的观察结果,其中最近邻(NN)匹配将每个治疗(对照)个体分配给与其距离最小的 M 个控制(治疗)个体.比率匹配-确定匹配数M(与非参数统计中的偏差-方差权衡)和对 NN 匹配估计器进行大样本统计推断是两个困难的问题.Imbens 2006的工作固定了M,但就算用了偏差校正可以缓解固定M的M-NN匹配ATE估计器渐进偏差的问题,但仍然是低效的.

本文通过将测量x的匹配次数的KM(x)转为密度比估计,并以此桥接偏差校正匹配估计/双鲁棒估计/双机学习估计.

##### 基于NN最近邻匹配的误差修正密度比估计器

$\hat{\gamma}_M^{bc}=\hat{\gamma}^{reg}+\frac{1}{n}[\sum_{i=1,D=1}^n(1+\frac{K_M(i)}{M})\hat{R_i}-\sum_{i=1,D=0}^n(1+\frac{K_M(i)}{M})\hat{R_i}]$

NN匹配$\forall x,z\in R,M \in [N_0]$

1.$X_{m}(\textbullet):R^d \rightarrow\{X_i\}_{i=1}^{N_0}$是满足以下条件的输入z的返回值映射

$\sum_{i=1}^{N_0}1(||X_i-z||\le||x-z||)=M$

2.$K_m(\textbullet):R^d\rightarrow \{0\} \bigcup[N_1]$是满足以下条件的x的匹配次数的返回值映射

$K_M(x)=K_M(x;\{X_i\}_{i=1}^{N_0},\{Z_j\}_{j=1}^{N_1})=\sum_{j=1}^{N_1}1(||x-Z_j||\le||X_{(M)}(Z_j)-Z_j||)$

3.$A_m(\textbullet):R^d\rightarrow B(R^d)$是满足以下条件的$R^d$到$R^d$中Borel集的对应映射

$A_M(x)=A_M(x;\{X_i\}_{i=1}^{N_0})=\{z \in R^d:||x-z||\le||X_{(M)}(z)-z||\}$

3得到了2下的catchment area$\forall x\in R^d,K_M(x)=\sum_{j=1}^{N_1}1(Z_j\in A_M(x))$

基于NN匹配的密度比估计器

$\forall x \in R^d, \hat{r}_M(x)=\hat{r}_M(x;\{X_i\}_{i=1}^{N_0},\{Z_j\}_{j=1}^{N_1})=\frac{N_0}{N_1}\frac{K_M(x)}{M}$

$L_p$风险一致性:$lim_{N_0\rightarrow \infty}E[\int_{R^d}|\hat{r}_M(x)-r(x)|^{p}f_0(x)dx]=0$

基于不同的case用下面2个算法可以得到时间复杂度更优的有效估计

##### 算法1:样本点的密度比估计器,时间复杂度为$O((d+N_1M/N_0)N_0logN_0)$

输入:$\{X_i\}_{i=1}^{N_0},\{Z_j\}_{j=1}^{N_1},M$,输出:$\{\hat{r_M}(X_i)\}_{i=1}^{N_0}$

基于$\{X_i\}_{i=1}^{N_0}$建立KD树

for j=1:$N_1$

​	基于KD树寻找$Z_j$的M-NN并存为$S_j$

统计$[N_0]$每个元素在$\cup^{N_1}_{j=1}S_j$出现的次数记为$\{K_M(X_i)\}_{i=1}^{N_0}$并基于定义计算出$\{\hat{r_M}(X_i)\}_{i=1}^{N_0}$

##### 算法2:包含观测值和新数据的密度比估计器,当新点i.i.d且和X概率度量和样本点独立时和时间复杂度为$O((d+N_1M/N_0)(N_0+n)log(N_0+n))$

输入:$\{X_i\}_{i=1}^{N_0},\{Z_j\}_{j=1}^{N_1},M,新点\{x_i\}_{i=1}^n$,输出:$\{\hat{r_M}(X_i)\}_{i=1}^{N_0},\$\{\hat{r_M}(x_i)\}_{i=1}^{N_0}$

基于:$\{X_i\}_{i=1}^{N_0} \cup \{x_i\}_{i=1}^n$建立KD树

for j=1:$N_1$

​	空集$S_j,S_j^\prime,m=1$

​	while $|S_j| < M$

​		基于KD树寻找$ Z_j \in \{X_i\}_{i=1}^{N_0} \cup \{x_i\}_{i=1}^n$的m-NN记为tmp

​		if  (tmp in $\{X_i\}_{i=1}^{N_0} )$:{index存入$S_j$} else {index存入$S_j^\prime$}

​	m+=1

统计$[N_0]$每个元素在$\cup^{N_1}_{j=1}S_j$和$\cup^{N_1}_{j=1}S_j^\prime$出现的次数记为$\{K_M(X_i)\}_{i=1}^{N_0},\{K_M(x_i)\}_{i=1}^{n}$并基于定义计算出$\{\hat{r_M}(X_i)\}_{i=1}^{N_0},\{\hat{r_M}(x_i)\}_{i=1}^{n}$

##### 统计性质分析

$[(X_i,D_i,Y_i)]^n_{i=1}$是$(X,D,Y)$的n个独立副本,治疗变量$D\in \{0,1\}$协变量X结果变量Y绝对连续密度$f_x,n_0=\sum_{i=1}^n(1-D_i),n_1=\sum_{i=1}^nD_i$

最近邻匹配估计

$\hat{r}_M=n^{-1}\sum_{i=1}^n[\hat{Y_i}(1)-\hat{Y_i}(0)]=\frac{1}{n}[\sum_{i=1,D_i=1}^n(1+\frac{K_M(i)}{M})Y_i-\sum_{i=1,D_i=0}^n(1+\frac{K_M(i)}{M})Y_i]$

结果回归版本:$\hat{r}^{reg}=n^{-1}\sum_{i=1}^n[\hat{\mu}_1(X_i)-\hat{\mu}_0(X_i)]$

d>1的误差修正版本:$\hat{r}^{bc}_{M}=\hat{r}^{reg}+\frac{1}{n}[\sum_{i=1,D_i=1}^n(1+\frac{K_M(i)}{M})\hat{R_i}-\sum_{i=1,D_i=0}^n(1+\frac{K_M(i)}{M})\hat{R_i}]$

倾向得分通用估计量$\hat{e}(x):R^d\rightarrow R,e(x)=P(D=1|X=x)$

$\hat{r}^{dr}_{M}=\hat{r}^{reg}+\frac{1}{n}[\sum_{i=1,D_i=1}^n\frac{1}{\hat{e}(X_i)}\hat{R_i}-\sum_{i=1,D_i=0}^n\frac{1}{1-\hat{e}(X_i)}\hat{R_i}]$

发散M的渐进分析

给定三个基本假设:1.无混杂性和重叠假设-即强可忽略性条件2.允许结果模型的错误指定3.结果模型一致估计和附加假设

1.$\forall \ x \in X,\omega \in \{0,1\} ,E[U_{\omega}^2|X=x]$一致有界且远离0

2.$\exist \kappa>0,\forall x\in X,\omega\in\{0,1\},E[|U_{\omega}|^{2+k}|X=x]$一致有界

3.$max_{t\in \Lambda_{floor(d/2)+1}}||\part^t\mu_\omega||_{\infty}$有界,$\Lambda_k$是正整数k中满足$\sum_{j=1}^dt_j=k$的d维向量集合

4.近似精度假设$\forall \omega \in \{0,1\} ,l \in [floor(d/2)],\gamma_l>0.5-l/d,l=1,2,...,floor(d/2)\\max_{t\in \Lambda_{floor(d/2)+1}}||\part^t\mu_\omega||_{\infty}=O_p(1)\\max_{t\in \Lambda_{floor(d/2)+1}}||\part^t\hat{\mu}_\omega-\part^t\mu_\omega||_{\infty}=O_p(n^{-\gamma_l})$

对于ATE的半参效率下界有$\sigma^2=E[\mu_1(X)-\mu_0(X)+\frac{D(Y-\mu_1(X))}{e(x)}-\frac{(1-D)(Y-\mu_0(X))}{1-e(x)}-\gamma]^2$

对于$\hat{\gamma}^{bc}_{M}$有$\hat{\sigma}^2=\frac{1}{n}\sum_{i=1}[\hat{\mu}_1(X)-\hat{\mu}_0(X)+(2D_i-1)(1+\frac{K_M(i)}{M})\hat{R_i}-\hat{\gamma}^{bc}_{M}]$

$\hat{\gamma}^{bc}_{M}$的双重稳健性:对于满足假设的$(X,D,Y).(P_{X|D=0},P_{X|D=1}).(P_{X|D=1},P_{X|D=0})$和$Mlog n/n \rightarrow 0,M\rightarrow \infty,n\rightarrow \infty$有:$ \hat{\gamma}^{bc}_{M} - \gamma\stackrel{p}{\rightarrow}0$

$\hat{\gamma}^{bc}_{M}$的半参有效性:对于满足假设的$(X,D,Y).(P_{X|D=0},P_{X|D=1}).(P_{X|D=1},P_{X|D=0})$

定义$\gamma=\{min_{l\in [floor(d/2)]}[1-(0.5-\gamma_l)\frac{d}{l}]\}\wedge[1-\frac{d}{2(floor(d/2)+1)}]$,$M/n^\gamma \rightarrow 0,M\rightarrow \infty,n\rightarrow \infty$有:$ \sqrt{n}(\hat{\gamma}^{bc}_{M} - \gamma)\stackrel{d}{\rightarrow}N(0,\sigma^2),\hat{\sigma}^2\stackrel{p}{\rightarrow}\sigma^2$

若结果模型平滑,即使固定M由于matching差异收敛仍能保持一致性,且推论可得d=1的情况下$\hat{\gamma}_M$是半参有效的.

##### 基于NN最近邻匹配的误差修正密度比估计器的双机学习版本(方差估计相同,附加1. M随n多项式增长2. Lipschitz 型条件)

$n\%K=0,[I_k]^K_{k=1}$是$[n]$的随机split,样本数$n'=n/K,k \in [K],\omega \in \{0,1\},\hat{\mu}_{\omega,k}(\textbullet)$使用了$[(X_j.D_j,Y_j)]_{j=1,j\notin I_k}^n,K_{M,k}(i)$是匹配次数

$\tilde{\gamma}^{bc}_{M,k}=K^{-1}\sum_{k=1}^{K}\check{\gamma}^{bc}_{M,k}$

$\check{\gamma}^{bc}_{M,k}=n^{'-1}\sum_{i=1}^n[\hat{\mu}_{1,k}(X_i)-\hat{\mu}_{0,k}(X_i)]+\frac{1}{n'}[\sum_{i=1,i\in I_k,D=1}^n(1+\frac{K_{M,k}(i)}{M})(\hat{Y_i}-\hat{\mu}_{1,k}(X_i))-\sum_{i=1,i\in I_k,D=0}^n(1+\frac{K_{M,k}(i)}{M})(\hat{Y_i}-\hat{\mu}_{0,k}(X_i))]$

修改额外假设3.4为以下假设

1.$\omega \in \{0,1\},E[U^2_\omega]$有界且不为0

2.$\exist \kappa>0,E[|Y|^{2+k}]$有界

3.无穷范数近似精度$\forall \omega \in \{0,1\},||\hat{\mu}_\omega-\mu_\omega||_{\infty}=O_p(n^{-d/(4+2d)})$

$\tilde{\gamma}^{bc}_{M,k}$的双重稳健性:对于满足假设的$(X,D,Y).(P_{X|D=0},P_{X|D=1}).(P_{X|D=1},P_{X|D=0})$和$Mlog n/n \rightarrow 0,M\rightarrow \infty,n\rightarrow \infty$有:$ \tilde{\gamma}^{bc}_{M,k} - \gamma\stackrel{p}{\rightarrow}0$

$\tilde{\gamma}^{bc}_{M,k}$的半参有效性:对于满足假设的$(X,D,Y).(P_{X|D=0},P_{X|D=1}).(P_{X|D=1},P_{X|D=0})$

选取$M=\alpha n^{2/(2+d)},\alpha>0$有$ \sqrt{n}(\tilde{\gamma}^{bc}_{M,k} - \gamma)\stackrel{d}{\rightarrow}N(0,\sigma^2),\hat{\sigma}^2\stackrel{p}{\rightarrow}\sigma^2$

d的取值建议等于连续变量个数,$\alpha$建议取1(理论最优没给,玄学起来了)
