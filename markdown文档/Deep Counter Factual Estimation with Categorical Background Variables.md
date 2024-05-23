# Deep Counter Factual Estimation with Categorical Background Variables
提出反事实查询预测Counter Factual Query Prediction (CFQP)，当背景变量是分类变量时从连续观察中推断反事实的新颖方法

假设处理和可观察量是连续的，并且对处理响应贡献最大的隐藏变量是分类的，我们可以依靠混合分布的可识别性的DL近似恢复反事实

因果模型$M=<U,V,F>,U=\{U_x,U_T,U_{\epsilon},W\}$是背景隐外生变量集.$V=\{X,T,Y\}$是可观测变量,强可忽略假设下有

$\begin{flalign}&p(Y_{t'}=y'|X=x,Y=y,T=t)\\&=\frac{p(Y_{t'}=y',X=x,T=t,Y=y)}{p(X=x,T=t,Y=y)}\\&=\int_up(Y_{t'}(u)=y')p(U=u|X=x,T=t,Y=y)\end{flalign}$

因此需要

1.估计$p(U=u|X=x,T=t,Y=y)$

2.给定$do(T=t')$

3.根据$U=u,T=t'$计算$p(Y_{t'}(u)=y')$

反事实$\rho$-可识别:即真实反事实与识别反事实的误差界渐进于阈值

$\forall t' \in T,x \in X,y \in Y,\lim_{ N \rightarrow \infty} \rho(v_{t'}(x,y,y),\hat{v_{t'}}(x,y,t))\le \sigma$

![image-20230710111707302](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20230710111707302.png)

假设1.分类背景假设:背景变量$U_\epsilon$可以分解为分类变量$U_Z$和连续变量$U_\eta$,体现为隐变量的分层

$p(Y=y|X=x,T=t)=\sum_{u_z\in \{1,...,K\}}P(U_Z=u_Z)\int p(U_\eta=u_\eta)I[f_Y(x,y,u_Z,u_\eta)=y]du_\eta$

用$\gamma$表示带CTE的pdf,$\gamma_k$表示带混合因素的pdf,上式可以表示如下

$Y=|X,T\sim \gamma(X,T)=\sum_{k=1}^K \omega_k\gamma_k(X,T)$

假设$U_\eta\sim N(0,\Sigma^2)$有高斯混合$\gamma(X,T)=\sum_{k=1}^K \omega_kN(\mu_k(X,T),\Sigma_k^2)$

假设2.连续假设:对于连续函数X. T存在矩,本质上是可观察的连续变化假设

$u_k^r(x,t)=E_{Y\sim\gamma_k(x,t)}[Y^r]\in C(x,t)\forall r\in \N,K \in [K]$

假设3.群聚假设,存在一些聚类中心可以将k分群

$\forall k \in [K],\forall x,t \in(X \times T),u_k(x,t)=E_{Y\sim\gamma_k(x,t)}[Y],E_{Y\sim\gamma_k(x,t)}[||Y-\mu_k(x,t)||_2]\le \sigma$

分类背景变量的反事实可识别性:$W_1(.,.)$为第一Wasserstein距离

$\lim_{N\rightarrow \infty}E_{Y\sim \gamma(x,t)[W_1(v_{t'}(x,Y,t),v_{t'}^N(x,Y,t))]}\le \sigma$对于可加性噪声有

$\lim_{N\rightarrow \infty}W_1(v_{t'}(X,Y,T),v_{t'}^N(X,Y,T))=0$

# 算法1:反事实查询预测CFQP

数据X, Y, T,聚类数K,epoch数量$e_{max}$更新周期$\Delta$

结果:K个隐模型$m_k$

初始化:X. T. Y训练基础模型$m_0$有,$m_0(x,t)=E[Y|X=x,T=t]$

for epoch=1 : $e_{max}$

​	$\min L_0=E[(m_0(X,T)-Y)^2]$

end

$r=m_0(X,T)-Y$

初始聚类:根据r分到K'类里

for epoch=1 : $e_{max}$

​	for i=1:$\Delta$

​		$\min L_k=E[(m_k(X_k,T_k)-Y_k)^2]$

​	end

​	$R=\{m_k(X,T)-Y : k=1,...,K\}$

​	根据新残差更新聚类

end



