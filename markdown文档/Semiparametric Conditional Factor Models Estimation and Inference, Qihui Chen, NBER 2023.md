# Semiparametric Conditional Factor Models: Estimation and Inference, Qihui Chen, NBER 2023
考虑一个半参因子模型,$z$是因子

$y_{it}=\alpha(z_{it})+\beta(z_{it})'f_t+\epsilon_{it},i=1,..,N,t=1,...,T$

假设$\alpha(\cdot)$不存在,$\beta(\cdot)$是线性的,有$\alpha(\cdot)=0,\beta(z_{it})=\Gamma'z_{it}$,可以重写上式

$Y_t=Z_t\Gamma f_t+\epsilon_t$

使用标化PCA,处理一下有可以进行估计的下式(1)

$(Z'_tZ_t)^{-1}Z'_tY_t=\Gamma f_t+(Z'_tZ_t)^{-1}Z'_t\epsilon_t$

考虑一个非0$\alpha(\cdot)$的一般情况,假设$\{\alpha_m(\cdot)\}_{m\leq M}$, $\{\beta_{km}(\cdot)\}_{m\leq M}$避免维度灾难问题,于是有

$\alpha(z_{it})=\sum_{\{m=1,M\}}\alpha_m(z_{it},m),\beta_k(z_{it})=\sum_{\{m=1,M\}}\beta_{km}(z_{it},m)$

使用筛法估计$\alpha_m(\cdot)和\beta_{km}(\cdot)$,需要引入基函数$\{\phi_j(\cdot)\}_{j\ge1}$,例如B样条.傅里叶.多项式等,基函数等,重写如下所示,基本假设有筛法J其后的误差项是趋近于0的

$\alpha_m(z_{it},m)=\sum_{\{j=1,J\}}a_{m,j}\phi_j(z_{it},m)+r_{m,J}(z_{it},m)$

$\beta_{km}(z_{it},m)=\sum_{\{j=1,J\}}b_{km,j}\phi_j(z_{it},m)+\sigma_{km,J}(z_{it},m)$

写成矩阵形式有

$\alpha(z_{it})=a'\phi(z_{i,t})+r(z_{i,t}),\beta(z_{i,t})=B'\phi(z_{i,t})+\sigma(z_{i,t})$

基于假设省略误差项有

$Y_t=\Phi(Z_t)a+\Phi(Z_t)Bf_t+\epsilon_t$

使用回归方法,让$\Phi(Z_t)$表征$Y_t$,有

$\widetilde{Y_t}=a+Bf_t,\widetilde{Y_t}=(\Phi(Z_t)'\Phi(Z_t))^{-1}\Phi(Z_t)'Y_t$

1.对上式减去$\widetilde{Y_t}$的均值然后进行PCA可以估计出B

2.继续假设$a'B=0$,有$a\approx(I_{JM}-B(B'B)^{-1}B)\bar{\widetilde{Y}}$

3.$f_t\approx(B'B)^{-1}B'\widetilde{Y_t}$

$\hat{\alpha}(z)=\hat{\alpha}'\phi(z),\hat{\beta}(z)=\hat{B}'\phi(z),\hat{F}=\widetilde{Y}'\hat{B}$

$M_T=I_T-1_T1_T'/T$

对上述估计使用等权的加权bootstrap,给定了标准指数分布和$var(w_i)=\omega_0=1$,T=2,K=1,进一步有bootstrap估计量

$\hat{B}^*=\widetilde{Y}^*M_T\hat{F}(\hat{F}'M_T\hat{F})^{-1},\hat{a}^*=(I_{JM}-\hat{B}^*(\hat{B}^{*'}\hat{B}^{*})^{-1}\hat{B}^{*'})\bar{\widetilde{Y}}^*$

与下列分布参数有关

$(\sqrt{NT/\omega_0}(\hat{a}^*-\hat{a},\sqrt{NT/\omega_0}(\hat{B}^*-\hat{B})\sim(G_a^*,G_b^*)$

更进一步的

$H=(F'M_TF/T)^{1/2}\gamma\nu^{-1/2},M=H\nu^{-1}$

$\nu$是$(F'M_TF/T)^{1/2}B'B(F'M_TF/T)^{1/2}$特征值的对角矩阵,$\gamma$是对应的特征向量矩阵 

$G_a^*=(I_{JM}-BHH'B')(N_1^*-G_B^*H^{-1}\bar{f})-BHG_B^{*'}a$

$G_B^{*}=N_2^*B'BM$

$N_1^*,N_2^*$分别为$N^*$的第一列和最后K列

