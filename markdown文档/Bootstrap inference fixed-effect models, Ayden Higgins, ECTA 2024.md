#### Bootstrap inference fixed-effect models, Ayden Higgins, ECTA 2024



一般的面板估计时间数量m固定而个体数量n是动态的,但当m没有跟n以同样比率增大的情况下矩阵列估计是渐进有偏的,进而导致大样本的渐进正态分布也是错误的.当前论文表明bootstrap参数可以一致估计矩阵列渐进的MLE分布和渐进bias.这也意味着合理的bootstrap可以在大样本下得到合适的假设检验覆盖范围(上次分享相关的EBCI是就是fix给定$\alpha$​,贝叶斯下可信区间太小无法合理cover的问题),且不需要debias,这个结果对于FE的均值(e.g.平均边际效应).LR检验.score检验也成立.



##### 一. Bias分析

###### 1.对于一个sufficiently-regular(充分正则)允许固定外生协变量$x_i$​非线性FE panel模型的MLE结果

$l(\varphi,\eta_i|z_{it})=log \space f(y_{it}|y_{it-p},...,y_{it-1},x_{it};\varphi,\eta)$

$(\hat{\varphi},\hat{\eta_1},...,\hat{\eta_n}):=arg \space max_{\varphi,\eta_1,...,\eta_n} \sum^n_{i=1}\sum^m_{t=1}l(\varphi,\eta_i|z_{it})$​

$\large{\rho_{i,m}:=(\frac{1}{m}\sum^m_{t=1}E(\frac{\partial^2l(\varphi_0,\eta_{i0}|z_{it})}{\partial\varphi\partial\eta_{i}'}))(\frac{1}{m}\sum^m_{t=1}E(\frac{\partial^2l(\varphi_0,\eta_{i0}|z_{it})}{\partial\eta_{i}\partial\eta_{i}'}))^{-1}}$​

$\large{\Omega_{nm}:=-1/nm \sum^n_{i=1}\sum^m_{t=1}E(\frac{\partial^2l(\varphi_0,\eta_{i0}|z_{it})}{\partial\varphi\partial\varphi'}-\rho_{i,m}\frac{\partial^2l(\varphi_0,\eta_{i0}|z_{it})}{\partial\eta_{i}\partial\varphi'})}$

$\Sigma:=(lim_{n,m \rightarrow \infty}\Omega_{nm})^{-1}$

$n,m \rightarrow \infty,n/m\rightarrow\gamma^2,0<\gamma<\infty: \sqrt{nm}(\hat{\varphi}-\varphi_0)\overset{L}{\rightarrow}N(\gamma\beta,\Sigma)$​

即$\beta$是非随机渐近的bias,若$n/m$不接近于0是无法忽略bias的,这种情况下有一种偏差校正方法:$\hat{\varphi}-\hat{\beta}/m$

###### 2.给定函数$\mu$的例如平均边际效应或者FE的矩:

$\Delta:=lim_{n,m \rightarrow \infty}1/nm\sum^n_{i=1}\sum^m_{t=1}E(\mu(z_{it},\varphi_0,\eta_{i0}))$

的MLE结果$\hat{\Delta}:=1/nm\sum^n_{i=1}\sum^m_{t=1}\mu(z_{it},\hat{\varphi},\hat{\eta_i})$即使做了偏差校正也是存在bias的

###### 3.对$\phi(\varphi_0)=0$做LR检验和拉格朗日乘数LM检验,考虑前者的MLE结果:

$\hat{\eta_i}(\varphi):=arg \space max_{\eta_i}\sum_{t=1}^ml(\varphi,\eta_i|z_{it})$​

$\hat{\varphi}:=arg \space max_{\varphi}\sum^n_{i=1}\sum^m_{t=1}l(\varphi,\hat{\eta_i}(\varphi)|z_{it})$

原假设

$\check{\varphi}:=arg \space max_{\varphi:\phi(\varphi)=0}\sum^n_{i=1}\sum^m_{t=1}l(\varphi,\hat{\eta_i}(\varphi)|z_{it})$​

LR统计检验量

$\hat{\omega}=2\sum^n_{i=1}\sum^m_{t=1}(l(\hat{\varphi},\hat{\eta_i}(\hat{\varphi})|z_{it})-l(\check{\varphi},\hat{\eta_i}(\check{\varphi})|z_{it}))$

会因为bias而服从非中心的$\chi^2$​​分布



##### 二. Bootstrap渐近性质

给定参数$\theta:=(\varphi,\eta_1,...,\eta_n)$,有期望$E_\theta$和概率$P_\theta$,$\varphi.\eta_i$的参数空间$V_\varphi.V_\eta$,有$\theta$的参数空间$\Theta=V_\varphi *V_\eta*...*V_\eta$和后者的子集$\Theta_0$

###### 假设1:

a. $\varphi \in V_\varphi, \eta_i \in V_\eta$的密度$f$是连续函数

b. 真实参数在$\Theta_0 \in \Theta$中

###### 假设2:

考虑从$z_{it-q}$和$z_{it+q}$基于$\theta$生成的两个$\sigma$代数$A_{it}(\theta),B_{it}(\theta)$,有混合系数

$a_i(\theta,h):=sup_{1\le t\le m}sup_{A\in$A_{it}(\theta)}sup_{B\in$B_{it+h}(\theta)}|P_\theta(A\cap B)-P_\theta(A)P_\theta(B)|$

有个覆盖$\Theta_0$的开集$\Theta_1:=\{\theta\in\Theta:d(\theta,\Theta_0)<\sigma\},\sigma>0$

$d(\theta,\Theta_0):=inf\{||\theta-\vartheta||_2:\vartheta\in \Theta_0\}$,e.g.$\theta$到$\Theta_0$的距离

a. $sup_{1\le i\le n}sup_{\theta\in \Theta_1}a_i(\theta,h)=O(r^h),0<r<1$

###### 假设3:平滑性和矩要求

a. $l(\varphi,\eta_i|z_{it})$对于 $\varphi,\eta_i$几乎必然四次可微

b. $l(\varphi,\eta_i|z_{it})$及其相关的交叉导数,直到四阶,几乎必然有界$b(z_{it})$​满足

$sup_{1\le i\le n}sup_{1\le t\le m}sup_{\theta\in \Theta_1}E_\theta(|b(z_{it})|^q) <\infty$

$3+(dim(\varphi)+dim(\eta_i))/2<qs,0<s<0.1$​

c. $m \rightarrow \infty,\theta \in \Theta_1, 1/m\sum_{t=1}^mE_\theta(b(z_{it})) $对i一致收敛到$lim_{m \rightarrow \infty}1/m\sum_{t=1}^mE_\theta(b(z_{it}))$​

$H_i(\varphi,\eta_i|\vartheta):=lim_{m \rightarrow \infty}\frac{1}{m}\sum_{t=1}^mE_\vartheta(l(\varphi,\eta_i|z_{it}))$

###### 假设4:时序TS变化识别参数假设

a. $\large{\inf_{1\le i\le n}\inf_{\theta\in \Theta_1}(H_i(\varphi,\eta_i|\theta)-sup_{\{(\bar{\varphi},\bar{\eta_i}):||(\bar{\varphi},\bar{\eta_i})-(\varphi,\eta_i)||_2>\varepsilon\}}H_i(\bar{\varphi},\bar{\eta_i}|\theta))>\sigma_\varepsilon}$

######  假设5:矩阵列渐进假设

a. $n,m \rightarrow \infty,n/m\rightarrow\gamma^2,0<\gamma<\infty$

###### 假设6:良定的$\varphi$的渐进方差假设

第一部分的符号重写成$\Omega_{nm,\theta}$强调对后者的依赖,$\omega_{min}(A).\omega_{max}(A)$是A的最小特征值和最大特征值

$\exist \epsilon_1,\epsilon_2,\varepsilon_1,\varepsilon_2$是有限的且满足

a. $\epsilon_1 \le \inf_{1\le i\le n}\inf_{\theta\in \Theta_1}\omega_{min}(\frac{1}{m}\sum_{t=1}^mE_\theta(\frac{\partial^2l(\varphi,\eta_{i}|z_{it})}{\partial\eta_{i}\partial\eta_{i}'}))\\\le  \sup_{1\le i\le n}\sup_{\theta\in \Theta_1}\omega_{max}(\frac{1}{m}\sum_{t=1}^mE_\theta(\frac{\partial^2l(\varphi,\eta_{i}|z_{it})}{\partial\eta_{i}\partial\eta_{i}'}))\le \epsilon_2$

b. $\varepsilon_1\le \inf_{\theta\in \Theta_1}\omega_{min}(\Omega_{nm,\theta}) \le\sup_{\theta\in \Theta_1}\omega_{max}(\Omega_{nm,\theta}) \le\varepsilon_2$​

###### 定理1:给定假设1-6,$P^*$是bootstrap概率测度

$\forall \varepsilon>0,P(\sup_a|P^*(\sqrt{nm}(\hat{\varphi}^*-\hat{\varphi})\le a)-P(\sqrt{nm}(\hat{\varphi}-\varphi_0)\le a)|>\varepsilon  )=o(1)$

###### 推论1:给定假设1-6

$Q^*_\alpha$是满足的$P^*(c'(\hat{\varphi}^*-\hat{\varphi})\le Q^*)\ge\alpha$最小的$Q^*$,c是$||c||_1<\infty$的给定向量有

$\forall \alpha \in(0,1),P(c'\hat{\varphi}-Q^*(\alpha)\le c'\varphi_0) =\alpha+o(1)$

###### 推论2:由于$\hat{\Sigma}.\hat{\Sigma}^*$的一致性,给定假设1-6

$Q^*_\alpha$是满足的$P^*((c'\hat{\Sigma}^*c)^{(-1/2)}c'(\hat{\varphi}^*-\hat{\varphi})\le Q^*)\ge\alpha$最小的$Q^*$,c是$||c||_1<\infty$的给定向量有

$\forall \alpha \in(0,1),P(c'\hat{\varphi}-(c'\hat{\Sigma}^*c)^{(-1/2)}Q^*(\alpha)\le c'\varphi_0) =\alpha+o(1)$

即studentized bootstrap在大样本有正确的覆盖范围,假设检验的t检验随着$n,m\rightarrow \infty$也会逼近$\alpha$

定理1和推论1.推论2为basic bootstrap提供了推断合理性.

###### 定理2:给定假设1-6,$\phi$是$V_\varphi$的非随机连续可微向量值函数,有

$\forall \varepsilon>0,P(\sup_a|P^*(\sqrt{nm}(\phi(\hat{\varphi}^*)-\phi(\hat{\varphi}))\le a)-P(\sqrt{nm}(\phi(\hat{\varphi})-\phi(\varphi_0))\le a)|>\varepsilon  )=o(1)$

即通过$\phi$将推断拓展到了非线性,为LR提供了铺垫.

###### 定理3:给定假设1-6,基于定理2考虑原假设$\phi(\varphi_0)=0$

额外假设真实参数在$\Theta \cap \{ \varphi\in V_\varphi: \phi(\varphi)=0\}$中,$\phi$在$V_\varphi$中,且五阶连续可微.导数有界.雅克比矩阵有最大行的秩,有

$\forall \varepsilon>0,P(\sup_a|P^*(\hat{\omega}^*\le a)-P(\hat{\omega}\le a)|>\varepsilon  )=o(1)$

定理3控制了LR统计检验量,score统计检验量也有同样的结论.



##### 三. Bootstrap推断

固定协变量下,给定从原始数据$y_{it}^*$拟合转移密度$f(y_{it}^*|y^*_{it-p},...,y^*_{it-1},x_{it};\hat{\varphi},\hat{\eta_i})$中bootstrap的观测值$y_i^*:=(y_{i1}^*,...,y_{im}^*)$,有MLE:

$(\hat{\varphi}^*,\hat{\eta_1}^*,...,\hat{\eta_n}^*):=arg \space max_{\varphi,\eta_1,...,\eta_n} \sum^n_{i=1}\sum^m_{t=1}l(\varphi,\eta_i|z_{it}^*)$

$z^*_{it}:=(y^*_{it-p},...,y^*_{it-1},y^*_{it},x_{it})$

###### 1. 置信区间CI

$n,m \rightarrow \infty,n/m\rightarrow\gamma^2$有bootstrap测度的弱收敛

$ \sqrt{nm}(\hat{\varphi}-\varphi_0)\overset{L^*}{\rightarrow}N(\gamma\beta,\Sigma)$

以此构造不需要偏差校正的bootstrap. e.g选择一致维度的向量$c$. $Q^*(\alpha):=\inf\{q:\alpha\le F^*(q)\}$是隐含分位数函数.

$F^*(a):=P^*(c'(\hat{\varphi}^*-\hat{\varphi})\le a)$​有大样本下相等的上单侧和双侧的CI

$\{^*:c'\hat{\varphi}-Q^*(\alpha)\le c'\hat{\varphi}\}$

$\{c'\hat{\varphi}:c'\hat{\varphi}-Q^*(1+\alpha/2)\le c'\hat{\varphi}\le c'\hat{\varphi}-Q^*(1-\alpha/2)\}$

由于方差的渐进性质使用studentized bootstrap并用$(c'\hat{\Sigma}c)^{(-1/2)}$缩放有临界值$(c'\hat{\Sigma}^*c)^{(-1/2)}c'(\hat{\varphi}^*-\hat{\varphi})$

###### double bootstrap方案:考虑了bias

固定协变量下,从拟合转移密度$f(y_{it}^{**}|y^{**}_{it-p},...,y^{**}_{it-1},x_{it};\hat{\varphi}^*,\hat{\eta_i}^*)$中bootstrap的观测值$y_i^*:=(y_{i1}^{**},...,y_{im}^{**})$,并MLE得到$\hat{\varphi}^{**},\hat{\eta_i}^{**}$

考虑和$F^{**}(a):=P^{**}(c'(\hat{\varphi}^{**}-\hat{\varphi}^{*})\le a)$关联的分位数函数$Q^{**}(\alpha):=\inf\{q:\alpha\le F^{**}(q)\}$

$\forall \alpha \in(0,1),\hat{\alpha}^*(a)= P^*(c'\hat{\varphi} \in \{c'\hat{\varphi}:c'\hat{\varphi}^{*}-Q^{**}(\alpha)\le c'\hat{\varphi}\} )$是上单侧CI的覆盖概率

令$\hat{\alpha}^*$满足$\hat{\alpha}^*(a)=\alpha$,有

$\{c'\hat{\varphi}:c'\hat{\varphi}-Q^*(\alpha^*)\le c'\hat{\varphi}\}$

studentized bootstrap和双侧CI都可以类似方法构建,通过迭代让覆盖范围更大可能也是可行的.

$\Delta$的CI也同理$$\hat{\Delta}^*:=1/nm\sum^n_{i=1}\sum^m_{t=1}\mu(z_{it}^*,\hat{\varphi}^*,\hat{\eta^*_i})$$

###### 2.点估计$c'\varphi_0$

由于bootstrap分布$F^*$的中位数是$c'\beta/m$的有效估计量,因此$c'\hat{\varphi}-Q^*(1/2)$是偏差校正估计,当然给定截尾参数的截尾均值可能也是可行的.

由于偏差校正估计是的一阶方差不变,因此$(c'\hat{\Sigma}c)/nm$是$c'\hat{\varphi}-Q^*(1/2)$​方差的有效估计量.当然截尾参数之后bootstrap可能也是可行的.

###### 3.假设检验

bootstrap p值渐进服从均匀分布,e.g.大样本中,拒绝原假设$c'\varphi_0\le c'\hat{\varphi}$的条件是$(c'\hat{\Sigma}c)^{(-1/2)}c'(\hat{\varphi}-\hat{\varphi})$超过$(c'\hat{\Sigma}^*c)^{(-1/2)}c'(\hat{\varphi}^*-\hat{\varphi})$的$(1-\alpha)$分位数.

对于LR 考虑原假设$\phi(\varphi_0)=0$,定义$F^*(a):=P^*(\hat{w^*}\le a),Q^*$是LR统计检验量的bootstrap分位数函数,对于MLE结果

$\large{\check{\varphi}^*:=arg \space max_{\varphi:\phi(\varphi)=\varphi:\phi(\hat{\varphi})}\sum^n_{i=1}\sum^m_{t=1}l(\varphi,\hat{\eta_i}^*(\varphi)|z_{it}^*)}$

$\hat{\eta_i}(\varphi)^*:=arg \space max_{\eta_i}\sum_{t=1}^ml(\varphi,\eta_i|z_{it}^*)$​

LR 统计检验量在大样本下

$\hat{\omega}^*=2\sum^n_{i=1}\sum^m_{t=1}(l(\hat{\varphi}^*,\hat{\eta_i^*}(\hat{\varphi}^*)|z_{it}^*)-l(\check{\varphi}^*,\hat{\eta_i}^*(\check{\varphi}^*)|z_{it}^*))$

拒绝原假设的条件是$\hat{\omega}>Q^*(1-\alpha)$,或者用p值形式$p^*:=1-F^*(\hat{\omega})$

$\hat{\varphi}:=arg \space max_{\varphi}\sum^n_{i=1}\sum^m_{t=1}l(\varphi,\hat{\eta_i}(\varphi)|z_{it})$

double bootstrap方案:对于$\hat{\omega}^{**}$和$F^{**}(a):=P^{**}(\hat{w^{**}}\le a)$的分位数函数$Q^{**}$有

$\hat{\omega}>Q^*(1-\alpha^*)$

$\alpha^*$是满足$\hat{\alpha}^*(\alpha):=1-F^*(Q^{**}(1-a))$的$\hat{\alpha}^*(\alpha^*)=\alpha$的解

