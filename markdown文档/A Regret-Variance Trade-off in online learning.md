# A Regret-Variance Trade-off in online learning



问题场景如下:

考虑预测一个序列，每个时刻 t，有若干个专家（expert）来给出对于该时刻可能出现状态的预测，玩家结合这些『建议』来决定一个预测，然后该时刻的实际序列被揭晓，根据预测的准确性，玩家会遭受一个损失。预测的目标就是使得玩家积累起来遭受的损失最小。具体地，希望最小化 regret，regret 定义为玩家遭受的损失和某一个专家（基准）遭受损失的差值。



考虑学习模型和专家模型,计算学习模型和专家模型的平方差,对于K个专家指数加权平均EWA可以得到负后悔或方差-后悔的o(logK)界限
较大的经验方差允许提前停止在线批量转换，并优于同类最佳预测器的风险。 当我们不考虑提前停止时，我们还恢复了模型选择聚合的最佳速率.
提出一种采样算法,方差较大时更少采样的同时保证期望的最优后悔界限
在T个在线学习流程,t个学习者会分别学习并预测,进而得到损失集合$l_{t}(\hat{y_{t}})$,有遗憾

$R_{T}=\sum_{t=1}^{T}{(l(\hat{y_{t}})-l(y^*_{t}))}$

考虑专家的预测$y_{t}(i)$有学习者的预测$\hat{y_{t}}=\sum_{i=1}^{K}p_{t}(i)y_{t}(i)$,其中$p_{t}(i)$是会变化的专家的概率分布

在特定假设下,有线性遗憾的上界为:

$\tilde{R_{T}}=\sum_{t=1}^{T}(\hat{y_{t}}-y^*_{t})l^{'}_{t}(\hat{y_{t}})\le \frac{C_{t}}{\eta}+B_{T}+\eta\sum_{t=1}^{T}(\hat{y_{t}}-y^*_{t})^{2}\\\eta \in (0,H]\\H,C_T,B_T \ge 0$



$p(i)\propto g(i)$表示$p(i)=g(i)/ \sum_{i^{'}=1}^{K}g(i^{'})$

## 算法1:基于专家的预测

输入:$\eta \in (0,0.5],M>0 $

初始化:$\gamma=\frac{\eta}{M^2},p_1(i)=1/K$

for t=1 : T

​	获得专家预测$y_{t}(i)$

​	得到预测值$\hat{y_{t}}=\sum_{i=1}^{K}p_{t}(i)y_{t}(i)$

​	收到$g_t,\kappa_t$

​	$	\tilde{l_t}(i)=\gamma(y_t(i)-\hat{y_t})g_t+\kappa_{t-1}(\gamma(y_t(i)-\hat{y_t})g_t)^2$

​	$$p_{t+1}(i)\propto exp(-\kappa_t)\sum_{s=1}^{K}\tilde{l_t}(s)$$

end

$\sum_{t=1}^{T}(\hat{y_{t}}-y_t(i))g_t\le \frac{M^2log(K)}{\kappa_T\eta}+\eta\sum_{t=1}^{T}\kappa_{t-1}(\hat{y_{t}}-y^*_{t})^{2}$

$\sum_{t=1}^{T}(\hat{y_{t}}^{EWA}-y_t^{*})\ge T/2$

## 算法2:在线学习的线性回归

输入:$\eta,\sigma,G,z>0$

初始化:$\gamma=\frac{\eta}{G^2},w_1=0,\Sigma_1^{-1}=I/\sigma$

for t=1 : T

​	收到$x_t$

​	$W_t=\bigcap_{s=1}^{t}\{w:|<w,x_s>|\le Z\}$

​	$w_t=\arg \min_{w\in W_t}(w-\tilde{w_t})^{\top}\Sigma^{-1}_{t}(w-\tilde{w_t})$

​	预测$\hat{y_{t}}=<w_t,x_t>$

​	收到$g_t,\kappa_t$

​	$z_t=\gamma x_tg_t$

​	$\Sigma_{t+1}^{-1}=\kappa_t2z_tz_t^{\top}+\Sigma_t^{-1}$

​	$\tilde{w_{t+1}}=w_t-z_t\sum_{t+1}$

end

$\begin{flalign}&\sum_{t=1}^{T}(\hat{y_{t}}-y_t(u))g_t\\&\le \frac{dG^2}{2\kappa_T\eta}\log{(1+D^2\eta^2(\max_{t=1,...,T}||x_t||^2_2)\frac{T}{d}}+\frac{G^2}{2\eta}+\eta\sum_{t=1}^{T}\kappa_{t}(\hat{y_{t}}-y_{t}(u))^{2}\end{flalign}$

界限是

$\begin{flalign}&R(\frac{1}{T}\sum_{t=1}^{T}w_t)-R(w)=O(\frac{log(K)+log(1/\sigma)}{\mu T}),\\&\eta=\mu/4,\sigma=1,Z=1,G=1,\kappa_t=1,g_t=l^{'}_t(<w_t,X_t>)\end{flalign}$

## 算法3:在线batch早停模型选择聚合

输入:T,M,早停阈值S

初始化:$S=0,\eta和M与算法1一致$

while $(S<T) \& (S>\mu/8\min_{f\in F}\sum_{t=1}^{S}(\hat{y_t}-f(X_t))^2$

​	收到$X_t$,传入专家预测$f_i(X_t)$进算法1

​	从算法1收到$p_t,\hat{y_{t}}=\sum_{i=1}^{K}p_{t}(i)f_i(X_t)$

​	预测$\hat{y_t}$收到$l_t$

​	传入$g_t=l^{'}_t(\hat{y_t}),\kappa_t=1$到算法1

​	S=S+1

end

输出$ \hat{f}=\frac{1}{S}\sum_{t=1}^{S}\sum_{i=1}^{K}p_t(i)f_i$

最佳聚合率:在损失函数适当的有界性和曲率,基于随机样本构建估计器,使得概率至少为$1-\sigma$的界限:

$R(\hat{f})-\min_{f \in F}R(f)=O(\frac{log(K)+log(1/\sigma)}{T})$

算法3可以得到这么一个界限



$R(\hat{f})-\min_{f \in F}R(f)=O(\frac{log(K)+log(1/\sigma)}{\mu T}),\eta=\mu/4,S=\infty,M=1$

## 选择性采样:算法3

$q_t概率o_t=1$,若当轮$o_t=1$,使用$\frac{o_t}{q_{t-1}}l_t$更新,可以方便控制$\kappa_t$,$\beta>0$是学习者选择的

$q_t=\min\{1,\frac{\beta}{\sqrt{\min_{i}\sum^{s=1}_{t}(\hat{y_s}-y_s(i))^2}}\}$

此时$\beta=O(\mu^{-1.5}log(K))$即可得到较优结果

## 选择性采样:在线回归

$q_t=\min\{1,\beta/\min_{w\in W_{t}\bigcap\{w:||w||_2\le D\}}\sum_{s=1}^{t}(\hat{y_s}-<w,x_s>)^2\}$

$\eta=\mu/4,\sigma=D^2,G\ge \max_t|l^{'}_t(\hat{y_t})|,Z=G/2,\kappa_t=q_t,g_t=\frac{o_t}{q_t}l_{t}^{'}(\hat{y_t})$

