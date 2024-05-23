# Confidence-based Reliable Learning under Dual Noises

标签(y噪声)和图像(x噪声)存在双重噪声

1.提出了一种基于标签置信度(过滤y)和最大置信度(过滤x)的样本过滤器而无需指定噪声比

2.对噪声数据不确定性进行惩罚



图像噪声有三种:标签错误的y噪声/图像很糊的x噪声1/背景噪声的x噪声2

$D=\{(x_i,y_i)\}^N_{i=1}$是含噪的图像-标签对,概率分布为$p_\theta(y|x)=p(y|f_\theta(x))$

图像分类是形如MLE以下函数,甚至还能加上L2正则项做MAP估计

$\min_{\theta}l(\theta,D)=-\frac{1}{N}\sum_{i=1}^{N}log(f_\theta(x_i)[y_i])$

为了估计不确定性BNN会给$p_\theta$先验分布,然后用$p(\theta|D)$做后验推理

使用深度集成方法估计了M个随机初始化独立训练的DNN$\{f_{\theta_m}\}^M_{m=1}$,并投票得到预测

$\frac{1}{M}\sum_{m=1}^{M}f_{\theta_m}(x)$

可用于过滤y噪声的标签置信度估计如下

$L-Con(x)=\frac{1}{M}\sum_{m=1}^{M}f_{\theta_m}(x)[y]$

假设是模型会为具有 y 噪声的训练数据产生低 L-Con,而为其他数据产生高 L-Con,这个假设不会随着训练而变化

可用于过滤x噪声的最大置信度估计如下,该值与数据不确定性有与香农熵相关的内在联系

$M-Con(x)=\max_{j}(\frac{1}{M}\sum_{m=1}^{M}f_{\theta_m}(x))[j]$

数据不确定性随着输入图像质量的下降而增加

$w_i^l=\begin{cases}
0, L-Con(x_i)\le \epsilon_1 \\
1,otherwise
\end{cases}$

$w_i^k=\begin{cases}
0, M-Con(x_i)\le \epsilon_2 \\
1,otherwise
\end{cases}$

$w_i^s=w_i^l*w_i^k$

过滤方法如上所示,$w_i^s$过滤了阈值低于$\epsilon$的噪声样本

模型不确定性=总不确定性-数据不确定性,可以用深度集成方法近似计算

$I[y,\theta|x;D]=H[E_{p(\theta|D)}(p(y|x,\theta))]-E_{p(\theta|D)}[H(p(y|x,\theta))]$

$I[y,\theta|x;D]\approx H[\frac{1}{M}\sum_{m=1}^{M}p_{\theta_m}(y|x)]-\frac{1}{M}\sum_{m=1}^{M}H[p_{\theta_m}(y|x)]$

优化目标为

$\min_{\theta_m}l(\theta_m,D)=\begin{cases}
\sum_{i=1}^{N}-log(f_{\theta_m}(x_i))[y_i]),w_i^s=1 \\
\sum_{i=1}^{N}I(y,\theta|x_i,D),w_i^s=0
\end{cases}$

## 算法1 含x,y噪声的DNN训练

输入:含噪数据集D,集成数量M,阈值$\epsilon_1,\epsilon_2$

初始化M个神经网络$f_{\theta_M}$

for m=1:M

​	#预热学习率训练

​	$\theta^{(m)}<-WarmUp(D,\theta^{(m)})$

end

while e<MaxEpoch

​	for $B \in D$

​		计算$L-Con,M_Con$

​		得到$w_i^s$

​		更新每个神经网络

​		$L(\theta_m,B)=\sum_{(x_i,y_i) \in B}(1-w_i^s)I(y_i,\theta)+w_i^sL_{CE}(\theta_m,B)$

​	end

​	e+=1

end

M大于4,取5就差不多了,$\epsilon_1,\epsilon_2$可以考虑取0.02和0.05,模型效果对这两个阈值不太敏感

