#### Helen: Optimizing CTR Prediction Models with Frequency-wise Hessian Eigenvalue Regularization, Zirui Zhu, WWW 2024

发现top数据的Hessian特征值和特征频率存在强正相关导致特征收敛于尖锐的局部最小值(锐度感知最小化SAM表明减低loss锐度可增加模型泛化能力),因此提出CTR预估的.是用归一化特征频率的自适应扰动对频率Hessian特征值进行正则化的优化器Helen.

##### CTR预估与模型优化

$S\{(x_i,y_i)\}_{i=1}^n$是user. item及交互的分类信息训练集表征,m个特征one-hot后的样本服从分布$D$且为用户点击与否的二元标签.预测网络$f(x;[h,[e^j_1,e^j_2,...,e^j_{s_j}]])$,h是隐藏层,$e^j_s$是第j个特征的第s个维度emb

省流:连续特征离散化/分箱,得到的很多特征的离散化的emb就是样本.

优化如下所示,$L(x,y,f(x;w))$是损失函数

$w^*=arg \space min_{w} E_{(x,y)\sim D}L(x,y,f(x;w))$

对于未知分布有经验风险最小化ERM结果

$\hat{w}=arg \space min_{w} \frac{1}{n}\sum _{i=1}^nL(x_i,y_i,f(x_i;w))$

###### 锐度感知最小化SAM

SAM的$L_S(w)$考虑了$l_p$球内的临近点,约束限制了扰动的幅度:

$L^{SAM}_S(w)max_{||\epsilon||_p\le \rho}L_S(w+\epsilon)$

使用一步梯度近似优化不可行问题:

$\hat{\epsilon}(w)=\frac{\nabla_wL_S(w)}{||\nabla_wL_S(w)||}\simeq arg\space max_{||\epsilon||_p\le \rho}L_S(w+\epsilon)$

更新为

$g=\nabla_wL^{SAM}_S(w)\simeq \nabla_wL_S(w)|_{w+\hat{\epsilon}}$

###### Helen优化器

基于梯度的优化方法可以收敛到局部最小$w^*$使得

$g=\nabla_wL_S(w^*)=0,H=\nabla_w^2L_S(w^*)$​是半正定的且不存在负的特征值.如果loss的Hessian矩阵的特征值很大,容易在病态优化上出现泛化性能更差的收敛.

特征频率计算方法为:

$N_k^j(S)=\sum_{i=1}^nx_i^j[k]$

对应嵌入$e$的权重的梯度$g$的Hessian矩阵$H$,对于DeepFM和PNN都出现了特征值与特征频率相关的情况-在CTR预估特征维度很高且高度稀疏,而且高热特征出现的频率会比低热特征高很多,导致特征分布高度倾斜,进而导致高热特征收敛到更尖锐的局部最小中.

最小化SAM损失函数会在原始损失$L_S(W)$在$O(\rho)$临近流形中引入bias$arg \space min _{w}\lambda(\nabla_w^2L_S(w))$进而减少了特征值,但是扰动半径$\epsilon$会影响到优化的结果(过小/过大导致进入平坦/锐化局部最小)

Helen使用特征$k$的域$j$频率$N_k^j$计算扰动半径$\rho_k^j$,$\zeta$可以对于低频特征的域的避免扰动半径过小.

$\rho_k^j=\rho\cdot max\{\frac{N_k^j}{max_kN_k^j},\zeta\}$​

对SAM的近似进行一阶泰勒展开有

$\hat{\epsilon}(e^j_k)=arg\space max_{||\epsilon||_p\le \rho*k}L_S(e^j_k+\epsilon)\simeq  \rho^j_k\cdot\frac{\nabla_{e^j_k}L_S(w)}{||\nabla_{e^j_k}L_S(w)||}$

$g^{Helen}= \nabla_wL_S(w)|_{w+\hat{\epsilon}(w)}$

###### Helen优化器算法

for i in range(T):

​	数据集S中的b个样本集$B$

​	$g=1/b\sum_{x\in B} \nabla_wL_B(w)$

​	$\hat{\epsilon}(w)=[\rho\cdot g_h/||g_h||]$

​	for k in range(len(feature)):

​		for j in range(len(field)):

​			$N_k^j(S)=\sum_{i=1}^nx_i^j[k]$

​			$\rho_k^j=\rho\cdot max\{\frac{N_k^j}{max_kN_k^j},\zeta\}$​

​			$\hat{\epsilon}(e^j_k)= \rho^j_k\cdot\frac{\nabla_{e^j_k}L_S(w)}{||\nabla_{e^j_k}L_S(w)||}$

​			$\hat{\epsilon}(w).append(\hat{\epsilon}(e^j_k))$

​	$g^{Helen}= \nabla_wL_S(w)|_{w+\hat{\epsilon}(w)}$

​	$w=w-\eta\cdot g^{Helen}$