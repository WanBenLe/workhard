#### "What Data Benefits My Classifier?" Enhancing Model Performance and Interpretability through Influence-Based Data Selection, Anshuman Chhabra, ICLR 2024

https://github.com/anshuman23/InfDataSel

提出使用基于树的影响(利用影响函数)估计模型来理解和解释哪些样本特征对验证集上所需的评估函数的模型性能贡献的影响力估计模型,从数据特征空间的角度解释分类器的性能,并提出了基于影响力的数据选择方法去选择不同数据,以增强模型的实用性.公平性和鲁棒性,并可用于传统分类场景.分布变化.公平中毒攻击.效用规避攻击.在线学习和主动学习.



##### 影响函数

给定训练集$Z=\{(x_i,y_i)\}^n_{i=1}$和使用经验风险最小化+损失函数$l$的分类模型,有最佳参数$\hat\theta=arg min_{\theta \in \Theta}1/n\sum l(x_i;y_i,\theta)$.影响函数基于评估感兴趣数量的影响,衡量改变验证集上无限小的样本权重的影响:对$x_j$的权重给予很小的$\epsilon$缩减得到新的最佳参数$\hat\theta(x_j;-\epsilon)=arg min_{\theta \in \Theta}1/n\sum l(x_i;y_i,\theta)-\epsilon l(x_i;y_i,\theta)$,有是事实影响:

$L^*(x_j;-\epsilon)=f(\hat\theta(x_j;-\epsilon))-f(\hat\theta(x_j))$,前者可不使用$x_j$重新训练得到.假设l严格凸且二次可微,有f可微并且有:

$L^*(x_j;-\epsilon)=lim_{\epsilon \rightarrow 1}f(\hat\theta(x_j;-\epsilon))-f(\hat\theta(x_j))\nabla_{\hat\theta}f(\hat\theta)^{\top}H_{\hat\theta}^{-1}\nabla_{\hat\theta}l(x_i;y_i,\hat\theta)$

$H_{\hat\theta}=\sum_{i=1}^{n}\nabla_{\hat\theta}^2l(x_i;y_i,\hat\theta)$是凸的l的海塞矩阵

即通过移除$x_j$重新训练度量了影响$L(-x_j)$

##### 对效用的影响

加入验证集V,有对效用的影响度量如下:

$L^{util}(-x_i)=\sum_{(x,y)\in V}\nabla_{\hat\theta}l(x;y,\hat\theta)^{\top}H_{\hat\theta}^{-1}\nabla_{\hat\theta}l(x_i;y_i,\hat\theta)$

##### 对公平性的影响

将群体公平性引入影响函数f,有二元敏感属性和预测类别概率$\hat{y}$,即有公平性度量和影响:

$f^{DP-fair}(\hat\theta,V)=|E_V[\hat y|g=1]-E_V[\hat y|g=0]|$

$L^{DP-fair}(-x_i)=\nabla_{\hat\theta}f^{DP-fair}(\hat\theta,V)^{\top}H_{\hat\theta}^{-1}\nabla_{\hat\theta}l(x_i;y_i,\hat\theta)$

##### 对抗鲁棒性的影响

考虑一个线性的白盒对手,对验证集V的每个样本x添加扰动以得到对抗验证集$V'$:$x'=x-γ\frac{\hat θ^\top x+b}{\hat θ^\top\hat θ}\hat θ$,其中$\hat θ \in R^d$是系数,$b ∈ R$ 是截距,$γ > 1$是扰动系数,于是有对对抗鲁棒性的影响:

$L^{robust}(−x_i) = \sum_{(x',y)\in V'}\nabla_{\hat\theta}l(x';y,\hat\theta)^{\top}H_{\hat\theta}^{-1}\nabla_{\hat\theta}l(x_i;y_i,\hat\theta)$

##### 非凸处理

1.线性模型作为非凸模型emb的代理

2.非凸模型增加阻尼项使得海塞矩阵正定可逆

3.特定任务有特定的二阶影响函数





##### 算法1:树模型进行影响估计

输入:训练集$Z$,验证集$V$,影响函数$L^F$,超参$\lambda$

输出:-影响估计树$\hat h$

$M=\empty$

for $(x_i,y_i) \in Z$:

​	$q_i=|x_i,y_i|$

​	$M=M\cup \{(q_i,L^F(-x_i))\}$

使用$M$训练CART树的$h$​

返回分层收缩后的$\hat h(q_i)$,见下



定义叶到根节点的路径$t_w \subset t_{w-1}\subset... \subset t_0$最左为叶节点,最右为根节点

函数$\phi$返回输入树节点的样本数量,函数$\xi$返回输入查询$q$和树节点$t$的预测的平均值

有$h(q_i)=\xi(q_i,t_0)+\sum^w_{j=1}\xi(q_i,t_j)-\xi(q_i,t_{j-1})$和分层收缩正则:

$\hat h(q_i)=\xi(q_i,t_0)+\sum^w_{j=1}\frac{\xi(q_i,t_j)-\xi(q_i,t_{j-1})}{1+\lambda/\phi(t_{j-1})}$



##### 算法2:数据修剪

输入:训练集$Z$,验证集$V$,影响函数$L^F$,预算$b$

输出:修剪后数据集$Z'$

$J=\empty,K=\empty,Z'=\empty$

for $(x_i,y_i) \in Z$:

​	$J=J\cup \{L^F(-x_i)\}$

​	$K=K\cup \{i\}$

J = sort( J , ascending = True)

$b'=\sum 1_{j<0,j \in J}$

$Z'=Z\cup \{x_i\},\forall i \notin K_{:min\{b,b'\}}, x_i \in Z$

返回$Z'$

本质上是删除了负面影响的样本