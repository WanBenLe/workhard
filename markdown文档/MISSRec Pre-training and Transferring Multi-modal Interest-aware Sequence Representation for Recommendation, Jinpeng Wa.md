MISSRec: Pre-training and Transferring Multi-modal Interest-aware Sequence Representation for Recommendation, Jinpeng Wang, ACM MM, 2023

顺序推荐(SR)根据用户的历史交互序列来预测用户潜在感兴趣的项目.

$s_i=[I_1,I_2,...,I_t]$预测$I_{t+1}$,i是第i个用户

对于稀疏ID和不一致的ID映射限制了模型的可转移性.并对冷启动问题和流行度偏差问题处理不佳.本文提出MISSRec,是用于SR的多模态预训练和迁移学习框架.



用户端:Transformer的捕获序列级多模态用户兴趣encoder-兴趣感知decoder

候选项目:动态融合生成自适应项目表示.

1.多模态协同取决于用户而且是动态的.

2.信息冗余可能会压倒基本兴趣(大多数用户行为倾向于同质化商品),平等对待会导致预测偏差.



MISSRec 由 7 个主要组件组成：

1.通用多模态item表征

a.使用冻结的BERT和ViT提取文本特征$f^t=\phi^t(I)$和视觉特征$f^v=\phi^v(I)$.

b.特征增强,对文本和视觉特征各做2次增强得到$\tilde{F_i^t}.\tilde{F_i^{t'}}.\tilde{F_i^v}.\tilde{F_i^{v'}}$

2.特定于模态的适配器弥合差异,这么做允许预训练也可以用于参数高效传输

$X_i=[\psi^t(\tilde{F_i^t});\psi^v(\tilde{F_i^v})]$

3.多模式兴趣发现(MID)模块

对历史交互序列的item集$\mathcal{X}=\{x_i\}_{i=1}^N$走𝑘-最近邻的密度峰值聚类算法DPC-KNN

对于第i个token基于k-最近邻得到局部密度

$\rho_i=exp(-\frac{1}{k}\sum_{x_j \in kNN(x_i)}||x_i-x_j||^2_2)$​

进而计算到其它更高局部密度的token的最小距离

$\sigma_i=min_{j :\rho_j>\rho_i}||x_i-x_j||_2^2,\exist j \space s.t. \rho_j>\rho_i$​

或者和其它token的最大距离

$\sigma_i=max_j \space||x_i-x_j||_2^2,else$

最后使用$K_c$最高$\rho_i*\sigma_i$的质心作为兴趣远行,并得到每个token的兴趣分配.

最后去重得到兴趣分配序列$\beta_1,\beta_2,...,\beta_M$

token-兴趣分配可以预存储

4.统一序列建模和多模态融合兴趣序列的多模态兴趣感知序列模型

a.给定$K_e$个标准encoder得到兴趣序列$\beta$对应的编码兴趣序列$\xi_1,\xi_2,...,\xi_M$​

b.给定$K_d$个decoder,编码兴趣序列$\xi_1,\xi_2,...,\xi_M$会作为K和V

逐项位置emb$p_i=[p_1,p_2,...,p_{L_i}]$添加到item token$X_i=[x_1,x_2,...,x_{L_i}]$作为decode的query

$u_i=y_{L_i}$后者是decode output的最后一位

5.动态融合的候选项目表征

对候选集$I_k$走2得到$x_k^t.x_k^v$,对user$i$走3.4得到序列表示$u_i$,进而计算得到自适应表征$v_k$,$\alpha>0$是重要性权重

$\Large {v_k=\frac{e^{\alpha \cdot<u_t,x_k^t>  }\cdot x^t_k+e^{\alpha \cdot <u_i,x^v_k> }\cdot x^v_k}{e^{\alpha \cdot <u_i,x_k^t>  }+e^{\alpha \cdot <u_i,x_k^v>  }}}$

令$s^t_{i,k}=<u_i,x_k^t>,s_{i,k}^v=<u_i,x_k^v>$,有在mean pooling和max pooling直接的可分解的融合匹配分:

$\Large {<u_i,v_k>=\frac{s^t_{i,k}\cdot e^{\alpha \cdot s^t_{i,k} }+s^v_{i,k}\cdot e^{\alpha \cdot s^v_{i,k} }}{e^{\alpha \cdot s^t_{i,k} }+e^{\alpha \cdot s^v_{i,k}  }}}$

6.自监督对比学习预训练

a.序列-item对比学习

$\Large {t_i^{S-I}=-log\frac{exp(<u_i,v_{T_i+1}>/\tau)}{\sum^B_{j=1}exp(<u_i,v_{T_j+1}>/\tau)}}$

温度系数$\tau >0$,$B$是mini-batch的数量

b.序列-序列对比学习

$\Large {t_i^{S-S}=-log\frac{exp(<u_i,u_i'>/\tau)}{\sum^B_{j=1}exp(<u_i,u_j>/\tau)+exp(<u_i,u^{'}_{j}>/\tau)}}$​

跟文本增强的序列拉近

c.预训练总loss

![image-20240313140001686](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240313140001686.png)

$\mathcal{L}_{pre-train}=\frac{1}{B}\sum^B_{i=1}[t_i^{S-I}+\lambda \cdot t_i^{S-S}+\frac{\gamma}{M_i^2}\sum^{M_i}_{m,m'=1}<\xi_m,\xi_{m'}>] \space\lambda,\gamma>0$

最后一项是使得decoder兴趣多样化的正交正则化

7.预训练和下游微调任务以实现可转移的推荐

![image-20240313140047002](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240313140047002.png)

a.高效fine-tune直接预测下一个item,反正就是一句话:哪里不用冻哪里.

$\mathcal{L}_{fine-tune}=\frac{1}{B}\sum^B_{i=1}[t_i^{S-I}+\frac{\gamma}{M_i^2}\sum^{M_i}_{m,m'=1}<\xi_m,\xi_{m'}>] \space\gamma>0$

b.冷启动目标域的归纳迁移

$P_{r_{ind}}(I_{T_{i+1}}|I_1,I_2,...,I_{T_{i}})=softmax(<u_i,v_{T_{i+1}}>)$​

c.暖启动目标域的直推迁移

$P_{r_{trd}}(I_{T_{i+1}}|I_1,I_2,...,I_{T_i})=softmax(<\bar{u_i},v_{T_{i+1}}+{z_{T_{i+1}}}>)$​

$z_{T_{i+1}}$是$I_{T_{i+1}}$的ID emb,$\bar{u_i}$是item序列ID的encode序列表征







