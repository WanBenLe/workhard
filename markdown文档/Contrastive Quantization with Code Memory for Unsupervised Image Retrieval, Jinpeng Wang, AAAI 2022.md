#### Contrastive Quantization with Code Memory for Unsupervised Image Retrieval, Jinpeng Wang, AAAI 2022
对于大规模检索系统,基于量化和二进制的hash储存方法(例如说milvus用二进制存储+汉明距离等计算与L2-Norm存储后IP)相对更省内存和高效.

前人一摞工作专注在对原始模型上量化和重建损失在原始预训练模型里,因而极度依赖预训练模型的质量(而上篇分享的TransEncoder用两个模型就是因为模型可能有bias之类的)

具体的挑战有三个和本文新方法MeCoQ的解决方案:

1. 抽样偏差:对比学习如果走无监督,正样本可能会被当成负样本:使用debias框架

2. 模型退化:对比学习中量化的codewords会很接近,进而降低了表征能力:codewords正则化

3. 性能冲突:batch越大效果越好,然而显存越大:新的量化代码存储库

##### 模型框架:

对于$N_D$张图片,将每张图片从展平的P维映射到的B bit的二进制表示.

1.对于抽样的图片X,使用T和T'获得两张相关图

2.对图片进行进行嵌入

3.对嵌入进行量化得到二进制表示,这是正例

4.其它图片的嵌入的二进制就是负例

5.在4之外维护一个先进先出的量化表示存储库M,作为额外负例(之前文本对比学习也维护了这玩意去做K-means保持类内信息,之类的),M是不参与推理的

![image-20231105155803367](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231105155803367.png)

为了将基于聚类的codeword可以反向传播玩DL,有以下可训练量化方案:

原本的codebook $C=C^1*C^2*...*C^M$的每段都量化了一部分内容,假设存在z使得等分成M等长d(d=D/M)维的,可以进行如下所示的归一.Attention之后近似$\alpha$-softmax的操作相关的近似可微分化操作很多,之前也分享过的ApproxNDCG也是用了可微分化的近似处理:

$z^m=z^m/||z^m||_2,c_i^m=c_i^m/||c_i^m||_2$

$\hat{z}^m=Attention(z^m,C^m,C^m)=\sum_{i=1,K}p_i^mc_i^m$

$p_i^m=sortmax_\alpha(z^{m\top}c_i^m)=\frac{exp(\alpha z^{m\top}c_i^m)}{\sum_{j=1,K}exp(\alpha z^{m\top}c_j^m)}$

$p=concatnate(p^1,p^2,...,p^M)$

$\hat{z}=concatnate(\hat{z^1},\hat{z^1},...,\hat{z^M})$

接下来是个debias的DCL损失函数

$L_{DCL}=-\sum_{q=1,2N}log{\frac{exp(s_{q,k+}/\gamma)}{exp(s_{q,k+})/\gamma+N_{In-Batch}}}$

$N_{In-Batch}=\sum_{k^-=1,k^- \notin\{q,k^+\} ,2N}[\frac{exp(s_{q,k-}/\gamma)}{1-\rho^+}-\frac{\rho^+exp(s_{q,k-}/\gamma)}{1-\rho^+}]$

正则化防止模型退化,分析为啥可以防止模型退化我摸了,感兴趣的可以看原文

$\Omega_C=\frac{1}{MK^2}=\sum_{m=1,m}\sum_{j=1,K}\sum_{j=1,K}c_i^{m\top}c_k^m \le \epsilon$

实际上,可微分的软量化方案导致了以下定义下更少的特征漂移:

$Drift(X',t;\Delta t)=1/|X'|\sum_{x\in X'}||h(x;\theta^t_h)-h(x;\theta^{t-\Delta t}_h)||_2^2$

内存增强训练:预热之后存$N_M$进内存库,训练的时候走量化模块得到$\hat{z}$,因为是作为负例,跟$N_{In-Batch}$有点小像

$N_{Memory}=\sum_{i=1,N_M}[\frac{exp(s_{q},M_i/\gamma)}{1-\rho^+}-\frac{\rho^+exp(s_{q,k+}/\gamma)}{1-\rho^+}]$

最后结合到$L_{DCL}$有完整的损失函数

$L_{MeCoQ}=-\sum_{q=1,2N}log{\frac{exp(s_{q,k+}/\gamma)}{exp(s_{q,k+})/\gamma+N_{In-Batch}+N_{Memory}}}$

把正则化和嵌入的加上,就有了完整的学习目标

$min_{\theta_h,C}\quad E\quad L_{MeCoQ}+\beta||\theta_h||_2^2+\gamma\Omega_{C}$

编码与检索

数据库进行以下硬编码后聚合

$i_{db}^m=argmax_{1 \le i \le K} z_{db}^{m^{\top}}c_i^m ,\hat{z}^m_{db}=c^m_{i^m_{db}}$

对于query,可以预先建立一个查找表,符号不好打,是预计算的和codewords的相似度,进而可以计算AQS非对称量化相似度,$i^m_{db}$是第m codebook的codeword的index值

$AQS(x_q,x_{db})=\sum_{m=1,M}z_q^{m^\top}c^m_{i^m_{db}}/||z_q^m||_2$