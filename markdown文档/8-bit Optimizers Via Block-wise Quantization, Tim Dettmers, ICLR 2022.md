#### 8-bit Optimizers Via Block-wise Quantization, Tim Dettmers, ICLR 2022

提出使用分块量化.动态量化和稳定嵌入层的8 bit Adam优化器,用于减少训练参数的内存.

对于RNN之类的激活参数依赖的节省效果不佳.

##### 基础

###### 带动量的SGD和Adam

公式如下所示,$\beta_1.\beta_2$是平滑常数,$\epsilon$是小常数,$\alpha$是学习率
$$
Momentum(g_t,w_{t-1},m_{t-1})=\begin{cases} m_0=g_0,Init
\\m_t=\beta_1m_{t-1}+g_t
\\w_t=w_{t-1}-\alpha\cdot m_t
\end{cases}
$$

$$
Adam(g_t,w_{t-1},m_{t-1},r_{t-1})=\begin{cases} r_0=m_0=0,Init
\\m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\\r_t=\beta_2r_{t-1}+(1-\beta_2)g_t^2
\\w_t=w_{t-1}-\alpha\cdot m_t/(\sqrt{r_t}+\epsilon)
\end{cases}
$$

###### 量化函数

$Q^{map}:[0,2^k-1]\rightarrow D$有三步:

1.计算用于映射到D的归一化常数$N=max(|T|)$

2.找到输入tensor T/N最接近的对应值$q_i$

3.存储i和得到结果$T^Q$,反量化为:

$T_i^D=Q^{map}(T_i^Q)\cdot N=arg \space min_{j=0}^{2^n}|Q^{map}_j-T_i/N|$

###### 动态树量化

1.数据第一位存符号2.后续数据存指数大小3.设置为1的第一位的后续都用于:4.线性量化

可达$10^-7$​的精度

##### 8位优化器

###### 1.分块量化

输入张量划分为独立量化的$B=2048$​较小块输入tensor在8 bit优化器反量化回32 bit优化器更新后,将状态tensor分块且根据块最大值归一化,逐元素动态量化成8 bit后存储索引,反量化根据索引取出后反归一化,分块量化可以限制异常值,且每个块都能跨内核并行处理.

分块量化处理如下所示

$T^{Q}_{bi}=arg \space min_{j=0}^{2^n}|Q^{map}_j-T_i/N_b|\|_{0<i<B}$

###### 2.动态量化

Adam状态严格正删去动态量化树的符号位,并用于扩展动态树量化以缓解Adam在stage 2的剧烈变化.

###### 3.稳定嵌入层

使用Xavier初始化稳定嵌入层并在input_ids嵌入前进行层归一化,同时该层使用标准精度的优化器状态优化(原本是32/16 bit的就是32/16 bit优化器状态),用于减少NLP输入token不均匀分布导致的梯度方差.