#### LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models, Yixiao Li, ICLR2024

在预训练模型同时上量化和LoRA,为了减少全精度模型和量化模型的能力并提高下游任务泛化能力,提出了能找到低秩初始化的LoftQ(LoRA-Fine-Tuning-Aware Quantization).

##### 量化

$X^{INT}=round((2^N-1)F(X^{HP}))$

F是归一化函数,一般用max-min归一化$F(X)=(X-X_{min})/(X_{max}-X_{min})$

NF4归一化(4-bit NormalFloat Quantization)假设$X\sim N(0,\sigma^2)$

$F(X)=\Phi(X/\sigma)$,$\Phi$​是正态分布函数的累积分布函数

矩阵的近似量化:$q_N(\cdot):R^{m*n}\rightarrow R_N^{m*n},R_N:\{\Gamma[i]\in R|0\le i < 2^N\}$

##### 反量化

$\Gamma[i]=F^{-1}\frac{i}{2^N-1},i=0,1,...,2^N-1$

$X^D=\Gamma[X^{INT}]$

##### 低秩自适应LoRA

LoRA更新两个附在冻结的预训练权重矩阵W上的小矩阵A. B

$Y=XW+XAB^\top,r\ll min\{d_1,d_2\}$

初值设置为$A\sim N(0,\sigma^2),B=0$,但是走量化$Q=q_N(W)$之后$Q+AB^\top$会有误差

##### 交替优化LoftQ的算法

LoftQ通过联合优化N-bit量化权重Q. A. B目标来初始化网络,F是Frobenious范数,大概就是用Q作为有效近似,进而最小化量化后的误差

$min_{Q,A,B}||W-Q-AB^\top||_F$

输入:预训练权重WW, 目标秩r,N-bit量化函数,交替步长T(T=1可能就足够了,也可以试试T=5,实验表明对T不太敏感)

初始化:$A_0= 0,B_0= 0$

for i in range(T):

​	$Q_t=q_N(W-A_{t-1}B_{t-1}^\top)$

​	$A_t,B_t=SVD(W-Q_t)$

输出$Q_T,A_T,B_T$

##### LoRA fine-tune

冻结整数权重M,使用类似于AdamW的优化器优化低秩自适应,前向传播时使用$\Gamma$进行反量化