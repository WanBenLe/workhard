#### Infobatch : Lossless Training Speed Up by Unbiased Dynamic Data Pruning, Ziheng Qin, ICLR 2024

提出根据损失分布随机修剪一部分信息量较少的样本,并重新调整剩余样本的梯度以近似原始梯度的O(1)时间复杂度的InfoBatch以解决数据修剪的梯度期望偏差问题,并可兼容核心集选择.

##### 一.静态剪枝

数据集$D=\{z_i\}|^{|D|}_{i=1}=\{x_i,y_i\}|^{|D|}_{i=1}$对于每个样本有丢弃概率$P(Z;H)\in\{0,1\}$依赖于分数$H(z)$,其中$\bar{H}$是阈值

$P(Z;H)=1(H(z)<\bar{H})$​

##### 二.动态剪枝

$H_t$在每个训练step的时间状态$t$里改变,概率依赖于步长$P_t(z)=P(z;H_t)$

由于低分样本容易重复,修剪减少梯度更新数量,大规模数据的排序开销大因此仍有问题

##### 三.InfoBatch剪枝策略

1.剪枝概率$P_t(z)$

$P_t(z)=r,H_t(z)<\bar{H_t}$

$P_t(z)=0,H_t(z)>\bar{H_t}$

$r\in\{0,1\}$是超参数剪枝概率,$\bar{H_t}$​是分数均值,不需要排序因而从$O(logN)$降到$O(1)$

2.使用loss作为分数$H_t(z)$

3.对于t>0的epoch,剪枝后样本分数不变,没有剪枝的样本分数变成最新的loss$H_{t+1}(z)=L(z)$

4.第一个epoch如果没有loss则初始化为1

5.对于没有被剪枝的$H_t(z)<\bar{H_t}$样本,梯度扩大为$1/(1-r)$倍维持梯度期望

6.剪枝只在前$\sigma \cdot C$epoch进行,$\sigma\in (0,1)$是超参,C是总epoch数,后续使用完整训练保持性能

