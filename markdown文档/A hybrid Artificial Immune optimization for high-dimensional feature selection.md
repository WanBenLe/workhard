#### A hybrid Artificial Immune optimization for high-dimensional feature selection, Knowledge-Based Systems, Yongbin Zhu, 2023
提出Fisher滤波+柯西算子+精英个体致死变异+变异/更新的自适应调整因子的克隆选择算法HFSIA用于特征筛选.

##### 基于搜索策略的特征选择方法

1.全局最优搜索策略:效率低限制多

2.随机搜索策略:容易陷入局部最优

3.启发式搜索策略:依赖于问题

##### 克隆选择算法CSA

元启发法更容易得到最优解,混合算法结合可以得到开发质量和搜索质量的均衡,其中有基于人工免疫算法的CSA,步骤如下

1.抗体初始化:生成一组候选解决方案.

2.亲和力评估:计算抗体池每种抗体的亲和力. 

3.选择和克隆:选择亲和力最高的n个抗体,并按照与抗原的亲和力成比例克隆这n个抗体,形成克隆组C. 

4.高频突变:对克隆群体进行高频突变,并产生成熟的抗体群体D.

5.群体更新:重新选择D,形成记忆细胞集M.生成d 个新抗体来替换P 中亲和力较低的抗体. 

重复步骤2-5到满足终止条件.

##### HFSIA的解码与编码

抗体基因是1\*d的二元向量,0代表特征没被选择,d是特征数量

##### HFSIA的适应度

$fitness=\omega*E_r+(1-\omega)*p/q$

$\omega$是(0,1)的权重,$E_r$是分类错误率,p/q是特征筛选率

$fitness=\omega*E_r+(1-\omega)*\sqrt{(b-p/q)^2}$

尝试改动有上式,b是目标的特征筛选率

##### HFSIA的初始化种群

柯西分布比正态分布有更大的取值范围和突变步长,使用标准柯西分布($x_0=0,\gamma=1$)

$f(x,x_0,\gamma)=\frac{1}{\pi}[\frac{\gamma}{(x-x_0)^2+\gamma^2}]$

$F(x,x_0,\gamma)=\frac{1}{\pi}arctan(\frac{x-x_0}{\gamma})+0.5$

使用标准柯西分布生成初始化种群,大于阈值$\eta=-0.2$就选择,否则不选择

##### HFSIA的变异与更新策略

对精英抗体进行致死突变,对个体有害但是有利维持群体杂合状态.并赋予了与当前迭代次数完成比相关的

1. 克隆适应度排名前列的$n=c_r*N$个精英个体

2. 当$t_0$+k的适应度低于$t_0$时,flag=1

3. flag=1时计算自适应线性加速因子$\sigma=0.5-\frac{t}{T_{max}}$

4. $\sigma$为阈值基于标准柯西分布生成致死基因列
5. 让精英满足致死基因列的置0.

扩大种群规模,引入调整因子$\theta=\frac{t}{T_{max}}$

Fisher分数:第i特征不同分类的方差和第k特征同分类的方差

$F_k=\frac{\sum_{k=1}^Cn_k(u^i_k-u^i)^2}{\sum_{k=1}^Cn_k(\sigma^i_k)^2}$



##### HFSIA特征筛选算法

输入:训练数据$orData$,抗体组数量$N$,最大迭代次数$T_{max}$,克隆抗体的选择率$c_r$

输出:特征筛选子集$R_s$

初始化算法参数$T_{max}=50,N=10,c_r=0.5,\eta=-0.2,\omega=0.99,T_{Fisher}=200$

计算Fisher分数并应用Fisher滤波得到特征子集$ftD$

$P \leftarrow GeneratePop(N,ftD)$生成初始种群

for i in range(T_max):

​	$Fitness \leftarrow FitnessFun(P)$得到适应度函数

​	sort(p),根据fitness排序

​	筛选并保留了最佳的N个样本

​	$bestAb_i,C\leftarrow  Select(P,c_r)$筛选$n=c_r*N$个精英个体

​	$C$基于突变策略进行更新

​	使用调整因子$\theta=\frac{t}{T_{max}}$生成补充种群$P'$

​	$P \leftarrow \{bestAb_i\cup C \cup P'\}$	

$R_s=best(P)$

