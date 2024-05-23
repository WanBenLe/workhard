Optimal binning: mathematical programming formulation, Guillermo Navas-Palencia

最佳分箱是在给定离散或连续数值目标的情况下将变量最佳离散化为分箱. 提出了解决了前人未解决的约束可扩展的.引入凸混合整数规划公式(包括ILP整数线性规划和MIQP混合整数二次规划)来解决二元/连续/多类目标类型的最佳分箱问题.

分箱是一种将连续变量的值离散化为分箱的技术,可以解决缺失值/异常值/统计噪声/数据缩放的问题,增强对变量与给定目标之间非线性依赖性的理解,进而降低模型复杂性,并后续可用于执行数据转换.

分箱首先生成初始细粒度离散化的预分箱,随后根据给定的约束细分箱.

预分箱生成严格单调的m个分割和n=m+1个分箱$s_1<s_2<...<s_m$,分箱为$(-\infin,s_1),...,[s_1,s_2),...,[s_m,\infin)$

n个预分箱的细分箱的决策变量由初始值为对角矩阵的下三角矩阵组成,$X_{ij}\in\{0,1\},\forall(i,j)\in \{1,...,n:i\ge j\}$

![image-20240109142134618](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240109142134618.png)

细分箱基础约束有三个

1.只能合并不能删除

$\sum_{i=1}^nX_{ij}=1,j=1,...,n$

2.只能合并连续的分箱

$X_{ij}<X_{ij+1},i=1,...,n;j=1,...,i-1$

3.存在最后一个分箱囊括了到无穷的范围

$\exist [s_k,\infin), k\le n,X_{nn}=1$

二元目标的混合整数规划

非事件NE(y=0)标化计数$p_i$和事件E(y=1)标化计数$q_i$对于分箱i有下式,r是对应的数量

$p_i=\frac{r_i^{NE}}{r_T^{NE}},q_i=\frac{r_i^{E}}{r_T^{E}}$

有反比的WoE(证据权重)和D(事件率),对数项是odds

$WoE=log(\frac{r_T^{E}}{r_T^{NE}})-logit(D_i)$

$D_i=(1+e^{WoE_i-log(\frac{r_T^{E}}{r_T^{NE}})})^{-1}$

有无界的Jereys散度即信息价值IV,Jensen-Shannon距离$JSD\in[0,log(2)]$

$J(P||Q)=IV=\sum_{i=1}^{n}(p_i-q_i)log(\frac{p_i}{q_i})$

$JSD(P||Q)=0.5*(D(P||M)+D(Q||M)),M=0.5*(P+Q)$

好的分箱有三个属性

1.缺失值单独成箱

2.每箱至少5%观测值

3.不存在0 NE/E计数的分箱

$V_{ij}=(\sum^{i}_{z=j}\frac{r_z^{NE}}{r_T^{NE}}-\frac{r_z^{E}}{r_T^{E}})log(\frac{\sum^{i}_{z=j}r_z^{NE}/r_T^{NE}}{\sum^{i}_{z=j}r_z^{E}/r_T^{E}}),i=1,...,n;j=1,...,i$

基本优化问题由最大化IV,总IV可以用按行加和得到:

$max_{X}\sum_{i=1}^nV_{ii}X_{ii}+\sum_{j=1}^{i-1}(V_{ij}-V_{ij+1}X_{ij})$

$s.t. \sum_{i=j}^nX_{ij}=1,j=1,...,n$

$X_{ij}-X_{ij+1}\le0,i=1,...,n;j=1,...,i-1$

$d+\sum_{i=1}^nX_{ii}-b_{max}=0,0\le d\le b_{max}-b_{min}$(增加了稀疏性处理)

$r_{min}X_{ii}\le\sum_{j=1}^ir_jX_{ij}\le r_{max}X_{ii},i=1,...,n$

$r^{NE}_{min}X_{ii}\le\sum_{j=1}^ir_j^{NE}X_{ij}\le r_{max}^{NE}X_{ii},i=1,...,n$

$r^{E}_{min}X_{ii}\le\sum_{j=1}^ir_j^{E}X_{ij}\le r_{max}^{E}X_{ii},i=1,...,n$

$X_{ij}\in\{0,1\},\forall(i,j)\in \{1,...,n:i\le j\}$

单调性约束:使用了$\beta$作为分箱间的事件发生率$D_i$的差异控制

升序约束

$D_{zz}X_{zz}+\sum_{j=1}^{z-1}(D_{zj}-D_{zj+1})X_{zj}+\beta(X_{ii}+X_{zz}-1) \\\le 1+(D_{ii}-1)X_{ii}+\sum_{j=1}^{i-1}(D_{ij}-D_{ij+1})X_{ij},i=2,...,n;z=1,...,i-1$

降序约束

$D_{ii}X_{ii}+\sum_{j=1}^{i-1}(D_{ij}-D_{ij+1})X_{ij}+\beta(X_{ii}+X_{zz}-1) \\\le 1+(D_{zz}-1)X_{zz}+\sum_{j=1}^{z-1}(D_{zj}-D_{zj+1})X_{zj},i=2,...,n;z=1,...,i-1$

凹型约束

$a=(D_{ii}X_{ii}+\sum_{z=1}^{i-1}(D_{iz}-D_{iz+1})X_{iz})$

$b=(D_{jj}X_{jj}+\sum_{z=1}^{j-1}(D_{jz}-D_{jz+1})X_{jz})$

$c=(D_{kk}X_{kk}+\sum_{z=1}^{k-1}(D_{kz}-D_{kz+1})X_{kz})$

$i=3,...,n;j=2,...,i-1;k=1,...,j-1$

$-a+2*b-c\ge X_{ii}+X_{jj}+X_{kk}-3$

凸型约束

$a-2*b+c\ge X_{ii}+X_{jj}+X_{kk}-3$

右边部分保证了冗余约束

峰(倒U)型约束

$y_i+y_z+1+(D_{zz}-1)X_{zz}+\sum_{j=1}^{z-1}(D_{zj}-D_{zj+1})X_{zj}\\\ge D_{ii}X_{ii}+\sum_{j=1}^{i-1}(D_{ij}-D_{ij+1})X_{ij},i=2,...,n;z=1,...,i-1  $

$2-y_i-y_z+1+(D_{ii}-1)X_{ii}+\sum_{j=1}^{i-1}(D_{ij}-D_{ij+1})X_{ij}\\\ge D_{zz}X_{zz}+\sum_{j=1}^{z-1}(D_{zj}-D_{zj+1})X_{zj},i=2,...,n;z=1,...,i-1  $

谷(U)型约束

$y_i+y_z+1+(D_{ii}-1)X_{ii}+\sum_{j=1}^{i-1}(D_{ij}-D_{ij+1})X_{ij}\\\ge D_{zz}X_{zz}+\sum_{j=1}^{z-1}(D_{zj}-D_{zj+1})X_{zj},i=2,...,n;z=1,...,i-1  $

$2-y_i-y_z+1+(D_{zz}-1)X_{zz}+\sum_{j=1}^{z-1}(D_{zj}-D_{zj+1})X_{zj}\\\ge D_{ii}X_{ii}+\sum_{j=1}^{i-1}(D_{ij}-D_{ij+1})X_{ij},i=2,...,n;z=1,...,i-1  $

如果确定了变化点的位置,可以转化为规模更小的两个单调性约束

控制分箱数量

非凸MINLP控制std

$max_{X,\mu,w}  \sum_{i=1}^nV_{ii}X_{ii}+\sum_{j=1}^{i=1}(V_{ij}-V_{ij+1})X_{ij}-\gamma t$

$extra \ \ \  s.t. \sum_{i=1}^n w_i^2\le (m-1)t^2$

$u=\sum_{i=1}^n\sum_{j=1}^nr_jX_{ij}/m$

$w_i=\sum_{j=1}^ir_jX_{ij}- \mu X_{ii}$

$m=\sum_{i=1}^nX_{ii}$

$m\ge 0$

$\mu \ge 0$

$w_i \in R$

非凸MINLP控制HHI(没有额外约束)

$max_{X}  \sum_{i=1}^nV_{ii}X_{ii}+\sum_{j=1}^{i=1}(V_{ij}-V_{ij+1})X_{ij}-\frac{\gamma}{r^2_T} \sum_{i=1}^n( \sum_{j=1}^ir_jX_{ij})^2 $

非凸MINLP控制最大最小分箱差异

$max_{X,p_{min},p_{max}}  \sum_{i=1}^nV_{ii}X_{ii}+\sum_{j=1}^{i=1}(V_{ij}-V_{ij+1})X_{ij}-\gamma (p_{max}-p_{min})$

$extra \ \ \  s.t. p_{min}\le r_T(1-X_{ii})+\sum_{j=1}^ir_jX_{ij}$

$p_{max}\ge \sum_{j=1}^ir_jX_{ij}$

$p_{min}\le p_{max}$

$m=\sum_{i=1}^nX_{ii}$

$p_{max}\ge 0$

$p_{min} \ge 0$

最大化p值约束

$R^{NE}_{ij}=\sum_{z=j}^ir_z^{NE},R_{ij}^E=\sum_{z=j}^ir_z^E,i=1,...,n;j=1,...,i$

![image-20240109163337250](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240109163337250.png)

混合整数规划局部启发式搜索重构

引入对角线x,0累加器a,合并数量z减少优化维度

![z](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240109164025289.png)