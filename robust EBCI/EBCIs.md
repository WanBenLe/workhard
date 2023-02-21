Robust Empirical Bayes Confidence Intervals, Timothy B. Armstrong, ECTA, 2022

放个关系略小但近来撸了一遍的EM算法.

1.假设服从A分布,A分布参数服从B分布

2.固定初始参数计算期望

3.固定期望更新参数

4.合法性校验迭代出最终参数,合理早停

5.该bootstrap的就bootstrap做区间估计.



给定样本X服从某分布,有以下经验贝叶斯EB的参数估计方法

1.计算边缘分布,独立重复抽样之后上矩估计,然后用后验分布估计参数

2.计算边缘分布,独立重复抽样之后写极大似然函数上第二类极大似然

3.不管后验分布用先验分布算近与边缘分布有关的后验期望

EB区间覆盖率:

在bootstrap的时候独立重复抽样估计参数和CI,CI cover真实参数的占比,如果分布合理应当有95%的覆盖率,而实际上对于先设~N()的而不满足的前提下区间覆盖率会偏低.

本文提出了修正CI使得平均覆盖率增加,文章特别说明如果focus这最大最小之类的参数.真实效应.或者效应分布上是不合适的.



对于数据$\{Y_{i},X_{i},\hat{\sigma_{i}}\}_{i=1}^{n}$

robust EBCI:

$\hat{\theta_{i}}\pm cva_{\alpha}(\hat{\sigma_{i}^{2}}/\hat{\mu_{2}},\hat{k})\hat{\omega_{EB,i}}\hat{\sigma_{i}}$

其中

$cva_{\alpha}(\hat{\sigma_{i}^{2}}/\hat{\mu_{2}},\hat{k})=\rho^{-1}(m_{2,i},\kappa,\alpha) =(\sup_{F}E_{F}[r(b,\alpha)])^{-1}$

s.t.

$E_{F}[b^{2}]=\hat{\sigma_{i}^{2}}/\hat{\mu_{2}}$

$E_{F}[b^{4}]=\hat{k}(\hat{\sigma_{i}^{2}}/\hat{\mu_{2})}^2$

有

$\hat{\omega_{EB,i}}=\frac{\hat{\mu_{2}}}{\hat{\mu_{2}}+\hat{\sigma_{i}^{2}}}$

​	此处$cva$代表的是一个区间覆盖的临界值,-1表逆运算,实际上为了得到复杂$cva$有亿丢丢数学上的细节在附录B,给了$cva$是如何转化为一个优化问题的,并给了一些解析解,实际上会取数值解的结果如果比较离谱就上线性规划看看是不是鲁棒的.

![image-20230131201134833](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20230131201134833.png)

实际情况是因为有MATLAB源代码,所以结果挺好复现的,不带支撑材料的在优化算法的实现上对结果影响太大的...(特别是没有写出雅克比矩阵或者是标准型上cvxopt之类的)

![image-20230131202623177](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20230131202623177.png)

然后求最优的,也是个求权重的最优化问题

$\frac{1}{n}  \sum P(\theta_{i} \in [\omega_{EB}Y_{i}\pm cva_{\alpha}({\sigma_{i}^{2}}/{\mu_{2}})\omega_{EB}\sigma ]|\theta) \ge 1-\alpha$

技术细节在之前之前已经分享过了,有兴趣的自行zai.