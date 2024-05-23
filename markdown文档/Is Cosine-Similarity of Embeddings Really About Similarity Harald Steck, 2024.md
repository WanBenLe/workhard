###### Is Cosine-Similarity of Embeddings Really About Similarity? Harald Steck, 2024

$\large{cosSim(x,y)=\frac{<x,y>}{||x||||y||}}$

正则化会隐式控制最终emb的cosSim,使得后者在特定情况下是黑盒甚至是任意的.

考虑矩阵分解模型Matrix Factorization, MF

$X\in R^{n*p},X\simeq XAB^{\top},A.B\in R^{p*k},k\le p,AB^{\top} \in R^{p*p}$​

$X$是user-item矩阵.$\vec{b_i}$是B的第i个item的emb,$\vec{x_u} \cdot A$是user的已消费的item的emb$\vec{a_j}$的和.

这时候有4个需要关注的玩意

$(XAB^\top)_{u,i}=<\vec{x_u}\cdot A,\vec{b_i}>,cosSim(\vec{b_i},\vec{b_{i'}}),cosSim(\vec{x_u}\cdot A,\vec{x_{u'}}\cdot A),cosSim(\vec{x_u}\cdot A,\vec{b_i})$​

训练的时候有两种正则化方法,都是基于L2正则化的

$min_{A,B}||X-XAB^\top||^2_F+\lambda||AB^\top||_F^2$​

第一种情况是降噪,跟加dropout差不多,这种情况下预测精度更高

$min_{A,B}||X-XAB^\top||^2_F+\lambda(||XA||_F^2+||B||_F^2)$​

第二种情况本质是矩阵分解$X=PQ^\top,P=XA,Q=B$,跟权重衰减差不多

可知$ \forall R \in R^{k*k},\hat A R$和$\hat B R$跟目标解$\hat A,\hat B$是一样的可行解,R是旋转矩阵.

问题是对与第一种情况有:$ \forall D \in R^{k*k},\hat ADD^{-1}\hat B^\top$和第一个目标解$\hat A\hat B^\top$,D是对角矩阵

因此存在这么一种情况使得D对norm后的emb产生影响:

$\hat A^{(D)}:=\hat AD,\hat B^{(D)}:=\hat BD^{-1}$

$\hat A^{(D)}_{(norm)}:=\Omega_A X\hat AD,\hat B^{(D)}_{(norm)}:=\Omega_b \hat BD^{-1}$

$\Omega_A.\Omega_B$​是恰当的用于标化的对角矩阵



对第一个目标,在特殊情况下会导致item只与自己相关和与其他所有item都无关.而且余弦相似度的结果是任意而且是不唯一的.

对第二个目标而言结果是唯一的,但是无法保证item和user的这个独特的emb出的cos是最优的语义相似性



###### 缓解

训练的时候加层归一化

避免使用embedding space,而是投影回原始空间之后再算cosSim

使用$X\hat A \hat B^\top$​作为X的平滑版本且user在原始空间的emb然后算cosSim

先学emb再做归一化