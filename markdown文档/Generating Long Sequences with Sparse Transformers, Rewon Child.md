#### Generating Long Sequences with Sparse Transformers, Rewon Child

Transformer的时间和内存复杂度与序列长度平方增长$O(n^2)$

1. 提出了attn矩阵的稀疏分解减少为$O(n\sqrt{n})$,原则上表明self-attn可用于足够长的seq.

2. 重新计算attn矩阵节省内存
3. 快速训练的attn核

考虑自回归序列生成的概率极大似然化

$p(x)=\prod_{i=1}^np(x_i|x_1,...,x_{i-1};\theta)$

![attn](D:\Download\主题词选品\attn.png)

首先可视化了visual模型的attn矩阵,发现是高度稀疏的,因此不需要每个token都和所有token都算attn,首先定义一种连接模式$S=\{S_1,...,S_n\}$,attn计算有

$Attend(X,S)=(a(x_i,S_i)))_{i\in\{1,...,n\}}$

$a(x_i,S_i)=softmax(\frac{(W_qx_i)K_{S_i}^\top}{\sqrt{d}})V_{S_i}$

$K_{S_i}=(W_kx_j)_{j\in S_i},V_{S_i}=(W_vx_j)_{j\in S_i}$

W是Q. K. V的权重.完整的self-attn是$S_i=\{j:j\le i\}$​,意思就是每个token都可以跟前面的所有token attend到.

有p个系数attn的分解的self-attn的第m个头让索引$A_i^{(m)}\subset \{j:j\le i\},S_i=A_i^{(m)}$,需要进行有效选择($\forall j\le i$,让i可以在最大长度p+1内attend到j的A)使得$|A_i^{(m)}| \propto n^{1/p}$​

##### 图像和音频之类的周期可以用b图的stride

$A_i^{(1)}=\{t,t+1,...,i\},t=max(0,i-l)$

$A_i^{(2)}=\{j:(i-j)\space mod\space l=0\}$

第一种的意思是每个token都可以attend到前l个token

第二种的意思是前整数倍间隔l的token

##### 文本使用fixed 

$A_i^{(1)}=\{j:floor(j/l)=floor(i/l)\}$

$A_i^{(2)}=\{j:j \space mod \space l\in\{ t,t+1,...,l\}\},t=l-c$

意思就是给定c得到多个分割用的token,每个token可以attend到自己和分割token之前的所有token

##### 使用

$attn(X) = W_p \cdot attend(X,S)$

1.每个残差块使用一种注意力类型,按顺序/超参数确定的比例交错$attn(X) = W_p \cdot attend(X,A^{r\space mod\space p})$

r 是当前残差块的索引, 是分解式注意力头的数量

2.合并注意力头:一个注意力头关注两个分解注意力头都会关注的像素位置

$attention(X) = W_p\cdot attend(X,\cup_{m=1}^{p} A^{(m)})$ 

3.$n_h$多头并行计算后concat

$attn(X)= W_p( attend(X,A)^{(i)})_{i\in\{1,...,n_h\}}$

且权重维度变为$1/n_h$倍

##### 用于扩展的架构修改

$embed(X,W_e)=(x_iW_e+\sum_{j=1}^{n_{emb}o_i^{(j)}W_j})_{x_i\in X}$

$H_0=embed(X,W_e)$

$a(H)=dropout(attn(norm(H)))$

$ff(x)=W_2f(W_1x+b_1)+b_2$,$W_1$是输入维度的4倍,使用$1/\sqrt{2N}$初始化缩放

取$f(X)=X\odot sigmoid(1.702\cdot x)$

$b(H)=dropout(ff(norm(H+a(H))))$​

$resblock(H)=a(H)+b(H)$

$H_k=H_{k-1}+resblock(H_{k-1})$

$y=softmax(norm(H_N)W_{out})$

在残差块加法结束时应用重新计算减少存储参数量的内存成本

步幅$k$的注意力可以转置矩阵计算局部窗口,固定注意力可以聚合后分块计算,softmax操作融合,上三角矩阵不计算,消除负bias.存网络权重/跨GPU求均值/Q和K单精度,半精度计算激活和梯度,动态的loss缩放.