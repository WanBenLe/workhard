#### DeBERTav3: Improving DeBeRTa Using Electra-Style Pre-Training With Gradient Disentangled embedding Sharing, Pengcheng He, ICLR 2023

RTD进行MLM生成的二分类鉴别提高共享嵌入性能和引入梯度解缠嵌入共享GDES解决MLM和RTD的优化冲突问题.

###### Transformer

多个block,每个block有一个multi-head attn和MLP,由于self-attn无法编码单词位置信息,为token emb传入绝对位置/相对位置的position_ids,NLP使用相对位置更好.

###### DeBERTa

1.解缠注意力DA:使用2个独立向量分别表征内容和位置,词语见attn-weight使用词语内容和相对位置用解缠矩阵计算

2.增强型掩码解码器:MLM decoder中添加上下文的绝对位置信息

###### Masked Language Model

$max_{\theta} \space log  \space p_{\theta}(X|\tilde{X})=max_{\theta}\sum_{i\in C}log \space p_{\theta}(\tilde{x_i}=x_i|\tilde{X})$

其中$X=\{x_i\}$是序列,$\tilde{X}$是被随机mask的t%的token,$\theta$是用于将$\tilde{x_i}$预测回$x_i$的语言模型,C是$X$中被mask的token 的索引集(10%mask token不变,10%用随机挑选token替换,其它用[mask]替换).

###### Electra-Style的DeBERTa-V3

生成器(MLM,参数G)生成模糊token并替换原序列得到新序列,并训练二分类鉴别器(RTD,参数D)预测token是被替换的标记还是原始输入.

$L_{MLM}=E(-\sum_{i\in C}log \space p_{\theta_G}(\tilde{x}_{i,G}=x_i|\tilde{X}_G))$

$\tilde{X}_G$是生成器的输入,被随机mask的15% token,鉴别器的输入通过生成器输出概率采样的新token替换mask后token来构建.

$\tilde{x}_{i,D}= \tilde{x}_{i} \sim p_{\theta_G}(\tilde{x}_{i,G}=x_i|\tilde{X}_G)), \space i \in C$

$\tilde{x}_{i,D}= {x}_{i} , \space i \notin C$​

$L_{RTD}=E(-\sum_{i}log \space p_{\theta_D}(1(\tilde{x}_{i,D}=x_i)|\tilde{X}_D,i))$​

$L=L_{MLM}+L_{RTD}$​

2.鉴别器和生成器共享token emb(不共享会导致性能下降),但引入梯度解缠嵌入共享GDES处理因为鉴别器和生成器的训练损失将标记嵌入拉向不同的方向的冲突优化问题.

![image-20240601152534082](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240601152534082.png)

$E_D=sg(E_G)+E_{\Delta}$

sg为停止梯度算子,$E_{\Delta}$使用0矩阵初始化,并使用NES流程训练:每次迭代用MLM为RTD生成输入,$L_{MLM}$更新$E_G.E_D$,然后在$L_{MLM}$使用$E_{\Delta}$更新$E_D$.训练后$E_{\Delta}$添加到$E_D$并保存结果矩阵.

