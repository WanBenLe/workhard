#### Angle-Optimized Text Embeddings, Xianming Li, 2023

由于优化目标的cos函数在值接近1和-1的存在饱和区,梯度很小容易导致梯度消失,提出了AngIE,在复空间引入角度优化.大多数对比学习模型都在无监督下使用,难以在监督环境中受益,而对于2进制标签就很容易落入饱和区.

本文提出角度优化文本emb,首先将文本emb划分为实部和虚部,然后计算复空间的归一化的角度差作为优化目标.

长文本STS数据集:GitHub Issues 收集的,大约有 21K 个样本,我们使用重复问题作为正样本,将非重复问题作为负样本.

##### 模型架构

###### 输入层

###### 填充并生成句子的d维词嵌入,concat得到句嵌入作为input,然后走BERT/RoBERTa/LLaMA得到文本表征X.

###### Cosine目标

###### 高相似对的cos大于低相似对的cos

$L_{cos}=log[1+\sum_{s(X_i,X_j)>s(X_m,X_n)}e^{\frac{cos(X_m,X_n)-cos(X_i,X_j)}{\gamma}}]$

###### 批内负目标

###### 正样本是数据增强得到的有监督样本.试图识别可能重复但没被标为正样本的句子以减少错误负样本的噪声.

负样本生成的prompt

"You are a highly smart same-meaning/opposite-meaning sentence-generating system. Your job is to generate {size} synonymous/antonym sentences of a given input sentence. Input sentence: {text}. Output:” to generate positive/negative pairs."

$L_{ibn}=-\sum_{b}\sum_{i}log[\frac{e^{cos(X_{b_i},x^+_{b_i})/\gamma}}{\sum_j^N{e^{cos(X_{b_i},x^+_{b_j})/\gamma}}}]$

###### 角度目标

###### chunk分块将emb用复数表示并获得实部$X^{re}$与虚部$X^{im}$,最小化具有高相似度的对与低相似度的对的归一化角度差异

$z=a+bi \in C, w=c+di \in C$, a. b. c. d都是实数,进而有:

$\Delta \theta_{zw}=(\theta_z-\theta_w)=\frac{z}{w}/\gamma=abs[\frac{(ac+bd)+(bc-ad)i}{\sqrt{(c^2+d^2)(a^2+b^2)}}]$

$L_{angle}=log[1+\sum_{s(X_i,X_j)>s(X_m,X_n)}e^{\frac{\Delta \theta_{ij}-\Delta \theta_{mn}}{\gamma}}]$

$L=\omega_1+L_{cos}+\omega_2+L_{ibn}+\omega_3+L_{angle}$



温度参数0.05角度目标的是1,110m的uncased BERT base