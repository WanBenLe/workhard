Instance Smoothed Contrastive Learning for Unsupervised Sentence Embedding, Hongliang He, AAAI 2023



对于unsup-SimCSE方法而言,使用自己双dropout做正例,其它做负例,但是会出现负例跟正例虽然文本不一样但其实很像例如说商品里的粉色apple和黑色apple,这样子会损害泛化性能

本文提出动态FIFO(先进先出)缓存Embedding并检索得到正Emb,然后基于KNN加权得到平滑的Emb

为了减少伪阴性对进行过滤的直接剔除可能会损失丰富的正标签,因此适用软标签的标签平滑方法缓解网络过度自信.



IS-CSE:

1.一个先进先出的内存缓冲区，用于在训练过程中保存前面步骤中的Emb

2.在构建正对时，我们根据余弦相似度从内存缓冲区中检索Emb

3.对正嵌入进行加权平均运算以获得平滑的嵌入

流程为

1.走ESimCSE得到句子Emb$h_i$,和他的正Emb$h_i^+$

2.给定动态缓冲区$B$,缓存最近L(这里取了1024)条$h_i^+$,这个缓存需要停止梯度

3.基于KNN和cossim得到最新数据集$\{B,h_i^+\}$的分类,并取出和最新正Emb$h_i^+$相近的Emb数据(模型用了KNN)

KNN: $h^{s+}=softmax(\frac{h^+K^T}{\beta})K$,K为同类的Emb(这里参数取了16,$\beta$取了2)

K-means: $h^{s+}=\gamma h^++(1-\gamma)c^+$,$c^+$为聚类中心

新损失是SimCSE Loss和平滑对比的Loss的加权和

$L=L_{Instance}+\alpha L_{smoothing}$

$\alpha=min\{cos(\pi *\frac{T_i}{T_max})*(\alpha_{strat}-\alpha_{end}),0\}+\alpha_{end}$ (论文用了0.005到0.05)

