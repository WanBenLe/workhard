#### Matryoshka Representation Learning, Aditya Kusupati, ICLR2022

可能基于梯度训练的归纳偏差信息会扩散到整个表示向量,之前的低维表示学习方法1.训练维护开销高2.大量前向传递贵3.encode数据的多个副本的存储和内存成本4.即时维度选择昂贵或准确度显着下降

因此本文提出了O(log(d))嵌套显式优化低维向量的俄罗斯套娃表示学习MRL,学习从粗到细的表示,这些表示至少与独立训练的低维表示一样准确,并且可以跟HNSW(O(d log(N)))之类的图索引方法互补

新方法1.可以减少emb维度2.可以加速检索3.提高长尾少样本的分类效果4.允许跨模态训练

![image-20240131143224884](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240131143224884.png)

一个损失函数解决所有问题:

$min_{\{W^{(m)}_{m\in M}\},\theta_F}\frac{1}{N}\sum_{i\in [N]}\sum_{m\in M}c_M*L(W^{(m)}*F(x_i;\theta_F)_{1:m};y_i)$

不同维度$m$的表示接不同的稀疏线性分类层(参数化为$W^{(m)}$),并使用$c_M$对所有维度表示的loss加权求和得到新的loss.$c_M$可全设成1.如果还偷懒,稀疏线性分类层只接最大维度的,然后for循环取前X个维度计算得到加权loss,称为MRL-E(高效俄罗斯套娃表示学习)

对于每个嵌套维度都建议进行独立的单位归一化并使用L2检索(论文)



在ResNet50.ViT-B/16.ALIGN.BERT的表示学习.分类.自适应分类.检索.自适应检索甚至跨域都更佳



同时提出了漏斗检索方法:低维获取候选列表,在高维度重排,每次维度加倍,数量减半

例如说在8维返回前100万,在结果中16维返回50万,32维范围返回25万之类的



原论文提供的代码

![image-20240131165153888](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240131165153888.png)

![image-20240131165209329](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240131165209329.png)



