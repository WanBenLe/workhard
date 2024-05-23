AttentionRank: Unsupervised keyphrase Extraction using Self and Cross Attentions, Haoran Ding, EMNLP, 2021

AttentionRank:无监督关键词抽取

1. BERT自注意力机制

2. 层次注意检索机制HAR

![image-20230504175450084](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20230504175450084.png)

AttentionRank计算候选人的累积自注意力和交叉注意力并以对重要性进行排名,步骤为:

1.使用Part-of-Speech (PoS)进行词性分析,然后用NLTK生成名词短语,成为候选集,为了避免高通用短语带来的问题,去除词频大于$df_{\theta}$的短语

2.累积自注意力:

词层面自注意力:候选词在句中与其他词的注意力的和

$$a_{w}=\sum_{w^{'} \in s}a_{w^{'}w}$$

句层面的自注意力是每个词层面的和

$$a^{c}_{i}=\sum_{w \in c}a_{w}$$

文档层面自注意力是句层面的和

$$a_{c}=\sum_{i \in d}a^{c}_{i}$$

3.交叉注意力

从预训练BERT得到候选集的表示$E^{c}=\{e^{c}_{1},...,e^{c}_{m}\}$,和句表示$E^{i}=\{e^{i}_{1},...,e^{i}_{n}\}$

交叉注意力计算候选集表示和句表示间的相似矩阵

$$S=E^{i}\cdot E^{c\top}$$

句到候选集和候选集到句的相似如下所示

$$\bar{S}_{i2c}=softmax_{row}(S)$$

$$\bar{S}_{c2i}=softmax_{col}(S)$$

词层面的交叉注意力权重如下所示

$$A_{i2c}=\bar{S}_{i2c}\cdot E^{c}$$

$$A_{c2i}=\bar{S}_{i2c}\cdot\bar{S}^{\top}_{c2i}\cdot E^{i}$$

新的句子表示如下所示

$V^{i}=AVG(E^{i},A_{i2c},E^{i} \odot  A_{i2c},E^{i} \odot  A_{c2i})$

仍旧使用自注意力并求平均得到交叉注意力的重要性

$I=softmax_{row}(V^{i} \cdot V^{i\top}) \cdot V^{i}$

$\alpha^{i}=AVE(I[:,i])$

文档维度使用句维度的值有,$E^{d}=\{\alpha^{1},...,\alpha^{i}\}$,类似的有

$P=softmax_{row}(E^{d} \cdot E^{d\top}) \cdot E^{d}$

$p^{d}=AVE(P[:,i])$

同样的,候选集有,$E^{c}=\{e^{c}_{1},...,e^{c}_{m}\}$

$C=softmax_{row}(E^{c} \cdot E^{c\top}) \cdot E^{c}$

$p^{c}=AVE(C[:,i])$

最后计算候选集和文档的cos相似度

$$r_{c}=\frac{p^{c}\cdot p^{d}}{||p^{c}||\cdot ||p^{d}||}$$

3.最终得分

首先对$a_{c}$和$r_{c}$进行标准化处理,然后使用以下方法得到得分$s_{c}$,$d\in [0,1]$,后者需要微调,作者取0.8

$s_{c}=d*a_{c}+(1-d)*r_{c}$

超参数:

a.累积的自我注意值的贡献高于交叉注意相关性,然而最优的$d$得微调

b.累积的自我注意模型通过自我注意权重在文档上的累积,隐含地考虑了关键短语的重复.

c.对于短文档集,由于一份文件中只有几句话,所以这个词的重复率很低,交叉注意力相关性的影响更大,但仍需要强调关键短语和句子与文档之间的上下文相关性

d.对于短文档数据集,最好的$df_{\theta}$通常很小.长文档数据集的$df_{\theta}$相对较大.在$df_{\theta}$大于某个值之后,性能随着$df_{\theta}$的增加而下降,这意味着具有较大$df_{\theta}$的术语可以是特定语料库中文档的关键短语.
