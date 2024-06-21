#### NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models, Chankyu Lee, 2024

为了提高LLM作为通用emb的嵌入性能,提出了NV-Embed:

1. LLM的\<EOS\>token/mean pooling更好的潜在注意层来获得池化emb

2. 对比学习的时候移除了causal mask

3. 训练时两阶段对比指令调优,首先在检索数据集用指令进行对比学习(批内负样本+精选难负样本).二阶段将非检索数据及融合到指令调优(禁用批量负样本训练).

##### 双向注意力

LLM的causal mask虽然防止了信息泄露但是作为单向注意力限制了模型的表征能力,然而只需要在对比学习过程删除causal mask就能提高效果

##### 潜在注意力层

mean pooling会稀释关键短语重要信息,\<EOS\>emb会受到近因bias的影响

decoder最后一层的隐藏层改为query$Q \in R^{l*d}$,l是序列长度,d是隐藏层维度$K=V\in R^{r*d}$,r是可学习字典的潜在维度,$O=softmax(QK^{\top})V$,最后接正常的2层linear+GELU激活的MLP层,r=512,head=8,最后再一次mean pooling得到结果.

##### 两阶段指令调优

对于非检索任务而言,批内负样本可能是同类的,这种情况下批内负样本会影响非检索任务结果

因此一阶段使用指令对检索数据及对比学习,二阶段使用检索+非检索数据进行对比指令调优而不使用批内负样本

##### 样本构造

1. 微调了别的Encoder模型挖掘难负样本

2. 使用BM2.5硬阈值删除训练集和测试集内的高相似数据

3. mask了输出emb的指令标记(但是会影响结果)

4. 分类和聚类任务将label作为query结果