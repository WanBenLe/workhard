#### EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything, Yunyang Xiong

一.摘要

使用Masked图像预训练从SAM图像编码器重建特征的轻量级SAMI,然后使用SAMI构建EfficientSAM

虽然SAM效果很好,但是图像编码器ViT-H很大有632 m参数,基于prompt的解码器才3.87 m.我们利用SAM的masked图像预训练模型(224\*224的Image Net+重建loss)SAMI生成轻量级ViT backbone. SAMI使用SAM生成嵌入(避免从图像块重建),然后再用轻量级编码器训练masked图像模型.然后再用SAM解码器finetune(有监督数据).这种方法可以将ViT-H变为Tiny/Small/Base等.

![image-20231214165753302](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231214165753302.png)

二.相关工作

SAM和VIT从略

知识蒸馏是一种在不改变深度学习模型架构的情况下提高其性能的技术,学生模型的学习由教师模型的硬标签和软标签监督,软标签可以用各种方法传递教师模型的更多信息.

蒸馏方法将表示学习和分类解耦. 解耦知识蒸馏将经典的KD损失分为目标类知识蒸馏和非目标类知识蒸馏两部分,提高了知识迁移的有效性和灵活性. 另一项工作是从中间特征转移知识,例如说模型中间的特征等.

masked图像预训练使用对比学习给定不同增强学习高相似度增加不变性,但是依赖于负采样和增强方法.

masked图像建模(MIM)使用去噪自动编码器和上下文编码器通过重建masked图像块学习图像表征.

三.方法

masked自动编码器MAE

MAE带transformer的编码器和解码器,使用图像token作为输入,并分组为未masked token和给定masked比率的masked token.前者用于编码器提取特征,后者用于作为学习目标并在MIM重建.MAE使用75%的高掩码率防止数据泄露.

利用SAM的masked图像预训练SAMI

将SAM的图像编码器潜在特征作为MAE重建目标.编码器将未masked特征得到潜在特征表示,解码器在masked图像的输出emb下重建masked的token表示.

Cross-Attention解码器

只有masked token 需要重建,而encoder的输出充当锚点.Cross-Attention解码器query使用masked token, key和value使用编码器的未masked特征和masked特征走MAE得到emb并排序为原始位置.

线性投影头

用于对齐,解决SAM图像编码器和MAE输出特征dim不一致的问题.

重建损失

每个迭代SAMI有SAM的图像特征提取和MAE的前向和后向传播.SAM图像编码器记为$f^{sam}(X)=f^{sam}(\{x_i\}_{i=1}^N)$MAE编码器的输出记为$g^e(\{X_i\}i\in U)$,解码器输出记为$g^d(\{X_i\}i\in M)$,线性投影头记为$f^h(X)=h^{\theta}(\phi (g^e\{X_i\} i \in U  \oplus g^d \{X_i\}i\in M))$,有目标重建损失,N为输出tokens的数量.

$L_{W_e,W_d,W_{\theta}}=\frac{1}{N}\sum_{j=1}^{N}||f^{sam}(x)-f^h(x)||^2$

SAMI得到EfficientSAM

预训练后使用SAMI训练得到的编码器作为EfficientSAM的图像编码器和SAM的默认mask解码器.并在SA-1B微调得到各种下游任务的结果.

四.实验信息

SAMI在ImageNet-1K上训练400epochs,随机resize crop成224\*224. Masked图像预训练后不是用label信息.用于重建特征的SAM图像编码器使用ViT-H.ViT使用MSE进行预训练.batchsize=4096,AdamW($\beta_1=0.9,\beta_2=0.95$),学习率$2.4*10^-3$,0.05权重衰减.线性衰减学习率预热40 epochs. 75%掩码率和8 transformer和512 dim.