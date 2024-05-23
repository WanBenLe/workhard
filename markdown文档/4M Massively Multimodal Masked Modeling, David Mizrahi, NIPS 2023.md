### 4M: Massively Multimodal Masked Modeling, David Mizrahi, NIPS 2023

4M多模态训练方案:跨输入输出模态的masked train一个统一的表示空间以实现可拓展性,4M的关键功能有:

1.可以执行不同视觉任务

2.对于新的下游任务/输入模式fine-tune效果好

3.可以作为生成式模型

将多模态输入通过特定的分词器映射到离散token的集合/序列统一表示空间,回避Transformer了对encoder和head的依赖,可以建模多模态和参数共享.4M通过将一部分label作为输入,其余masked作为目标以解耦输入和输出的token/模态的数量依赖来进行高效训练.利用可用的单模态数据构造伪标记网络生成跨模态对齐.



#### 1.模态标记化

将所有模态映射到序列或离散label集.跨模态被视为在不同映射列间的预测.

a. 使用RGB. 几何信息(深度.表面法线).语义信息(captions.语义分割.边界框.CLIP tokenized特征)的混合训练数据作为有关场景几何和语义的信息先验和表示形式引导.数据同时包括了密集视觉.稀疏序列和DL的特征图.由于数据的多样性使得生成式成为可能.

b. 伪label多模模态训练数据集:只需要RGB图像数据因而具有了扩展性.

c. Tokenization:基于特定模态的分词器进行离散化映射.标题captions和边框会用WordPiece,边界框会议能够Pix2Seq. RGB. 深度法线语义分割和CLIP特征会用VQ-VAE(向量量化变异的自编码器)进行Tokenization.

##### 伪标签细节

表面法线和深度:将图片转为最接近的32倍数并在768之下后使用DPT-Hybrid使用跨任务一致性和3D数据增强预测.

语义分割:使用SwinB backbone预测并取argmax得到标签.

边界框:ViTDet ViT-H预测并删除置信度低于0.6的边界框.

CLIP特征图:CLIP-B16模型的 ViT-B/16 视觉backbone去最后一层Transformer输出,可视化将H*W PCA到前三个主成分并归一为RGB.

##### Tokenization细节

captions和边界框的Tokenization(总token 30 k):转为seq预测问题.对每种情况使用 1000 个特殊token的分辨率对角坐标(即最小和最大 x 和 y 坐标)进行建模.首先根据对象到原点的距离对序列进行排序,并且每个角坐标分配单独标记,可以得到比Pix2Seq大四倍且不易受mask干扰的token.对于captions和特殊(带语义分割的那个标签)的边界框tokens走了WordPiece,确保标签出现在captions时会生成相同token.

密集模态的Tokenization:使用VQ-VAE.

使用30个diffusion decoders从头开始端到端的训练RGB.法线和深度的分词器,使用了每层由3残差连接和上/下采样块组成的4个上下层的UNet, 2个最小的上层和下层加self-attention.处理C\*4*4以加速.将14\*14的encode token和32 dim codebook concat得到 16C\*56\*56并上采样为32\*56\*56,然后在output reshape为C\*244\*244.使用1000 DDPM和线性噪声训练diffusion decoder,推理使用25 DDIM从decoder中采样

ImageNet-1K的伪标签train 100 epoch后在CC12M训练15 epoch,并在224*224,batchsize=256训练后fine-tune到448\*448

降低codebook的维度并对encode vec做L2Norm,迭代后计算批次中映射到给定codebook条目的编码向量的数量，并从batch随机替换指数移动平均值(EMA)计数小于指定阈值的任何code.阈值公式如下所示:

$T_{replace}=\frac{BN_{tolens}}{c_{replace}*N_{vocab}}$batchsize*图片token数除以超参和codebook的词量

各个分词器的具体设置如下所示:

![image-20231226153706226](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231226153706226.png)

#### 2.跨模态训练单一兼容网络

将所有模态label为统一的表示空间使我们能够训练单个 Transformer 编码器-解码器,以通过label预测在不同模态之间进行映射。

a.多模态encoder:对于特定模态的token添加可学习的序列(1D)/密集模态(2D)正弦余弦位置emb, encoder使用可学习的逐块线性投影接受RGB像素输入兼容VIT主干.

b.多模态decoder: 所有token参与与其他token的cross-attention保证信息获取,是用attention mask分离不同模态的token,确保decoder对于特定模态产生一致输出.对于密集模态decoder输入为masked token.模态.位置,输出为预测masked的内容.对于稀疏模态输入为模态.位置,内容,输出是预测seq的下一个token. self-attention使用自回归模型的因果编码避免受未来token影响.由于都转为了离散数据,所以可以用交叉熵损失处理一切任务.decoder可以通过MaksVIT迭代生成多模态预测.

#### 3.多模态屏蔽预训练目标

Transformers 在不同任务集的数据和模型大小方面表现出了出色的可扩展性，特别是与masked重建等可扩展预训练目标配合使用时.多模态预训练使用类似MulitMAE的预训练策略:从所有模态的可视token/部分使并encode子集,并以此训练模型.

输入和输出mask:对于少量输入token大量输入会导致decoder计算成本很大,因此使用目标mask,即只对随机采样子集进行decoder.具体为:用参数为α的对称Dirichlet分布随机抽样得到输入token和输出token数量后使用对密集均匀抽样/稀疏跨度mask.

跨度mask:给定pmask,我们随机mask seq中的token，并用哨兵token替换每个连续的masked token跨度([S_1], [S_2],[S_3]) 然后，目标seq由由哨兵token界定的masked跨度组成，后跟最后的哨兵token表示结束.

迁移实验:对于不同实例化的下游,会使用他们的head到encoder并舍弃decoder.数据增强仍旧很重要.



#### 4.生成能力和探索学习表示

1.由于

a.前置多模态到Tokenization因此可以无条件为任何模态生成任何(通过迭代token预测)训练模式

b.使用masked训练,因此可以有条件(基于不同prompt)的进行内绘制和外绘制.

训练了4M-XL做图像生成,进行了超分辨率+特定下游fine-tune,详情如下:

两阶段生成:大模型生成低分辨率图片后小超分辨率模型做重建.为了规避Transformer的计算成本,基于4M- L fine-tune 100B token的超分辨率模型4M-SR,并上采样到448*448(28\*28 token).输入是低分辨率和高分辨率token的随机(64-1024均匀采样)子集,目标仅包含高分辨率的随机(固定为1024输出)token子集,同时基于完整的caption很少,使用混合mask策略(pmask=0的captions的alpha=5,其它模态alpha=0.05)占比1/4.

训练每个batch分辨率在244-448之间增量32采样,除了RGB走了5epoch的CC12M,其余分词器1ecoch, batchsize降到16,语义分割为10

文生图还做了特定领域/下游任务的fine-tune得到了captions倾斜模型:基于50B token fine-tune 4M,输入采样64-256,输出256.使用混合mask策略(pmask=0的captions的alpha=5,其它模态alpha=0.05),占比1/3.

##### 生成式的细节

生成时间表:确定seq的目标模态.解码方案.解码的token数量.温度.top-k和top-p.

解码方案:

MaskGIT: encode可见token, 对masked token解码并从预测分布中采样top-n,并重复

随机顺序自回归(ROAR):随机选择token进行解码

左到右自回归:用于seq数据而非类图像数据



2.控制并修改输入可以可视化模型基于学习预测的表现形式

多模态编辑:基于条件生成和修复能力,可以基于语义/几何或者根据模型的自抽取进行多模态编辑(仅使用一个模型)

多模态加权指导:通过计算无条件和条件logits加权和实现指定权重干预结果:

链式生成:每个完全生成的模态可以加到一个模态生成的条件中从而导致一致且质量更高的生成(例如说CLIP)

链式生成时会将完全生成的模态加入指导干预集中.

使用负面提示避免不需要的生成.

$Logits_{guided}=Logits_{uncond}+\sum^{n}_{i=1}w_i(Logits_{cond,i}-Logits_{uncond})$



#### 5.消融的模型部分

224*224,12个encoder和decoder,使用所有模态作为input和output,不使用图片增强.CC12M数据集100B token约400M masked样本,随机采样数量为12,alpha=0.2,使用10个模型的std和交叉熵损失,使用检查点和ZeRO-2和FSDP减少显存和资源的消耗,确保不超过20%的预训练数据训练,例如说降低分辨率和减少epoch.

![image-20231226163642378](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231226163642378.png)

![image-20231226165330410](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231226165330410.png)

###### 模型差异

1. 4M没有bias并在前馈中使用 SwiGLU 

2. 这在 4M 没有[CLS]

3. DeiT III 使用可学习的位置嵌入,4M在内的其它方法使用固定的正弦余弦位置嵌入

4. 在COCO上进行微调时，MAE同时使用绝对和相对位置嵌入,而所有其他方法仅使用绝对位置嵌

入

不同数据集不同任务fine-tune参数也有变化,详见原文



#### 6.优化方向

1. 更多的模态数据效果更好: LLM的特征/边缘/草图/人体姿势/视频/多视图图像.

2. Tokenizer的质量

3. 图像更高的分辨率和质量

4. 数据集的大小和质量/强化学习

   

#### 7.生成式功能

RGB->X:使用RGB生成法线box等x结果

链式生成:captions->中间结果->RGB,用于文生图

一对多生成

Any-to-Any

有条件和无条件的修复

图像变化

基于多模态指导的图像生成

探索学习到的表示