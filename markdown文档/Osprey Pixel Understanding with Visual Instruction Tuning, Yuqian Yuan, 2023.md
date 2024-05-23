#### Osprey: Pixel Understanding with Visual Instruction Tuning, Yuqian Yuan, 2023

近来的图像模型基本都基于图像/boxing和masked,这种情况下无法实现像素级理解也无法提供语义标签和语义属性和标题,为此提出了Osprey,一种文本mask指令调整,将细粒度mask区域合并到语言指令中扩展MLLM以实现像素级视觉理解.

Osprey使用CNN-CLIP backbone作为视觉encoder,与ViT相比可以泛化到更大的输入分辨率并有高效性和鲁棒性.使用mask感知视觉提取器从高分辨率输入提取视觉mask特征,并可以与SAM无缝集成获得多粒度语义.还构建了数据集Ospery-724k和附加指令样本.

SEEM . HIPIE. Semantic SAM可以提供语义标签但是不够,需要纳入额外的语义, 例如颜色.位置,甚至用于场景理解和推理的一般描述

##### Ospery-724k

由对象级和零件级掩码文本指令数据组成,并引入了带有简短响应格式化提示的负样本挖掘方法.

![image-20231227110125495](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231227110125495.png)

对象级指令:对于具有N个对象区域的图像,我们基于带有masked注释的公开数据集，充分利用其图像级和对象级标题,但是标题简单且短没啥语义.设计了一个pipeline生成对象类别/类型/动作/位置/颜色/状态.

1. LLaVA生成图像级描述

2. 边界框+区域标题+边框对象概念encode和场景空间位置encode+不同角度描述特定区域传入GPT-4生成区域及描述和对话.

零件级指令:使用含 55 种不同的属性,包括 29 种颜色.10 种图案和标记.13 种材料和 3 个反射率级别的数据集PACO-LVIS并使用GPT-4 QA得到零件级区域的指令跟踪数据

稳健性:询问给定区域是否属于特定类别,并预测是/否. 负样本挖掘的空间感知使模型能够识别空间上最接近给定对象的特定对象类别,而对于类别感知则根据与目标类名称的高度 SentenceBert语义相似度来选择负类别.并从前 8 个语义相似的候选者中随机选择一个类别,以增强负面类别的多样性.

灵活性:添加了简短的响应指令,涵盖特定对象区域的类别.颜色.类型.位置或数量并给了简短的回答提示.

![image-20231227112557884](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231227112557884.png)

##### Ospery架构

1. 图像级视觉encoder : ConvNeXt-Large CLIP作为视觉encoder,使用res4输出作为图像级特征

2. 像素级masked感知视觉提取器:对masked视觉特征编码,并收集区域的空间位置信息.

对1的结果做mask-pooling, 过线性投影层和MLP层获得视觉masked token.为了保留空间,对目标区域二进制mask reshape为224*224后展平,并于前述结果合并得到每个masked区域的emb

$t_i=\sigma(\sum^4_{j=1}P_j(MP(R_i,Z(x)_j)))$

3. LLM :文本使用Vicuna的tokenizer并得到emb,masked的区域使用\<region>进行占位(附加在引用区域名之后),mask token使用\<mask> 空间token使用\<position>,并使用\<image>\n作为图像token

这样子可以使得图像.masked.空间.文本都被输入到LLM进行decode.

输入是图像.masked和文本, tokenization后获得emb. 然后将cross-attention后的mask和emb给LLM获得细粒度语义理解.

Prompt:

\<image>

This provides an overview of the image.

Question 1: Can you give me a short description of region 1? 

Question 2: What is in region 2 ?

##### 训练:最小化下个token预测的loss

1. 图文对齐阶段:ConvNeXt-Large基于CLIP架构基于图像级特征和接在LLM后的MLP语言连接器进行图文特征对齐,训练的是图像级projector,视觉encoder的ConvNeXt-Large和LLM的LLaVA-1.5冻结.batchsize=128,lr=$10^{-3}$

2. masked文本对齐阶段:一阶段基础上新增masked感知视觉提取器,并且只训练该部分.batchsize=4. lr =2*$10^{-5}$,epoch=2

3. 端到端fine-tune:冻结视觉encoder,对图像级projector,masked感知视觉提取器和LLM fine-tune. batchsize=4. lr=$10^{-5}$,LLM最大长度为2048.

在4个80 G显存A100用deepspeed训练7/15/48 h, 输入大小维512*512