#### BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models, Junnan Li, ICML 2023

直接冻结pre-train的图像encoder和LLM,然后训练使用的轻量级查询转换器Q-fromer拟合模态差距.

冻结图像和LLM大模型只在loss阶段调整在图像-文本生成损失不足以弥补模态差距.

##### Q-fromer

参数量减少了54倍,且与输入图像分辨率无关.

![TCA$EAZY5SZSMVJNEWR79QB](C:\Users\70307\Desktop\TCA$EAZY5SZSMVJNEWR79QB.png)

由2个共享self-attn的转换器自模块组成:

1.与冻结图像encoder交互提取视觉特征的图像转换器

2.文本encoder-decoder的文本转换器

可学习的查询嵌入作为图像转换器的输入,跟self-attn交互的同时每隔一个转换器块和冻结的图像特征交互,还可以通过self-attn层和文本交互.

##### 第一阶段:冻结图像encoder,训练Q-former学习与文本相关的视觉特征视觉语言表示学习

图文对比学习ITC对齐表征使得互信息最大化(查询+图像输出和文本表征对齐),防止信息泄露用了单向self-attn mask,由于参数量导致模型小可用批内负样本训练,查询和文本使用不同attn mask控制交互

图文生成ITG查询提取文本信息,self-attn给文本token提取,并使用多模态因果self-attn mask禁止使用文本token信息.[DEC]替换[CLS]发出decoder任务.

图文匹配ITM预测图文是否匹配,使用双向self-attn mask

##### 第二阶段:Q-fromer输出连接到冻结LLM,执行视觉到文本生成的学习训练

首先FC将Q-fromer结果与LLM维度对齐,然后查询emb插入到文本emb前缓解灾难性遗忘.

decoder LLM用语言建模loss进行预训练.

encoder-decoder LLM训练将文本分成前后两部分,前半和视觉表示作为encoder输入,后半文本作为生成目标

![QQ图片20240613231554](C:\Users\70307\Desktop\QQ图片20240613231554.png)