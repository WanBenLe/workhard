#### RoBERTa: A Robustly Optimized BERT Pretraining Approach, Yinhan Liu



##### BERT

Encoder Only架构,跟ChatGLM一样

输入是两段([CLS],...,[SEP],...[EOS])三部分,token, sents,绝对位置编码,最多512 token

emb靠学习,相加后标准化就是emb.

MLM:选15% token,80%[mask],10%随机换,10%不变并预测

NSP:预测下一句是否为真实的下一句.

Adam的$\beta_1=0.9,\beta_2=0.999,\epsilon=1e-6$学习率衰减$L_2=0.01$

预热1 e-4学习率线性衰减10000 steps,给dropout=0.1,GELU激活,更新100 w次, batchsize=256,16 G文本



##### RoBERTa

1. 更大的batchsize(8 K),更多的数据160 G文本,设置$\beta_2=0.98$

2. 删除下一个句子是否为下一句的预测目标,使用Full-Sentence:每个输入包含从一个或多个文档中连续采样不多于512 token的完整句子,跨越文档进行采样时添加额外分隔符.
3. 50 k 字节对编码BPE词表
4. 动态修改mask模式:训练数据复制10次,每次的mask是不同的(共40次里会有4次重复)

<img src="C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240613183105537.png" alt="image-20240613183105537" style="zoom:50%;" />
