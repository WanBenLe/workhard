#### Trans-Encoder: Unsupervised sentence-pair modeling through self- and mutual-distillations, Fangyu Liu, ICLR 2022

Encoder模型有Cross-Encoder和bi-Encoder,后者双塔出向量算score,前者使用

\[CLS\]\句子一[SEP\]句子二[SEP]合并两个文本,由于可以学习到attention信息所以效果更好,但是前者性能不佳.

Trans-Encoder(TENC)是完全无监督的句子对模型,流程如下:

![image-20231015165608649](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231015165608649.png)

1.给定PLM,用SimCSE(给dropout,用InfoNCE损失函数)类似的方法训练bi-Encoder,双编码器需要共享权重避免嵌入出不一致的结果

$L_{infoNCE}=-\sum_{i=1,|X|}log\frac{exp(cos(f(x_i),f(\bar{x_i})/\gamma)}{\sum_{x_j \in N_i}exp(cos(f(x_i),f(\bar{x_i}))/\gamma}$

2.使用bi-Encoder进行嵌入并进一步得到句间相似度,这个结果可以用于训练Cross-Encoder,这里需要使用soft-BCE损失函数

$L_{BCE}=-1/N\sum_{n=1,N}(y_n*\log(\sigma(x_n))+(1-y_n)*\log(1-\sigma(x_n)))$

3.这里使用Cross-Encoder的结果得到了新的相似度,然后可以用相似度构造新的标签让双塔bi-Encoder学习,这里使用MSE损失函数

思路上可以看到是希望使用bi-Encoder学习到Cross-Encoder的结果,但是保留bi-Encoder的速度,由于PLM自身可能存在错误,所以提出了训练n个(n为PLM数量)不一样的TENC,用平均结果生成标签,后者称为Trans-Encoder-mutual(TENC-mutual).

最后取出最佳的bi-Encoder和Cross-Encoder视乎情况使用.

