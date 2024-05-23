#### Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, Deepak Narayanan, Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis 2021

省流:比deepspeed的ZeRO-3好.

![image-20231214144818746](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231214144818746.png)

GPU显存和训练时长是大模型训练的问题.本文提出interleaved pipeline调度组合TP. PP. DP,基于跨节点PP,节点内TP/DP,并使用kernel实现/计算图切分/硬件通信优化综合提高了10%的吞吐.

DP: 横向拓展好,但大模型会让单batchsize和总batchsize很小,降低了GPU利用率和增加了通信成本

TP: 层内Tensor并行,将transformer层卸载到多个GPU上和节点上.但是节点间RDMA通信比节点内NVLink慢.且小矩阵GEMMs(通用矩阵乘)会降低GPU利用率

PP: 流水线并行,模型层卸载到多个GPU的同时将batch也切成micro-batch的形式流水线并行.每个设备会进行一组操作需要step间进行设备同步.同步和参数更新导致的设备空闲叫做Pipeline bubble, batchsize越大GPU利用率越高.本文也使用新的pipeline schedule提高效率.

##### 一. Interleaved Schedule

下面给出了GPipe. PipeDream. Interleaved Schedule三种流水线并行的图

![img](https://diveblue.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F1dd15054-4a8a-41f1-a8cf-63949d9c3fd3%2F6c7f2f12-a244-4d37-b86f-031febec66c0%2FUntitled.png?table=block&id=baa65412-8630-4ac9-b0d8-b94255abf3fd&spaceId=1dd15054-4a8a-41f1-a8cf-63949d9c3fd3&width=2000&userId=&cache=v2)

![image-20231214135554429](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231214135554429.png)

![26E7EB40-9373-4b5f-B1B2-EDCB2CF8BE1E](C:\Users\SFC\Desktop\application\26E7EB40-9373-4b5f-B1B2-EDCB2CF8BE1E.png)

在GPipe情形下,有气泡比如下,p是pineline stage数,m是micro-batch数,$t_{pb}$是气泡时间,t是总时间

$\frac{t_{pb}}{t}=\frac{p-1}{m+p-1}$

瓶颈在于m不可能无限增大

PipeDream提前了反向时间,并在完成了反向运算之后就释放内存,减少了内存压力,可以提高m很大时的内存效率

Interleaved Schedule将GPU也分了多个stage进一步细化训练过程,跟PipeDream相比成倍的减少了空泡时间的比例,但同时的由于每个细分的GPU stage需要跨设备通信,因此通信也成倍的增加了.

##### 二. Tensor并行

![image-20231214141725368](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231214141725368.png)

处理了Transformer层,前者包含一个self-attention和2层MLP,后者由2个GEMMs和GeLU激活函数组成

$Z=Dropout(GeLU(XA)B)$

沿着列将A可以拆分成独立的部分,B可以沿着行拆分,最后Y也拆分成了2份,这样子可以避免GEMM通信.self-attention的K. Q. V可以按列拆分,线性层可以用按行分区的方法拆分,这样子整个transformer层就拆到了不同的GPU上并进行规约即可.

##### 三. 并行组合分析

此处主要优化是激活重计算时间换空间,反向时重新计算激活值避免保存前向运算的激活值.

p: PP大小

t: TP大小

d: DP大小

$n=p*t*d$,GPU数

B:总batchsize

b: micro-batchsize

m=$\frac{B}{b*d}$,micro-batch数

pineline并行中设d=1有

$\frac{p-1}{m}=\frac{n-t}{mt}$

Tensor并行的transformer的micro-batch有$8bsh\frac{t-1}{t}$通信量.s序列长度h隐含层大小

TP和PP组合通信量扩大l倍,l是pineline stage层数.通信量增加使得Tensor并行适合节点内.

DP和PP组合设t=1有,即增大总batchsize可以减少气泡

$\frac{p-1}{m}=\frac{n(b-d)}{B}$

DP和TP取决于跨节点TP通信量大和小矩阵太多GPU利用率会下降.b的影响如上述公式可分析得到.

##### 四.技术优化

PP同stage的不同TP设备transformer层输出一致会重复传输,通过传输前scatter(一对多通信传输)传输后all_gather(搜集数据到所有节点)用额外通信避免重复传输.

![image-20231214143558145](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231214143558145.png)

[batch, sequence, attention-head, hidden-size]改为[sequence, batch, attention-head, hidden-size]避免转置并允许使用GEMM kernel.

PyTorch JIT为逐元素操作bias+GeLU/bias+dropoutadd)生成融合算子

实现了支持常规mask和隐式因果编码causal mask的自定义融合算子(scale+mask+softmax)

##### 五.性能

浮点计算次数FLOPs公式,V是词表大小

$F=96Bslh^2(1+\frac{s}{6h}+\frac{V}{16lh})$

和deepspeed的ZeRO-3对比图

![image-20231214144754749](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20231214144754749.png)



