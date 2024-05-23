#### Reducing Activation Recomputation in Large Transformer Models, Vijay Korthikanti, SOSP 2023

激活的重新计算本来用于减少内存消耗,但是大多数的重新计算是冗余的,本文提出序列并行和选择性激活重新计算,结合张量并行可以减少5倍的激活内存和90%以上的激活重新计算时间节省.

模型分配到多个GPU成倍的减少显存和参数,但是张量并行引入了通信延迟和性能更差的小矩阵,管道并行需要存储多个microbatches 的激活来减少管道气泡,进而无法减少激活所需的内存且保持高设备利用率.相应的处理(梯度检查点/激活重新计算:不存储大部分激活,并根据需要重新计算并向后传递期间计算梯度)很影响性能,特别是transformer中的完全激活重新计算(层边界中存储并重新计算所有).

###### 重新激活的内存消耗

![image-20240206150058086](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240206150058086.png)

每层激活内存需求$sbh(34+5as/h)$,拆解如下所示

s是序列长度,b是微batchsize,h是hidden层维度,a是attention头数量

Q .K . V矩阵乘是$2sbh$

$QK^{\top}$是$4sbh$

softmax是$2as^2b$

softmax的drop是$as^2b$

V的attention是$2as^2b+asbh$,包含dropout和值

MLP有2个线性层($2sbh.8sbf$).GeLU($8sbh$)和dropout($sbh$)共$19sbh$

线性归一化需要$2sbh$*2的输入输出共$4sbh$



##### 张量并行

对attention和和MLP进行并行并引入了共轭的额外通信$f.\bar{f}$,块内激活也是并行化的

![image-20240206150117031](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240206150117031.png)

内存可以减少为$sbh(10+24/t+5as/ht)$​​,t为t路张量并行.



##### 序列并行

![image-20240206171545815](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240206171545815.png)

由于非张量并行区域序列维度独立因此可以走序列并行,为了避免张量并行外的额通信,引入转换器操作$g.\bar{g}$,并在序列并行后前向all-gather跑张量并行,并张量并行结束后前向reduce-scatter进入下一个序列并行,具体如下所示,由于张量并行和序列并行通信开销一样,所以白赚.

$[Y_1^s,Y_2^s]=LayerNorm([X_1^s,X_2^s])$

$Y=g(Y_1^s,Y_2^s)$因为GeLU需要完整的Y

$[Z_1^s,Z_2^s]=[GeLU(YA_1^C),GeLU(YA_2^C)]$

$W_1=Z_1^hB_1^r,W_2=Z_2^hB_2^r$

$[W_1^s,W_2^s]=\bar{g}(W_1,W_2)$

$[V_1^s,V_2^s]=[Dropout(W_1^s),Dropout(W_2^s)]$,Dropout可以走序列并行

除此之外不存完整的Y而是在单独存储$Y_i^s$,只在反向传播计算Y梯度时做all-gather

消耗内存可以减少为$\frac{sbh}{t}(34+5as/h)$



##### 管道并行

管道并行为了减少管道泡沫因此内存的变化需要额外分析,考虑1F1B:具有最小化管道气泡的调度将最大的内存压力放在管道的第一阶段.因此第一阶段必须存储transformer L层的激活值,需要$\frac{sbhL}{t}(34+5as/h)$的内存,其它pipeline,的内存取决于不同方案,Megatron-LM要存$L(1+\frac{p-1}{pm})$的激活,内存缩放系数是$(1+\frac{p-1}{pm})$,m是交错阶段的数量



##### 额外激活内存

embeddings层dropout需要$sbhp/t$

输出层前Norm需要$2sbh/t$​

投影到词表维度的输出层也需要$2sbh/t$

float32交叉熵损失函数需要$4sbv/t$​

pipeline一阶段激活大概是$4sbh/t(1+v/h)$

总的为$\frac{sbhL}{t}(\frac{P}{L}+\sigma_{p=1}\frac{4}{L}(1+v/h))$,$\sigma_{p=1}$是pipeline第一阶段的内存,由于这些额外内存很小所以可以忽略不计



##### 选择性激活重算

![image-20240207093529532](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240207093529532.png)

为了进一步减少内存,以额外前向传递的对所有 Transformer 层设置检查点的重新计算可显着减少训练模型所需的内存但会单来30% 到 40% 的计算时间开销,因此重算Q. K和V值的线性层增加维度后的attention操作,这部分对应$\frac{sbh}{t}(34+5as/h)$的$5as/h$部分,特别是当这部分大于34时减少超过50%,因此内存消耗降为$\frac{34sbh}{t}$.对于管道并行由于一阶段必须存储,因此存储microbatches的激活并选择性重算其它是更优的.



