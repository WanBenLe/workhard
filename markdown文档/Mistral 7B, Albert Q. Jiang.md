##### Mistral 7B, Albert Q. Jiang

提出全面优于Llama2 13B且在数学代码推理优于Llama1 34B的的Mistral 7B和微调可以prompt超越Llama2 13B-chat的Mistral 7B-Instruct

Mistral 7B提出了分组查询注意力GQA加快推理且用滑动窗口注意力SWA处理任意长度序列且降低推理成本.

模型参数如下所示:

![image-20240323155712426](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240323155712426.png)

###### 滑动窗口注意力SWA

超过窗口size $W$的信息使用堆叠的transformer层关注,position $i$层$k$的隐状态$h_i$蕴含了前一层position在$i-W$到$i$的的隐状态,$h_i$​最多可以访问到W*k的tokens,W=4096大概是13k的tokens下图是W=3的例子

![image-20240323155731181](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240323155731181.png)

###### 滚动缓存区cache

固定的注意力跨度可以进行滚动缓存限制缓存大小并优化分配,滚动缓存区的size为$W$,时间戳$i$的key和values存在$i \space mod \space W$的位置里.如果$i$比$W$​长则以FIFO先进先出覆盖.

###### 预填充和分块

如果预先知道prompt可以预填充,如果prompt太大就分块,使用$W$​作为块大小.每个块都需要计算对cache和块的注意力.如下图$W=3$,分成了三块,current的对自己做因果mask,对cache做滑窗,滑窗外的不进行计算限制了内存使用.

![image-20240323161708200](C:\Users\SFC\AppData\Roaming\Typora\typora-user-images\image-20240323161708200.png)

