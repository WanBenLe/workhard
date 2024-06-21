#### Improving Language Understanding by Generative Pre-Training, Alec Radford

省流:GPT

探索了允许数据跨域的二阶段训练方法

##### 一阶段训练使用无监督的语言建模目标

给定句token$U={u_1,...,u_n}$进行极大似然估计$L_1(U)=\sum_{i}logP(u_i|u_{i-k},...,u_{i-1};\Theta)$,k是上下文窗口,$\Theta$是估计联合概率$P$的神经网络的参数并使用SGD训练.架构使用decoder only:

$h_0=UW_e+W_p$

$h_l=trans(h_{l-1})\space\forall i \in[1,n]$

$P(u)=softmax(h_nW_e^\top)$

##### 二阶段训练使用有监督目标将参数调整到目标task上

$P(y|x^1,...,x^m)=softmax(h_l^mW_y)$

$L_2(C)=\sum_{(x,y)}logP(y|x^1,...,x^m)$

$L_3(C)=L_2(C)+\lambda\cdot L_1(C)$

位置emb是学习出来的,使用LayerNorm



#### Language Models are Unsupervised Multitask Learners, Alec Radford

省流:GPT2-证明了语言模型可以zero-shot中执行下游任务-无需任何参数或架构修改

语言建模给定序列为$(s_1,s_2,...,s_n)$的示例集$(x_1,x_2,...,x_n)$进行无监督分布估计,联合概率由于顺序性等价于条件概率的乘积.

$p(x)=\prod_{i=1}^np(s_n|s_1,...,s_{n-1})$

1. 数据来源从高质量网页开始,并以此判断其它网页的质量,并剔除wiki文本数据

2. 使用字节对编码BPE作为tokenizer但是阻止前者跨字符类别合并任何字节序列(空格除外)
3. 跟GPT相比,LayerNorm移到了输入,最后的self-attn也加了额外的LayerNorm
4. 缩放残差层$1/\sqrt{N}$​,N为残差层数量,词表扩到50257,句长扩到1024,batchsize扩到512
5. 训练只有一阶段
6. 为了帮助识别下游任务,给出prompt