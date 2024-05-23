# Cache-Augmented Inbatch Importance Resampling for Training Recommender Retriever
缓存增强型批内重要性重采样($\chi$IR)根据一定的概率对给定的小批量训练对中的item进行重新采样，缓存频繁采样的item来扩充候选项目集，目的是重用历史信息样本,能够根据inbatch item对query依赖负例进行采样，并捕获模型训练的动态变化，从而更好地逼近 Softmax，并进一步有助于更好的收敛

从数据集整体用均匀分布抽样减少分布偏差成本太高



推荐系统定义为query和item的集合$\{q_i\in \R^{d_u}\}_{i=1}^M,\{e_j\in \R^{d_i}\}_{j=1}^N$,mapping之后使用内积函数计算相似度$s(u,i)=<\phi_Q(q_u),\phi_I(e_i)>$并输出Top k,极大似然之后的损失函数为

$\begin{flalign}&L_{softmax}(D,\Theta)\\=&-\frac{1}{|D|}\sum_{(u,i)\in D}logP(i|u)\\=&-\frac{1}{|D|}\sum_{(u,i)\in D}log \frac{exp(s(u,i))}{\sum_{j \in I}exp(s(u,j))} \end{flalign}$

数量大的时候计算成本很大,采样后损失函数为

$L_{sampled\_softmax}(D,\Theta)=-\frac{1}{|D|}\sum_{(u,i)\in D}log \frac{exp(s'(u,i))}{\sum_{j \in S}exp(s'(u,j))}$

可以看到softmax是最佳的无偏抽样分布,然而计算成本还是太高,而且就算满足了前者,分布仍然依赖于给定的查询.

inbatch的抽样由于长尾性,基于流行度分布,query独立的抽样如频率抽样会更好,但是没有优化性能.给出一个基于重要性抽样的改进.

对mini-batch B,对query的item进行加权重采样,$pop$是item的频率,query重采样的item集合记为$R_{u}$

$w(i|u)=\frac{exp(s(u,i)-\log pop(i))}{\sum_{j \in I}exp(s(u,j)-\log pop(j))}$

可以证明该抽样在假设下的服从softmax分布和期望梯度的渐进无偏逼近,于是有

$L_{BIR}(B,\Theta)=-\sum_{(u,i)\in B}log \frac{exp(s(u,i))}{\sum_{j \in R_u}exp(s(u,j))}$

## 算法1 batch内重要性抽样BIR

输入:$D=\{(u,i)\}$,epochs T

输出:$\Theta$

从D中统计得到item流行度$P=\{pop(i)|\forall i \in I\}$

for e=1:T

​	for B $\in$ D

​		$U=\{u|(u,i)\in B\},I=\{i|(u,i)\in B\}$

​		将query和item mapping到相应的表示空间

​		$S=E_UE_I^\top$

​		for $ u \in U$ :

​			$w(i|u)=\frac{exp(s(u,i)-\log pop(i))}{\sum_{j \in I}exp(s(u,j)-\log pop(j))},\forall i \in I$

​			使用$w(i|u)$得到$R_u$

​		end

​		优化目标函数$L_{BIR}(B,\Theta)$

​	end

end

batch size越大,bias越小,流行度相差越多,bias越大



增加样本量可以控制bias和var,但是如果只用均匀分布/流行度分布的采样,容易导致过采样,提出了$\chi IR$采样

$L_{\chi IR}(B,\Theta)=-\lambda\sum_{(u,i)\in B}log \frac{exp(s(u,i))}{\sum_{j \in K_u}exp(s(u,j))}-(1-\lambda)\sum_{(u,i)\in B}log \frac{exp(s(u,i))}{\sum_{j \in R_u}exp(s(u,j))}$

## 算法2:缓存增强型批内重要性重采样($\chi$IR)

输入:$D=\{(u,i)\}$,epochs T,缓存数量C,超参数$\lambda$

输出:$\Theta$

初始化出现向量$o=\{0\}^N_{i=1}$,均匀分布抽取的缓存集$\C$

从D中统计得到item流行度$P=\{pop(i)|\forall i \in I\}$

for e=1:T

​	for B $\in$ D

​		$U=\{u|(u,i)\in B\},I=\{i|(u,i)\in B\}$

​		将query和item还有$\C$ mapping到相应的表示空间

​		$S=E_UE_I^\top,S'=E_UE_c^\top$

​		for $ u \in U$ :

​			$w(i|u)=\frac{exp(s(u,i)-\log pop(i))}{\sum_{j \in I}exp(s(u,j)-\log pop(j))},\forall i \in I$

​			使用$w(i|u)$重抽样得到$R_u$

​			$c(i|u)=\frac{exp(s'(u,i)-\log q(i))}{\sum_{k \in \C}exp(s'(u,k)-\log q(k))},\forall i \in \C$

​			使用$c(i|u)$重抽样得到$K_u$

​		end

​		$O=\{i|i\in K_u \bigcup R_u,\forall u \in U\},o_i=o_i+\sum_{j\in O}I[j=i]$

​		基于$w=o_i$从$\C$抽C样本作为新缓存

​		优化目标函数$L_{\chi IR}(B,\Theta)$

​	end

end

该算法需要计算$O$还有嵌入的$E_c$,$\C$和$O$都需要保存,空间复杂度和时间复杂度主要在这

$\lambda$取0.5时mini-batch和cache的梯度贡献可能比较均衡,支持更复杂的tower,$|K_u|=|R_u|=|B|/2$





1.通过item流行度减少mini-batch内的选择偏差有利于减少 softmax 分布的近似偏差

2.在mini-batch之外引入负例可以扩大候选集,进而有利于模型学习

3.缓存频繁采样的item有助于提高模型性能，表明频繁采样的item比均匀采样的item提供更多信息