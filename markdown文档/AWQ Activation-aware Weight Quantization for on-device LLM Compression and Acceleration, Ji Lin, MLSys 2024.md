#### AWQ: Activation-aware Weight Quantization for on-device LLM Compression and Acceleration, Ji Lin, MLSys 2024

提出AWQ激活感知权重量化,通过观察激活来搜索每通道缩放来保护更有效的显著权重并对其它权重来进行LLM的INT3/INT4低位权重量化,AWQ不需要反向传播和重建(说的就是GPTQ)因此也不会在校准集过度拟合,

测试发现根据L2范数/激活幅度确定重要性并保护1%的显著权重不进行量化可以提高性能且控制困惑度.但是如上操作可以提高性能不增加模型大小但是FP16的混合精度对系统实现来说很困难,因此提出每通道缩放来减少显着权重的量化误差.

量化函数可以表示为

$Q(w)=\Delta \cdot Round(w/\Delta),\Delta=\frac{max(|w|)}{2^{n-1}}$

$w$是权重,$\Delta$是量化缩放,$n$是量化位数,处理后有:

$Q(w\cdot s )\cdot x/s=\Delta'\cdot Round(ws/\Delta')\cdot x\cdot 1/s$

$\Delta'$是应用$s>1$后的新量化缩放,$RoundErr(\cdot)$由于四舍五入会服从$[0,0.5]$的均匀分布,误差均值为0.25,对w的改变一般不会改变组w的最大值因此$\Delta'\simeq \Delta$,FP16下的x和$\Delta$没有量化误差,因此有

$Err(Q(w)x)=\Delta \cdot RoundErr(w/\Delta)\cdot x$

$Err(Q(w\cdot s ) (x/s))=\Delta'\cdot RoundErr(ws/\Delta')\cdot x\cdot 1/s$

误差比率为$\Delta'/\Delta \cdot 1/s$,在上面的分析下可以得到w的相对误差更小,通过实验可以看出,s增加显著通道的相对误差会变小,但是$\Delta$的增大会导致非显著通道的误差被放大,因此需要进行联合优化:

$s^*=arg\space min_s L(s)$

$L(s)=||Q(W\cdot diag(s))(diag(s)^{-1}\cdot X)-WX||$

Q是量化函数,W是FP16的原始权重,X是较小校准集(避免校准集过大可以防止过度拟合)的输入特征.s是每个输入通道的缩放因子,$s^{-1}\cdot X$可融合到前一个算子中.为了避免量化函数不可导和近似梯度收敛稳定性较差的问题,使用激活尺度确定最佳缩放的搜索空间:

$s=s{X^{\alpha}},a^{*}=arg\space min_\alpha L(sx^{\alpha})$

$sX$是每个通道的平均激活尺度,$\alpha$是[0,1]之间的平衡显著和不显著权重的超参,0为不缩放,1为最激进的缩放.然后通过最小化裁剪后权重的MSE来选择最优$\alpha$



还有一个叫做tinychat的玩意,省流了.