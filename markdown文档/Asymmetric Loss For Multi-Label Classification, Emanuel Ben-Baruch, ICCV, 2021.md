#### Asymmetric Loss For Multi-Label Classification, Emanuel Ben-Baruch, ICCV, 2021

多标签分类的正负样本不均匀会导致训练低估正标签梯度,Focal Loss平等对待正负样本会导致负样本梯度损失累计和罕见正样本权重降低.提出ASL Loss动态降低权重容易的负样本和丢弃可能标注错误的样本,同样可用于单标签分类和对象检测.

1. 正负样本分离,基于不同的指数衰减

2. 通过硬阈值丢弃容易学习的负样本

3. 丢弃过难负样本,因为可能是错误标记的

##### 二元交叉熵和Focal Loss

K个标签每个标签出logit, loss为$L_{tot}=\sum_{k=1}^{K}L(\sigma(z_k),y_k)$​

每个label的loss为$L=-yL_{+}-(1-y)L_{-}$

Focal Loss为$L_{+}=(1-p)^{\gamma}log(p)$

$L_{-}=p^{\gamma}log(1-p),\gamma>0$

##### 非对称损失

可以看到当$\gamma$设置较高时,稀有正样本的梯度也会降低,为此使用非对称损失

$L_{+}=(1-p)^{\gamma_+}log(p)$

$L_{-}=p^{\gamma_-}log(1-p)$设置$\gamma_->\gamma_+$使得关注正样本贡献

##### 非对称概率转移

对于严重的样本不均衡,非对称损失的衰减不足够,因此引入了转移概率$p_m$把概率低于m的样本删除.

$p_m=max(p-m,0)$

$L_{-}=p_m^{\gamma_-}log(1-p)$

其中$\gamma_-=2,m=0.2,\gamma_+=0$

##### ASL Loss和自适应不对称损失函数的超参数

$L_{+}=(1-p)^{\gamma_+}log(p)$​

$L_{-}=p_m^{\gamma_-}log(1-p)$

概率差距$\Delta_p=p^+_t-p^-_t$ ,后者为正负样本的平均概率.

$\gamma_-=\gamma_-+\lambda(\Delta_p-\Delta_{p_{target}})$

其中$\lambda$是步长,$\Delta_{p_{target}}$设为0.1