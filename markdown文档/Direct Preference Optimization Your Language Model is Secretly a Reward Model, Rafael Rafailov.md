#### Direct Preference Optimization: Your Language Model is Secretly a Reward Model, Rafael Rafailov

提出基于prompt-好答案-差答案并基于简单二元交叉熵分类目标的直接偏好优化DPO

DPO更新增加了偏好响应相对于不偏好响应的相对对数概率,并基于每个示例的动态重要性权重以防止模型退化

##### SFT: 微调得到下游任务模型$\pi^{SFT}$

##### 奖励建模阶段:基于问题和$\pi^{SFT}$得到回答对$(y_1,y_2)\sim\pi^{SFT}(y|x)$,让人类标注可以得到答案的偏好$y_w \succ y_l$,对人类偏好的标注数据集$D=\{x^{(i)},y_w^{(i)},y_l^{(i)}\}_{i=1}^N$进行建模有概率和LL

$p^*=(y_1 \succ y_2|x)=\frac{exp(r^*(x,y_1))}{exp(r^*(x,y_1))+exp(r^*(x,y_2))}$​

$L_R(r_\phi,D)=-E_{(x,y_w,y_l)\sim D}[log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]$

$\sigma$是逻辑函数,$\pi^{SFT}(y|x)$一般是transformer后接维度为1的linear,并进行标准化奖励使得

$\forall x,E_{x,y\sim D}[r_\phi(x,y)]=0$

#####  RL Fine-Tuning 

$max_{\pi_\theta}\space  {E}_{x\sim D,y \sim \pi_\theta(y|x)}[r_\phi(x,y)]-\beta D_{KL}[\pi_{\theta_{old}}(y|x),\pi_{ref}(y|x)]]$

最经典的看PPO论文

##### 直接偏好优化DPO

可以推导$\pi_r(y|x)=1/Z(x)\pi_{ref}(y|x)exp(r(x,y)/\beta)$

$Z(x)=\sum_{y}\pi_{ref}(y|x)exp(r(x,y)/\beta)$

处理后有

$r_t(\theta)=\beta log \frac{\pi_r(y|x)}{\pi_{ref}(y|x)}+\beta logZ(x)$

实际上关注的是策略间的概率差

$p^*(y_1 \succ y_2|x)=\frac{1}{1+exp(\beta log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}-\beta log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)})}$

于是有DPO的优化目标

$L_{DPO}(\pi_\theta;\pi_{ref})=-E_{(x,y_w,y_l)\sim D}[log\sigma(\beta log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_1|x)})]$

梯度上有

$\nabla_\theta L_{DPO}(\pi_\theta;\pi_{ref})=-\beta E_{(x,y_w,y_l)\sim D}[\sigma(\hat{r}_\theta(x,y_l)-\hat{r}_\theta(x,y_w))\\ [\nabla_\theta log \pi(y_w|x)\\-\nabla_\theta log \pi(y_l|x)]$

即估计错误给高权,增大$y_w$减少$y_l$的似然值.

##### DPO流程

用promptx得到回答$y_1,y_2\sim\pi_{ref}(\cdot|x)$并得到人类标注$D=\{x^{(i)},y_w^{(i)},y_l^{(i)}\}_{i=1}^N$,并根据$L_{DPO}$优化.
