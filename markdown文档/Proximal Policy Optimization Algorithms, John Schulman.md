#### Proximal Policy Optimization Algorithms, John Schulman

省流:近端策略优化PPO

一般RL梯度如下所示$\pi_\theta$是随机策略$\hat{A}_t$是t时刻的优势函数

$\hat{g}=\hat{E}_t[\nabla_\theta log\pi_\theta(a_t|s_t)\hat{A}_t]$​

策略梯度方法通过以下优化得到

$L^{PG}(\theta)=\hat{E}_t[log\pi_\theta(a_t|s_t)\hat{A}_t]$

但是对于LLM相同轨迹容易导致破坏性的策略更新

##### 可信区域方法TRPO

$max_{\theta} \space \hat{E}_t[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t]$

$s.t.\space  \hat{E}_t[KL[\pi_{\theta_{old}}(\cdot|s_t),\pi_{\theta}(\cdot|s_t)]]\le \sigma$

共轭梯度之后有无约束优化问题

$max_{\theta}\space  \hat{E}_t[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t-\beta KL[\pi_{\theta_{old}}(\cdot|s_t),\pi_{\theta}(\cdot|s_t)]]$

定义$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)},r(\theta_{old})=1$

$L^{CPI}= \hat{E}_t[r_t(\theta)\hat{A}_t]$

##### 由于只加$\beta$的改进不足,为了避免策略变化过大进行额外clip改进$\epsilon=0.2$

$L^{CLIP}(\theta)=\hat{E}_t[min(r_t(\theta)\hat{A}_t,clip(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)]$​

可以确保策略更新在$[1-\epsilon,1+\epsilon]$​范围中

##### 自适应KL惩罚系数$\beta$

由于KL散度表现不佳,提出新的优化:

每次策略更新首先用几个minibatch进行以下更新

$L^{KLPEN}(\theta)=  \hat{E}_t[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t-\beta KL[\pi_{\theta_{old}}(\cdot|s_t),\pi_{\theta}(\cdot|s_t)]]$

$d= \hat{E}_t[KL[\pi_{\theta_{old}}(\cdot|s_t),\pi_{\theta}(\cdot|s_t)]]$​

$d<d_{targ}/1.5,\beta/=2$

$d>d_{targ}/1.5,\beta*=2$​

##### 添加熵奖励得到新优化目标

$L_t^{CLIP+VF+S}(\theta) \hat{E}_t[L_t^{CLIP}(\theta)-c_1L_t^{VF}(\theta)+c_2S[\pi_\theta](s_t)]$​

$c_1,c_2$是系数,$S$是熵奖励,$L_t^{VF}=(V_\theta(s_t)-V_t^{targ})^2$​

##### 推广一个适用于RNN的更新方式,有

$\hat{A}_t=\sigma_t+(\gamma\lambda)\sigma_{t+1}+...+(\gamma\lambda)^{T-t+1}\sigma_{T-1}$​

$\sigma_t=r_t+\gamma V(s_{t+1})-V(s_{t})$​

##### PPO

对于K epoch的每次迭代,N并行actors收集T的数据,使用NT数据进行更新

算法1: PPO, Actor-Critic

for iter=range(iter_all):

​	for actor in range(N):

​		get $\pi_{\theta_{old}} $ for T timesteps 

​		calc$\{\hat{A}_1,...,..\hat{A}_T\}$​

​	K epochs和minibatch size$M\le NT$内,基于$L$优化$\theta$

​	$\theta_{ols}\leftarrow \theta$
