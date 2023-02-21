给定$y=X\beta+\epsilon $在满足高斯马尔科夫假设时为BLUE.

但如果rank(X)<p即非满秩条件,估计$\beta=inv(X'X)X'Y$会存在问题,此时可能会用广义逆,但也可以通过在$||y-X\beta||^{2}$中加入L1和L2正则项成为Lasso回归和岭回归.

Lasso有坐标下降法和最小角回归法,也可以通过次梯度做近端梯度下降法从略.

但Lasso不再是unbias的估计,因此在Least squares after model selection in high-dimensional sparse models, Belloni, Bernoulli, 2013中,Belloni提出可以在Lasso之后用剩下的x变量和y变量进行OLS得到更好的估计,这个方法被称为post-Lasso.

当然要提的是Belloni在随后的另一篇,他提出了一个被称之为double-selection的方法去处理遗漏变量偏差问题Inference on Treatment Effects after Selection among High-Dimensional Controls, Belloni, The Review of Economic Studies, 2014

那么Lasso筛选变量,双选择减少遗漏变量问题,OLS估计,那么很自然的自相关异方差的Newy-West调整,内生性花式IV甚至是各种GMM就都能很自然的使用了.



了个小模型,借鉴了一下Airbnb这篇KDD 2018 best,只能说经典论文要多看.

Real-time Personalization using Embeddings for Search Ranking at Airbnb, Mihajlo Grbovic, KDD 2018

听说已经升DS的leader做了个uplift,于是看看相关的,这次仍然是因果推断,学过计量的应该很快就能理解到里面的思路,不一定非要揪着里面的DL不放.

Addressing Exposure Bias in Uplift Modeling for Large-scale Online Advertising, Wenwei Ke, ICDM 2021

https://github.com/aifor/eeuen

EEUEN显式曝光uplift效应网络
1.随机对照试验RCT,获得同质试验组并利用RCT数据,提出显示估计uplift的网络
2.考虑T到Y之间的非必然关系(即广告中的上架不等于曝光)
3.通过联合学习显式估计函数和解决处理的曝光偏差来模拟在线广告的uplift

$(x_{i},y_{i},y_{i},t_{i},e_{i})$
$x_{i}$:用户广告的处理.嵌入协变量
$y_{i}$:是否转化
$t_{i}$:是否进处理组
$e_{i}$:广告曝光
有unbias的ITE(个体处理效应)
$\gamma_{x_{i}}=y_{i}(1)-y_{i}(0)$
由于前者不可观测,估计CATE(条件平均处理效应)
$\gamma_{x_{i}}=E[Y(1)-Y(0)|X=xi]=E[Y|T=1,X=xi]-E[Y|T=0,X=xi]$
要求X,Y,T之间没有内生性问题
由于杂七杂八的问题进入T不代表真正曝光,这时候无法准确估计CATE
因此引入进组到曝光的概率函数
$\omega(X)=P(e_{i}=1|X=x_{i})$
有E(Y|T=1,E,X)和E(Y|T=0,E,X)的处理组和控制组
于是有
$E(Y|T=1,E,X)=\omega(X)*P_{T}(X,1)+(1-\omega(X)))*P_{T}(X,0)$
$E(Y|T=0,E,X)=\omega(X)*P_{C}(X,1)+(1-\omega(X))*P_{C}(X,0)$
无曝光的部分有
$\omega(X)=\omega(X)*\gamma_{e}(X)$
曝光的uplift为
$\gamma_{e}(X)=P_{T}(X,1)-P_{C}(X,1)$
于是
$\mu_{t}(X)=-\mu_{c}(X)+\omega(X)*\gamma_{e}(X)$
通过modeling估计曝光概率和uplift

模型基础输入
1.上下文（如网络状态、应用程序名称）
2.用户行为（如点击和购买历史）
3.用户配置文件（如年龄、性别）
4.候选广告（如品牌信息）
离散特征OneHot后concat,连续特征直接进

EUEN模型架构:
联合学习,同时给定的2个损失函数
$\hat{\theta}_{\lambda}=arg \enspace min \enspace \mathcal{J}(\theta)+\lambda*\mathcal{L}(\theta)$

$\mathcal{J}(\theta)=\sum_{i}t_{i}*[y_{i}-(\gamma(x_{i})+\mu_{c}(x_{i}))]^{2}$用于衡量处理组的曝光的损失函数

$\mathcal{L}(\theta)=\sum_{i}(1-t_{i})*[y_{i}-\mu_{c}(x_{i})]^{2}$用于衡量控制组曝光的损失函数

$\lambda>0$是用于平衡两者损失的超参数

EUEN训练过程

1.对照组的样本进入控制组网络(拿停止梯度)

2.处理组样本进入uplift网络

$\mathcal{J}(\theta)$会作为模型总损失更新网络所有参数,$\mathcal{L}(\theta)$则否

EEUEN

$\hat{\theta}_{\lambda,\gamma}=arg \enspace min \enspace \mathcal{J}(\theta)+\lambda*\mathcal{L}(\theta)+\gamma*\mathcal{F}(\theta)$

加入了进组-曝光的概率估计的loss,此时$\mathcal{J}(\theta)$和$\mathcal{F}(\theta)$为

$\mathcal{J}(\theta)=\sum_{i}t_{i}*[y_{i}-(\gamma_e(x_{i})*\omega(x_{i})+\mu_{c}(x_{i}))]^{2}$

$\mathcal{F}(\theta)=-\sum_{i}t_{i}*[e_{i}*log \enspace\omega(x_{i})+1-(e_{i})*log(1-\omega(x_{i}))]$

$\gamma$>0是用于平衡三者损失的另一个超参数

EEUEN训练过程

1.对照组的样本进入控制组网络(拿停止梯度)

2.曝光网络部分用处理组样本进入,label用真实曝光与否

3.uplift网络部分用处理组样本进入

其中1.3与EUEN相似,同样的$\mathcal{J}(\theta)$会作为模型总损失更新网络所有参数