import numpy as np
from scipy.stats import chi2, f
from numpy.random import rand
from sklearn.preprocessing import scale
from statsmodels.api import OLS
from numpy import log, round, ndarray


class TerasvirtaTest:
    """
    x : ndarray, n*k, y的影响变量
    y : ndarray, n*1, 需要检验是否线性的结果
    do_scale: bool, 做不做z-score
    对于影响y的x变量上究竟是线性还是非线性的是值得关注
    线性表现传统会被归到模型错误设定上,此时会有拉姆齐(1969)的RESET检验,该检验通过构造带约束的线性模型和无约束带高次项的模型进行F检验.
    但除了传统的多项式,(前馈)神经网络由于能较好的拟合数据被统计学家所考虑.White(1989)提出了White Test,
    通过使用能被写成线性回归形式的神经网络来进行Chi2检验和F检验,但White并没有给出神经网络的具体建议.
    Terasvirta(1999)基于在White Test的基础上,发现使用V23(X变量两两的二次quadratic和三次cubic项)
    在无优势非线性设定的时候能提高power更优的检验,该检验与约束和非约束检验一致,服从
    nR^2的Chi2(p)和基于SSR的F(p,n-p-k),其中n为样本量,p为无约束增加的变量,k是约束模型的变量
    如果Chi2.F检验拒绝了原假设则认为x对y的影响存在非线性部分

    Tests for Specification Errors in Classical Linear Least Squares Regression, Journal of the Royal
    Statistical Society, Series B, Ramsey, 1969.
    An Additional Hidden Unit Test for Neglected Nonlinearity in Multilayer Feedforward Networks,
    Proceedings of The International Joint Conference on Neural Networks, White 1989
    Power of the Neural Networks Linearity Test, Journal of Time Series Analysis, Terasvirta, 1993

    p.s.值得一提的是,对"神经网络"进行假设检验会让人有种"黑盒,嘿嘿...打开,嘿嘿...看看,嘿嘿...woow!"的联想,
    但实质上这里的NN是可以写成解析形式的,并且因为同样原因才能使用Chi2和F的约束检验方法
    对于机器学习特别是带黑盒部分的例如有RF和我们熟知的神经的网络的假设检验及相关框架,则在很遥远之后才被比较系统化的建立起来
    """

    def est(self):
        if self.do_scale:
            self.x = scale(self.x)
            self.y = scale(self.y)

        linear = OLS(self.y, self.x).fit()
        u = linear.resid
        ssr0 = np.sum(u ** 2)
        deal_x = self.x.copy()
        m = 0
        for i in range(self.x_var):
            for j in range(i + 1):
                temp = self.x[:, i] * self.x[:, j]
                deal_x = np.hstack((deal_x, temp.reshape(-1, 1)))
                m += 1
        for i in range(self.x_var):
            for j in range(i + 1):
                for k in range(j + 1):
                    temp = self.x[:, i] * self.x[:, j] * self.x[:, k]
                    deal_x = np.hstack((deal_x, temp.reshape(-1, 1)))
                    m += 1
        nonlinear = OLS(u, deal_x).fit()
        v = nonlinear.resid
        ssr = np.sum(v ** 2)
        self.chi2_stat = round(self.nobs * log(ssr0 / ssr), 4)
        self.dof1 = m
        self.dof2 = self.nobs - self.x_var - m
        self.chi2_p = round(1 - chi2.cdf(self.chi2_stat, self.dof1), 4)
        self.f_stat = round(
            ((ssr0 - ssr) / m) / (ssr / (self.nobs - self.x_var - m)), 4
        )
        self.f_p = round(1 - f.cdf(self.f_stat, self.dof1, self.dof2), 4)
        print(
            "dof1:",
            self.dof1,
            "dof2:",
            self.dof2,
            "\nchi2_stat:",
            self.chi2_stat,
            "chi2_p:",
            self.chi2_p,
            "\nF_stat:",
            self.f_stat,
            "F_p:",
            self.f_p,
        )

    def __init__(self, x: ndarray, y: ndarray, do_scale: bool):
        self.x = x
        self.y = y
        self.x_var = self.x.shape[1]
        self.nobs = self.x.shape[0]
        self.do_scale = do_scale
        self.chi2_stat = 0
        self.dof1 = 0
        self.dof2 = 0
        self.chi2_p = 0
        self.f_p = 0
        self.f_stat = 0


x_test = rand(2000, 5)
y_test = rand(2000, 1)
do_scale_test = True
TerasvirtaTest(x_test, y_test, do_scale_test).est()
