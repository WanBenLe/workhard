import numpy as np
from sklearn.metrics import mean_squared_error
from numpy import array, zeros, cov, kron, ones, diag, trace, log, diag_indices_from
from numpy.linalg import inv, det
from numpy.random import rand, seed
from numba import jit
from scipy.optimize import minimize

seed(42)


def array_mat(mat_array, i_shape):
    mat_gl = zeros((i_shape, i_shape))
    index = 0
    # 上三角矩阵
    for i in range(i_shape):
        mat_gl[i, i:] = mat_array[index : (i_shape - i) + index]
        index += i_shape - i
    # 对称矩阵
    mat_gl += mat_gl.T - diag(mat_gl.diagonal())
    return mat_gl


class Fg_Lasso:
    def __init__(self, x, y, lambda_lasso):
        self.x = x
        self.y = y
        self.mem = x.shape[0]
        self.t = x.shape[1]
        self.x_vars = x.shape[2]
        self.inv_cov_res = zeros((self.mem, self.mem))
        self.cov_res = zeros((self.mem, self.mem))
        self.lambda_lasso = lambda_lasso
        self.mat_gl = zeros((self.mem, self.mem))
        self.beta_fglasso = zeros((self.x_vars, 1))
        self.beta_fgls = zeros((self.x_vars, 1))

        self.post_ols()
        self.fit_omega_gl(disp=1)
        self.fg_lasso_est()
        self.fgls_est()

    def post_ols(self):
        # 先做OLS然后计算残差得到可行的cov
        beta_ols = zeros((self.mem, self.x_vars))
        residual = zeros((self.mem, self.t))
        for i, x_i in enumerate(self.x):
            y_i = self.y[i].reshape(-1, 1)
            beta_ols[i] = (inv(x_i.T @ x_i) @ (x_i.T @ y_i)).reshape(-1)
            residual[i] = y_i.reshape(-1) - beta_ols[i] @ x_i.T
        self.cov_res = residual @ residual.T / self.x_vars
        self.inv_cov_res = inv(self.cov_res)

    def fit_omega_gl(self, disp=0):
        vars = int((self.mem**2 + self.mem) / 2)
        x0 = [0.001] * vars  # 初始化参数——等权
        # 各个权重应该大于等于0的约束
        bnds = eval("(0, None)," * vars)
        res = minimize(
            self.min_omega_gl,
            x0,
            args=(self.mem, self.cov_res, self.lambda_lasso),
            method="SLSQP",
            tol=1e-7,
            bounds=bnds,
        )
        mat_values = res.x
        self.mat_gl = array_mat(mat_values, self.mem)
        if disp != 0:
            print("Omega_GL权重矩阵:\n", np.round(self.mat_gl, 4))
            print("Omega_GL损失函数:", res.fun)

    def fg_lasso_est(self):
        fglasso_1 = zeros((self.x_vars, self.x_vars))
        fglasso_2 = zeros((self.x_vars, self.y.shape[2]))
        for i in range(self.t):
            fglasso_1 += self.x[:, i, :].T @ self.mat_gl @ self.x[:, i, :]
            fglasso_2 += self.x[:, i, :].T @ self.mat_gl @ self.y[:, i, :]
        self.beta_fglasso = inv(fglasso_1) @ fglasso_2

    def fgls_est(self):
        # FGLS的系数,可以iter做连续更新,我摸鱼
        fgls_1 = zeros((self.x_vars, self.x_vars))
        fgls_2 = zeros((self.x_vars, self.y.shape[2]))
        for i in range(self.t):
            fgls_1 += self.x[:, i, :].T @ self.inv_cov_res @ self.x[:, i, :]
            fgls_2 += self.x[:, i, :].T @ self.inv_cov_res @ self.y[:, i, :]
        self.beta_fgls = inv(fgls_1) @ fgls_2

    @staticmethod
    def min_omega_gl(para_set, i_shape, res_cov, lambda_lasso):
        mat_gl = array_mat(para_set, i_shape)
        calc_mat = mat_gl
        row, col = diag_indices_from(calc_mat)
        calc_mat[row, col] = 0
        # L1正则项可以CV出来,我摸鱼
        loss = (
            trace(mat_gl @ res_cov) - log(det(mat_gl)) + lambda_lasso * np.sum(calc_mat)
        )
        return loss


"""
一般来说会有一些面板数据例如下面那样子的:
很多个国家的100天的总销量和这个国家每天的一些属性,例如live产品数.live店铺数.GDP等
一般来说宏观层面会试图估计出来每天的属性对总体会产生什么影响(而不是每个国家都一个公式)
但是这种情况下国家跟国家间是有一定关系的,在这种前提下不能做原本的iid假设而要做系统方程估计
Further Properties of Efficient Estimators for Seemingly Unrelated Regression Equations,
Arnold Zellner, International Economic Review, 1962
此时SUR被提出,SUR首先估计每个方程,然后用方程间的cov做GLS估计总体参数,
但是cov通常是未知的,所以有通过残差的cov作为可行cov的估计被称为FGLS
但是对于短T大n数据(如上面例子天数很少,影响变量很多),FGLS的SUR在渐进上面性质不佳
Estimation of high-dimensional seemingly unrelated regression models, Lidan Tan, Econometric Reviews, 2021
于是Lidan Tan等人提出FGLasso,使用带L1正则项的实对称权重矩阵优化代替残差cov,并统计学上证明了在短面板数据上cov和系数估计的渐进无偏性
"""
xall = rand(50, 100, 10)
yall = rand(50, 100, 1)
mdl = Fg_Lasso(xall, yall, 0.2)
print(np.round(mdl.beta_fglasso, 4).reshape(-1))
print(np.round(mdl.beta_fgls, 4).reshape(-1))
