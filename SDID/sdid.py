'''
Copyright <2021> <Ben Wan: wanbenfighting@gmail.com>
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix
from sklearn.metrics import mean_squared_error
import warnings

warnings.simplefilter('ignore')
cvxopt.solvers.options['show_progress'] = False


# Synthetic Difference-in-Differences, Dmitry Arkhangelsky, NBER & AER, 2021
# 不想写注释了,中文的自己看这个https://www.shangyexinzhi.com/article/4890167.html

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def est_omega(Y_c_pre, Y_t_pre, zeta=1):
    Y_c_pre = Y_c_pre.T
    nrow = Y_c_pre.shape[0]
    ncol = Y_c_pre.shape[1]
    assert nrow == Y_t_pre.shape[0], print(f'shape error! {nrow} != {Y_t_pre.shape[0]}')

    P = np.diag(np.concatenate([np.repeat(zeta, ncol), np.repeat(1 / nrow, nrow)]))

    q = np.zeros(ncol + nrow)
    A = np.concatenate([np.concatenate([Y_c_pre, np.diag(np.ones(nrow))], axis=1),
                        np.concatenate([np.ones(ncol), np.zeros(nrow)]).reshape(1, -1)])
    b = np.concatenate([Y_t_pre, np.ones(1)])
    G = - np.concatenate([np.diag(np.ones(ncol)), np.zeros([ncol, nrow])], axis=1)
    h = np.zeros(ncol)
    P, q, A, b, G, h = matrix(P), matrix(q), matrix(A), matrix(b), matrix(G), matrix(h)
    sol = cvxopt.solvers.qp(P=P, q=q, A=A, b=b, G=G, h=h)
    return np.array(sol['x'][:ncol])


def est_lambda(Y_c_pre, Y_c_post, zeta=1):
    nrow = Y_c_pre.shape[0]
    ncol = Y_c_pre.shape[1]
    assert nrow == Y_c_post.shape[0], print(f'shape error! {nrow} != {Y_c_post.shape[0]}')
    P = np.diag(np.concatenate([np.repeat(zeta, ncol), np.repeat(1 / nrow, nrow), np.array([1e-6])]))
    q = np.zeros(ncol + nrow + 1)
    A = np.concatenate([np.concatenate([Y_c_pre, np.diag(np.ones(nrow)), -np.ones([nrow, 1])], axis=1),
                        np.concatenate([np.ones(ncol), np.zeros(nrow), np.zeros(1)]).reshape(1, -1)])
    b = np.concatenate([Y_c_post, np.ones(1)])
    G = - np.concatenate([np.diag(np.ones(ncol)), np.zeros([ncol, nrow]), np.zeros([ncol, 1])], axis=1)
    h = np.zeros(ncol)
    P, q, A, b, G, h = matrix(P), matrix(q), matrix(A), matrix(b), matrix(G), matrix(h)
    sol = cvxopt.solvers.qp(P=P, q=q, A=A, b=b, G=G, h=h)
    return np.array(sol['x'][:ncol])


def sdid(Y, focal_time=30, w=np.array([]), plot='false'):
    sigma_sq = np.var(Y[:, 1:] - Y[:, :-1])
    Y_c = np.delete(Y, 0, axis=0)
    Y_t = Y[0, :]
    Y_c_pre = Y_c[:, :focal_time]
    Y_c_post = np.mean(Y_c[:, focal_time:], axis=1)
    Y_t_pre = Y_t[:focal_time]
    # Y_t_post = np.mean(Y_t[focal_time:])
    if len(w) == 0:
        omega_hat = est_omega(Y_c_pre, Y_t_pre, zeta=sigma_sq)
    else:
        omega_hat = w.reshape(-1, 1)
    lambda_hat = est_lambda(Y_c_pre, Y_c_post, zeta=sigma_sq)
    sum_omega_YiT = omega_hat.T @ Y_c_post
    sum_lambda_YNt = lambda_hat.T @ Y_t_pre
    sum_omega_lambda_Yit = omega_hat.T @ Y_c_pre @ lambda_hat
    Yhat_sdid = sum_omega_YiT + sum_lambda_YNt - sum_omega_lambda_Yit

    if plot.lower() != 'false':
        contrl_pre = (omega_hat.T @ Y_c_pre).reshape(-1)
        tre_pre = (Y_t_pre).reshape(-1)
        contrl_post = (omega_hat.T @ Y_c[:, focal_time:]).reshape(-1) + (lambda_hat.T @ Y_t_pre)[0] - \
                      (sum_omega_YiT + sum_lambda_YNt - sum_omega_lambda_Yit)[0][0]
        tre_post = Y_t[focal_time:].reshape(-1)
        a = np.concatenate((contrl_pre, contrl_post))
        b = np.concatenate((tre_pre, tre_post))
        plt.plot(a)
        plt.plot(b)
        plt.legend(["SDID contrl", "Tre"])
        plt.title('Tre time:' + str(int(focal_time)))
        plt.show()
        plt.close()

        # 可以调整,也可以不调整,论文上来说可能不调整更好
        adj = np.mean(a[:focal_time] - b[:focal_time])
        adja = a - adj
        # adj = 1
        plt.plot(adja)
        plt.plot(b)
        plt.title('adj ratio with:' + str(np.round(adj, 2)) + 'Tre time:' + str(int(focal_time)))
        plt.legend(["SDID contrl", "Tre"])
        plt.show()
        plt.close()

        plt.plot(b - adja)
        plt.title('diff,adj ratio with:' + str(np.round(adj, 2)))
        plt.show()

        profit_d = np.round(np.sum((b - adja)[focal_time:]) / np.sum((adja)[focal_time:]) * 100, 2)
        print(str(profit_d), '%adj指标增幅')
        profit_d = np.round(np.sum((b - a)[focal_time:]) / np.sum((a)[focal_time:]) * 100, 2)
        print(str(profit_d), '%不调整指标增幅')

        print(root_mean_squared_error(contrl_pre - adj, tre_pre))
        print(root_mean_squared_error(contrl_pre, tre_pre))
    return Yhat_sdid, omega_hat, lambda_hat


# placebo_se
def norm_omega(omega_boot):
    sum_omega = np.sum(omega_boot)
    if sum_omega == 0:
        return np.repeat(1 / len(omega_boot), len(omega_boot))
    else:
        return omega_boot / sum_omega


def placebo_se(Y, omega_sdid, boot_n=100):
    Y = Y[1:]
    sdid_all = []
    for i in range(boot_n):
        omega_boot = omega_sdid.reshape(-1)
        # 扔掉处理组数据
        choice_T = np.random.choice(np.arange(0, len(Y)), 1)[0]
        Y[0], Y[choice_T] = Y[choice_T], Y[0]
        omega_boot[0], omega_boot[choice_T] = omega_boot[choice_T], omega_boot[0]
        omega_boot = norm_omega(omega_boot[1:])
        # print(omega_boot)
        sdidf, d1, d2 = sdid(Y, focal_time=30, w=omega_boot)
        sdid_all.append(sdidf)
    return ((boot_n - 1) / boot_n) ** 0.5 * np.std(sdid_all)

