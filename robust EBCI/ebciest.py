from numpy import ones, sqrt, abs, kron, sort, unique, concatenate, \
    max, diff, argwhere, empty, min, argmax, arange, hstack, sum, array, nan
from numpy.linalg import inv, pinv
from numba import jit
from scipy.stats import norm
from cvaest import cva

def length_fct(ww, mu2_over_sigmasq, kappa, alpha):
    return ww * cva((1 / ww[0] - 1) ** 2 * mu2_over_sigmasq, alpha,kappa=kappa)[0]


def robust_ebci(w, mu2_over_sigmasq, kappa, alpha):
    if len(w) == 0:
        the_start = mu2_over_sigmasq / (1 + mu2_over_sigmasq)
        bnds = eval("(the_start, 1)," * 1)
        xinit = init_x0(the_start, 1)

        optr = minimize(length_fctopt, xinit, bounds=bnds, args=(mu2_over_sigmasq, kappa, alpha), method='TNC')
        w_estim, normlng = optr.x[0], optr.fun

    else:
        w_estim = w
        normlng = length_fct(w, mu2_over_sigmasq, kappa, alpha)
    return w_estim, normlng


def parametric_ebci(mu2_over_sigmasq, alpha):
    w_eb = mu2_over_sigmasq / (1 + mu2_over_sigmasq)
    z = norm.ppf(1 - alpha / 2)
    lngth = sqrt(w_eb) * z
    return w_eb, lngth, z


def moment_conv(Y, sigma, weights):
    if len(weights) == 0:
        weights = ones(len(Y))
    weights = weights / sum(weights)
    W2 = Y ** 2 - sigma ** 2
    W4 = Y ** 4 - 6 * (sigma * Y) ** 2 + 3 * sigma ** 4
    mu2_t = float(weights.T @ W2)
    mu4_t = float(weights.T @ W4)
    # only PMT
    mu2 = max(array([mu2_t, float(2 * ((weights ** 2).T @ (sigma ** 4)) / (weights.T @ (sigma ** 2)))]))
    kappa = max(array([mu4_t / mu2 ** 2,
                       float(1 + 32 * ((weights ** 2).T @ (sigma ** 8)) / (mu2 ** 2 * (weights.T @ (sigma ** 4))))]))
    return mu2, kappa


def ebci(Y, X, sigma, alpha, weight=array([])):
    Y_norm = Y

    if len(weight) == 0:
        weights = ones((len(Y_norm), 1))
    else:
        weights = weight
    if len(X) > 0:
        X_weight = sqrt(weights) * X
        Y_norm_weight = sqrt(weights) * Y_norm
        delta = pinv(X_weight[weights.reshape(-1) != 0, :]) @ (Y_norm_weight[weights != 0])
        mu1 = (X @ delta).reshape(-1, 1)
    else:
        mu1 = 0
        delta = array([])

    mu2 = array([])
    kappa = array([])

    if len(mu2) == 0:
        mu2, kappa = moment_conv(Y - mu1, sigma, weights)

    w_eb, lngth_param, a = parametric_ebci(mu2 / (sigma ** 2), alpha)
    if mu2 > 10 ** -8:
        w = w_eb
        kappa_cv = kappa
        n = len(Y)
        w_estim = ones((n, 1))
        w_estim[:] = nan
        normlng = ones((n, 1))
        normlng[:] = nan
        for i in range(n):
            the_w = w[i]
            w_estim[i], normlng[i] = robust_ebci(the_w, mu2 / (sigma[i][0] ** 2), kappa_cv, alpha)
    else:
        w_estim = w_eb
        normlng = 0
    thetahat = mu1 + w_estim* (Y_norm - mu1)
    ci = thetahat + (normlng * sigma) * array([-1 ,1])
    return thetahat, ci, w_estim, normlng, mu2, kappa, delta
