import numpy as np
from numpy import ones, sqrt, abs, kron, sort, unique, concatenate, \
    max, diff, min, argmax, arange, hstack, inf,  zeros_like,  ndarray, float64
from numba import jit
from scipy.stats import ncx2
from scipy.optimize import minimize, linprog, root_scalar
from MyMath import norm_cdf, norm_ppf, norm_pdf



def norm_cdf_f(a):
    if type(a) == float:
        return norm_cdf(a)
    else:
        b = zeros_like(a)
        for i in range(len(a)):
            b[i] = norm_cdf(a[i])
        return b



def norm_pdf_f(a):
    if (type(a) == float) or (type(a) == float64):
        return norm_pdf(float(a))
    else:
        b = zeros_like(a)
        for i in range(len(a)):
            b[i] = norm_pdf(a[i])
        return b



def norm_ppf_f(a):
    if type(a) == float:
        return norm_ppf(a)
    else:
        b = zeros_like(a)
        for i in range(len(a)):
            b[i] = norm_ppf(a[i])
        return b


@jit(fastmath=True)
def init_x0(ax, bx):
    x0 = ax + (0.5 * (3.0 - sqrt(5.0))) * (bx - ax)
    return x0


@jit(fastmath=True)
def lammax(x0, chi, t0, ip, tbar):
    if x0 >= tbar:
        val = delta(0, x0, chi)
    else:
        val = lam(x0, chi, t0, ip)
    return val


@jit(fastmath=True)
def objopt(x0, chi, m2, t0, ip, tbar, kappa):
    opt = float(r(x0, chi)) + r1(x0, chi) * (m2 - x0) + lammax(x0, chi, t0, ip, tbar)[0] * (
            kappa * m2 ** 2 - 2 * x0 * m2 + x0 ** 2)
    return float(opt)


@jit(fastmath=True)
def delta(x, x0, chi):
    if type(x) == ndarray:
        if x.shape < 2:
            x.reshape(-1, 1)
    else:
        x = np.array([x]).reshape(-1, 1)
    idx = (abs(x - x0) >= 1e-4)
    val = repmat(r2(x0, chi) / 2, x.shape[0], x.shape[1])
    val[:, idx.reshape(-1)] = ((r(x[idx], chi) - r(x0, chi)).reshape(-1) - r1(x0, chi) * (x[idx] - x0)) / (
            x[idx] - x0) ** 2
    return val


@jit(fastmath=True)
def negdeltaopt(x, x0, chi):
    if type(x) == ndarray:
        if x.shape < 2:
            x.reshape(-1, 1)
    else:
        x = np.array([x]).reshape(-1, 1)
    idx = (abs(x - x0) >= 1e-4)
    val = repmat(r2(x0, chi) / 2, x.shape[0], x.shape[1])
    val[:, idx.reshape(-1)] = ((r(x[idx], chi) - r(x0, chi)).reshape(-1) - r1(x0, chi) * (x[idx] - x0)) / (
            x[idx] - x0) ** 2
    return float(-val)


@jit(fastmath=True)
def r3(t, chi):
    if type(t) == ndarray:
        if t.shape[0] < 2:
            t.reshape(-1, 1)
    else:
        t = np.array([t]).reshape(-1, 1)

    idx = (t >= 2e-4)

    val = repmat(norm_pdf_f(chi) * (chi ** 5 - 10 * chi ** 3 + 15 * chi) / 60, t.shape[0], t.shape[1])
    tidx = t[idx]
    val[idx] = (norm_pdf_f(chi - sqrt(tidx)) * (
            tidx ** 2 - 2 * chi * tidx ** (3 / 2) + (2 + chi ** 2) * tidx - 3 * chi * sqrt(tidx) + 3) -
                norm_pdf_f(chi + sqrt(tidx)) ** (
                        tidx ** 2 + 2 * chi * tidx ** (3 / 2) + (2 + chi ** 2) * tidx + 3 * chi * sqrt(
                    tidx) + 3)) / (8 * tidx ** (5 / 2))
    return val


@jit(fastmath=True)
def delta1(x, x0, chi):
    if type(x) == ndarray:
        if x.shape < 2:
            x.reshape(-1, 1)
    else:
        x = np.array([x]).reshape(-1, 1)
    idx = (abs(x - x0) >= 1e-3)
    val = repmat(r3(x0, chi) / 6, x.shape[0], x.shape[1])
    val[:, idx.reshape(-1)] = ((r1(x[idx], chi) + r1(x0, chi)) - 2 * (r(x[idx], chi) - r(x0, chi)).reshape(1, -1) / (
            x[idx] - x0)) / (x[idx] - x0) ** 2
    return val


def lam(x0, chi, t0, ip):
    xs = sort(np.array([t0, ip]))
    if x0 >= xs[0]:
        xs = np.array([0, xs[0]])
    else:
        xs = unique(concatenate((np.array([0]), np.array([x0]).reshape(-1), xs)))

    vals = delta(xs, x0, chi).reshape(-1)
    ders = delta1(xs, x0, chi).reshape(-1)

    val = vals[0]
    xmax = 0

    if (ders <= 0).all() and vals[0] == max(vals):
        return vals, xmax
    elif (diff((ders >= 0).astype(int)) <= 0).all() and ders[-1] <= 0:
        try:
            the_ind = max(np.array([np.argwhere(ders < 0)[0][0], 1]))
        except:
            the_ind = len(xs) - 1
        the_start = xs[the_ind - 1]
        the_end = xs[the_ind]
    elif (min(abs(ders)) < 1e-6):
        the_ind_max = argmax(vals)
        the_start = xs[max(the_ind_max - 1, 1)]
        the_end = xs[min(the_ind_max + 1, len(xs) - 1)]
    else:
        print('multiple local optima')
    bnds = eval("(the_start, the_end)," * 1)
    xinit = init_x0(the_start, the_end)
    optr = minimize(negdeltaopt, xinit, bounds=bnds, args=(x0, chi), method='TNC')
    [the_xmax, mdelta] = optr.x[0], optr.fun

    if -mdelta > vals[0]:
        val = -mdelta
        xmax = the_xmax
    elif - mdelta < vals[0] - 1e-9:
        print('Optimum may be wrong for lam')

    return val, xmax


@jit(fastmath=True)
def repmat(a, b, c):
    return kron(ones((c, b)), a)



def r1(t, chi):
    if type(t) == ndarray:
        if t.shape[0] < 2:
            t.reshape(-1, 1)
    else:
        t = np.array([t]).reshape(-1, 1)
    idx = (t >= 1e-8)
    val = repmat(chi * norm_pdf_f(chi), t.shape[0], t.shape[1])
    try:
        val[idx.reshape(-1)] = (norm_pdf_f(sqrt(t[idx]) - chi) - norm_pdf_f(sqrt(t[idx]) + chi)) / (2 * sqrt(t[idx]))
    except:
        try:
            val[:, idx.reshape(-1)] = (norm_pdf_f(sqrt(t[idx]) - chi) - norm_pdf_f(sqrt(t[idx]) + chi)) / (
                    2 * sqrt(t[idx]))
        except:
            return chi * norm_pdf_f(chi)
    return (val)



def r2(t, chi):
    if type(t) == ndarray:
        if t.shape[0] < 2:
            t.reshape(-1, 1)
    else:
        t = np.array([t]).reshape(-1, 1)
    idx = (t >= 2e-6)
    val = repmat(norm_pdf_f(chi) * chi * (chi ** 2 - 3) / 6, t.shape[0], t.shape[1])
    tidx = t[idx]
    val[idx] = (norm_pdf_f(sqrt(tidx) + chi) * (chi * sqrt(tidx) + tidx + 1) + norm_pdf_f(sqrt(tidx) - chi) * (
            chi * sqrt(tidx) - tidx - 1)) / (4 * tidx ** (3 / 2))
    return float(val)



def f0(tt, chi):
    return float(r(tt, chi)) - tt * float(r1(tt, chi)) - float(r(0, chi))


def rt0(chi):
    if chi < sqrt(3):
        t0 = 0
        ip = 0
    else:

        if (abs(r2(chi ** 2 - 3 / 2, chi)) < 1e-12) or ((chi ** 2 - 3) == chi ** 2):
            ip = chi ** 2 - 3 / 2
        else:

            ip = root_scalar(r2, args=(chi), method='toms748', bracket=[chi ** 2 - 3, chi ** 2]).root

        # f0=@(tt) r(tt, chi) - tt* r1(tt, chi) - r(0, chi)
        lo = ip
        up = 2 * chi ** 2
        while f0(up, chi) < 0:
            lo = up
            up = 2 * up
        t0 = lo
        if f0(lo, chi) < 0:
            t0 = root_scalar(f0, args=(chi), method='toms748', bracket=[lo, up]).root
        elif f0(lo) > 1e-12:
            print('opt error')
    return t0, ip



def r(t, chi):
    if type(t) == ndarray:
        if t.shape[0] < 2:
            t.reshape(-1, 1)
    else:
        t = np.array([t]).reshape(-1, 1)
    idx = (sqrt(t) - chi <= 5)
    val = ones(t.shape)
    # print((-sqrt(t[idx]) - chi).shape)
    val[idx] = norm_cdf_f(-sqrt(t[idx]) - chi) + norm_cdf_f(sqrt(t[idx]) - chi)
    return val



def rho0(t, chi):
    t0, ip = rt0(chi)

    if t < t0:
        lf_t = np.array([0, t0])
        lf_p = np.array([1 - t / t0, t / t0])
    else:
        lf_t = t
        lf_p = 1

    maxnoncov = float(lf_p @ r(lf_t, chi))
    return maxnoncov, t0, ip, lf_p, lf_t


def rho(m2, chi,kappa=inf):
    if kappa == 1:
        return r(m2, chi)

    maxnoncov, t0, ip, lf_p, lf_t = rho0(m2, chi)

    if lf_p @ (lf_t ** 2) <= kappa * m2 ** 2:
        return maxnoncov, lf_t, lf_p
    a, tbar = lam(0, chi, t0, ip)
    if tbar > 0:
        bnds = eval("(0, tbar)," * 1)

        x0 = init_x0(0, tbar)
        optr = minimize(objopt, x0, bounds=bnds, args=(chi, m2, t0, ip, tbar, kappa), method='Nelder-Mead')
        [x0opt_below, maxnoncov_below] = optr.x[0], optr.fun
    else:
        x0opt_below = 0
        maxnoncov_below = objopt(0, chi, m2, t0, ip, tbar, kappa)

    bnds = eval("(tbar, t0)," * 1)
    x0 = init_x0(tbar, t0)
    optr = minimize(objopt, x0, bounds=bnds, args=(chi, m2, t0, ip, tbar, kappa),
                    method='TNC')
    [x0opt, maxnoncov] = optr.x[0], optr.fun
    if maxnoncov_below < maxnoncov:
        x0opt = x0opt_below
        maxnoncov = maxnoncov_below
    [a, xopt] = lam(x0opt, chi, t0, ip)
    lf_t = sort([x0opt, xopt])
    p = (m2 - lf_t[1]) / (lf_t[0] - lf_t[1])
    lf_p = np.array([p, 1 - p])
    primal_maxnoncov = float(r(lf_t, chi).T @ lf_p)

    primal_m2 = lf_t @ lf_p
    primal_kappa = (lf_t ** 2) @ lf_p / primal_m2 ** 2
    if max(np.array([maxnoncov, m2, kappa]) - np.array([primal_maxnoncov, primal_m2, primal_kappa])) > 1e-4:
        xs = sort(unique(concatenate((arange(5000) / 4999 * t0, np.array([m2]), lf_t))))
        numgrid = len(xs)
        bnds = eval("(0, None)," * numgrid)
        A_ub = (xs ** 2).reshape(1, -1)
        b_ub = kappa * m2 ** 2
        A_eq = hstack((ones(numgrid).reshape(-1, 1), xs.reshape(-1, 1))).T
        b_eq = np.array([1, m2])
        result = linprog(-r(xs, chi), A_ub=A_ub, b_ub=b_ub, bounds=bnds, A_eq=A_eq, b_eq=b_eq)
        opt = -result.fun
        if abs(opt - maxnoncov) > 1e-4:
            print('Linear program finds non-coverage ', opt,
                  'Direct approach finds non-coverage ', maxnoncov,
                  'Difference>0.001. This happened for chi=', chi, ', mu_2=', m2, ', kappa=', kappa)
    return maxnoncov, lf_t, lf_p


def cvaupopt(cchi, m2, kappa, alpha):
    a, b, c = rho(m2, cchi, kappa=kappa)
    return a - alpha


def CVb(B, alpha):
    if type(B) == ndarray:
        if B.shape < 2:
            B.reshape(-1, 1)
    else:
        B = np.array([B]).reshape(-1, 1)

    idx = (B < 10)
    val = B + norm_ppf_f(1 - alpha)
    val[idx] = sqrt(ncx2.ppf(1 - alpha, 1, B[idx] ** 2))
    return val



def cva(m2, alpha, kappa=inf):
    if (m2 == 0) or (kappa == 1):
        chi = CVb(sqrt(m2), alpha)
        lf_t = m2
        lf_p = 1
    else:
        if (1 / m2 < 1e-12) and (kappa < inf):
            chi, lf_t, lf_p = cva(m2, alpha, kappa=inf)
            if kappa < ((lf_t ** 2).T @ lf_p) / m2 ** 2:
                print('Assuming kappa constraint not binding.')
            return chi, lf_t, lf_p
        lo = float(0.99 * CVb(sqrt(m2), alpha))
        up = sqrt((1 + m2) / alpha)

        if abs(rho(m2, up, kappa=inf)[0] - alpha) >= 9e-6:
            up = root_scalar(cvaupopt, args=(m2, inf, alpha), method='toms748', bracket=[lo, up]).root

        if rho(m2,  up, kappa=inf)[0] - alpha >= -1e-5:
            chi = up
        else:
            chi = root_scalar(cvaupopt, args=(m2, kappa, alpha), method='toms748', bracket=[lo, up]).root

        the_alpha = rho(m2, chi, kappa=kappa)[0]
        [a, lf_t, lf_p] = rho(m2, chi, kappa=kappa)
        if abs(the_alpha - alpha) > 0.001:
            print('%s%f', 'The difference between non-coverage and alpha at CV is ', the_alpha - alpha)
    return chi, lf_t, lf_p


'''
print(cva(m2=4, kappa=3, alpha=0.05))
print(1)
from line_profiler import LineProfiler
p = LineProfiler()
p.add_function(r2)
p.add_function(r)
p.add_function(r1)
p_wrap = p(cva)
p_wrap(m2=4, kappa=3, alpha=0.05)
p.print_stats()
p.dump_stats('saveName.lprof')
print(1)
'''
