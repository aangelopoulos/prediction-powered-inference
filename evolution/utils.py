import numpy as np
import scipy as sc

def pinball_loss(f_N, theta, q):
    assert(q >= 0 and q <= 1)
    if np.isscalar(f_N):
        return q * (f_N - theta) if f_N >= theta else (1 - q) * (theta - f_N)
    diff_N = f_N - theta
    loss_N = q * diff_N
    idx = np.where(diff_N < 0)[0]
    loss_N[idx] = (1 - q) * -diff_N[idx]
    return loss_N

def get_quantile(y_n, q, y_lb: float = -np.inf, y_ub: float = np.inf):
    assert(q >= 0 and q <= 1)
    sortedy_n = np.sort(y_n)
    if q == 0:
        return y_lb  # or sortedy_n[0]
    if q == 1:
        return y_ub  # or sortedy_n[-1]
    how_many = int(np.ceil(q * y_n.size))
    assert(how_many <= y_n.size)
    return sortedy_n[how_many - 1]

def get_classical_ci(y_n, q, alpha, y_lb: float = -np.inf, y_ub: float = np.inf):
    n = y_n.size
    l = int(sc.stats.binom.ppf(alpha / 2, n, q))
    ci_prob = sc.stats.binom.pmf(l, n, q)
    u = l
    while ci_prob < 1 - alpha:
        u += 1
        ci_prob += sc.stats.binom.pmf(u, n, q)
        # TODO: edge case where need lower l to get coverage
    assert(u <= n)
    sortedy_n = np.sort(y_n)
    if l > 0:
        lb = sortedy_n[l - 1]
    else:
        assert(y_lb < sortedy_n[0])
        lb = y_lb
    if u < n:
        ub = sortedy_n[u]  # since should take u + 1
    else:
        assert(y_ub > sortedy_n[-1])
        ub = y_ub
    ql = float(l) / n
    qu = float(u + 1) / n
    assert(ql >= 0 and ql <= 1)
    assert(qu >= 0 and qu <= 1)
    return (lb, ub), ql, qu

def get_quantile_rectifier(y_n, f_n, theta):
    def rectifier_summand(y, f, theta):
        if (theta >= y and theta <= f) or (theta >= f and theta <= y):
            return np.abs(theta - y)
        if (f >= y and f <= theta) or (f >= theta and f <= y):
            return np.abs(f - y)
        return 0
    return np.array([rectifier_summand(y, f, theta) for y, f in zip(y_n, f_n)])

def get_pp_ci(y_n, y_N, f_n, f_N, q, alpha, delta1, delta2, B, quantile_lb, quantile_ub):
    assert(alpha >= 0 and alpha <= 1)
    assert(delta1 + delta2 < alpha)
    assert(delta1 > 0 and delta2 >= 0)

    n = y_n.size
    assert(f_n.size == n)
    N = f_N.size
    assert(y_N.size == N)

    # find q_l and q_u, bounds on population quantile based on N data points
    # (just need N dummy values)
    _, ql, qu = get_classical_ci(
        np.empty([N]), q, alpha - delta1 - delta2, quantile_lb=quantile_lb, quantile_ub=quantile_ub
    )
    print('ql = {:.3f}, qu = {:.3f}'.format(ql, qu))

    # compute imputed upper and lower bounds on quantile
    thetal_imputed = get_quantile(f_N, ql)
    loss_thetal_imputed = np.mean(pinball_loss(f_N, thetal_imputed, ql))
    thetau_imputed = get_quantile(f_N, qu)
    loss_thetau_imputed = np.mean(pinball_loss(f_N, thetau_imputed, qu))
    print('thetal_imputed = {:.3f}, loss_thetal_imputed = {:.3f}'.format(
        thetal_imputed, loss_thetal_imputed
    ))
    print('thetau_imputed = {:.3f}, loss_thetau_imputed = {:.3f}'.format(
        thetau_imputed, loss_thetau_imputed
    ))

    # compute CLT-based UB on rectifier using n labeled data points
    # TODO: need to split budget for l and u?
    rectl_n = get_quantile_rectifier(y_n, f_n, thetal_imputed)
    mul = np.mean(rectl_n)
    sigmal = np.std(rectl_n, ddof=1)
    z = sc.stats.norm.ppf(1 - delta1, loc=0, scale=1)
    rectl_ub = mul + z * sigmal / np.sqrt(n)

    rectu_n = get_quantile_rectifier(y_n, f_n, thetau_imputed)
    muu = np.mean(rectu_n)
    sigmau = np.std(rectu_n, ddof=1)
    rectu_ub = muu + z * sigmau / np.sqrt(n)
    print('rectl_ub = {:.3f}, rectu_ub = {:.3f}'.format(rectl_ub, rectu_ub))

    # compute Hoeffding bound (root-N term)
    hoeffding_ub = B * np.sqrt(np.log(1 / delta2)) / np.sqrt(2 * N)

    # find quantile LB
    # TODO: technically find CIs for both and take extrema of both ends
    lb_fn = lambda theta: loss_thetal_imputed + rectl_ub + hoeffding_ub - np.mean(pinball_loss(f_N, theta, ql))
    if lb_fn(quantile_lb) >= 0:
        lb = quantile_lb
    else:
        assert(lb_fn(quantile_lb) < 0 and lb_fn(thetal_imputed) > 0)
        lb = sc.optimize.brentq(
            lb_fn,
            quantile_lb,
            thetal_imputed
        )

    # find quantile UB
    ub_fn = lambda theta: loss_thetau_imputed + rectu_ub + hoeffding_ub - np.mean(pinball_loss(f_N, theta, qu))
    if ub_fn(quantile_ub) >= 0:
        ub = quantile_ub
    else:
        assert(ub_fn(quantile_ub) < 0 and lb_fn(thetau_imputed) > 0)
        ub = sc.optimize.brentq(
            ub_fn,
            thetau_imputed,
            quantile_ub
        )

    # CI with no statistics
    # TODO: oracle can use full alpha budget not alpha - d1 - d2
    rectl = np.mean(get_quantile_rectifier(y_N, f_N, thetal_imputed))
    lb_fn = lambda theta: loss_thetal_imputed + rectl - np.mean(pinball_loss(f_N, theta, ql))
    if lb_fn(quantile_lb) >= 0:
        ideal_lb = quantile_lb
    else:
        ideal_lb = sc.optimize.brentq(
            lb_fn,
            quantile_lb,
            thetal_imputed
        )

    # find quantile UB
    rectu = np.mean(get_quantile_rectifier(y_N, f_N, thetau_imputed))
    ub_fn = lambda theta: loss_thetau_imputed + rectu - np.mean(pinball_loss(f_N, theta, qu))
    if ub_fn(quantile_ub) >= 0:
        ideal_ub = quantile_ub
    else:
        ideal_ub = sc.optimize.brentq(
            ub_fn,
            thetau_imputed,
            quantile_ub
        )

    return (lb, ub), (ideal_lb, ideal_ub), rectl_ub, rectu_ub, hoeffding_ub


def get_pai_ci(y_n, f_n, f_N, q, alpha, delta, B, quantile_lb, quantile_ub):
    n = y_n.size
    assert(f_n.size == n)
    N = f_N.size
    assert(delta < alpha)

    # compute imputed loss on imputed estimate
    theta_imputed = get_quantile(f_N, q)
    loss_N = pinball_loss(f_N, theta_imputed, q)
    loss_imputed = np.mean(loss_N)

    # compute CLT CI on deviations between losses on Y and f
    absdiff_n = np.abs(y_n - f_n)
    mu = np.mean(absdiff_n)
    sigma = np.std(absdiff_n, ddof=1)
    z = sc.stats.norm.ppf(1 - delta, loc=0, scale=1)
    rect_ub = np.max([q, 1 - q]) * (mu + z * sigma / np.sqrt(n))


    # compute Hoeffding bound (root-N term)
    hoeffding_ub = np.sqrt(2) * B * np.sqrt(np.log(1 / (alpha - delta))) / np.sqrt(N)

    # compute UB on loss
    loss_ub = loss_imputed + 2 * rect_ub + hoeffding_ub

    # search for interval around imputed estimate
    fn = lambda theta: loss_ub - np.mean(pinball_loss(f_N, theta, q))
    if fn(quantile_lb) > 0:
        lb = quantile_lb
    else:
        lb = sc.optimize.brentq(
            fn,
            quantile_lb,
            theta_imputed
        )
    if fn(quantile_ub) > 0:
        ub = quantile_ub
    else:
        ub = sc.optimize.brentq(
            fn,
            theta_imputed,
            quantile_ub
        )
    return (lb, ub), loss_imputed, rect_ub, hoeffding_ub, loss_ub

