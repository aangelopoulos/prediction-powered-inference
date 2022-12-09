import os, sys
sys.path.insert(1, '../')
import numpy as np
import scipy as sc
from concentration import wsr_iid
from sklearn.model_selection import train_test_split

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
    # ql = float(l) / n
    # qu = float(u + 1) / n
    # assert(ql >= 0 and ql <= 1)
    # assert(qu >= 0 and qu <= 1)
    return lb, ub

def trial(y_all, f_all, q, n, alpha, delta, theta_grid, parallelize, eps: float = 1e-3, c: float = 0.75, use_hoeffding: bool = True):
    # TODO: make this mimic IID better?
    y_n, y_N, f_n, f_N = train_test_split(y_all, f_all, train_size=n)
    ci_cl = get_classical_ci(y_n, q, alpha)

    Cthetas = [
        wsr_iid( ((y_n <= theta).astype(float) - (f_n <= theta).astype(float) + 1) / 2,  # rescale into [0, 1]
                delta, np.arange(eps, 1, eps), parallelize=parallelize, c=c) * 2 - 1 for theta in theta_grid]
    q_diff = [q - (f_N <= theta).mean() for theta in theta_grid]
    if use_hoeffding:
        hoeff =  np.sqrt(np.log(2 / (alpha - delta)) / (2 * y_N.size))
    else:
        hoeff = 0
    indexes = np.array(
        [(Cthetas[j][0] - hoeff <= q_diff[j]) & (Cthetas[j][1] + hoeff >= q_diff[j]) for j in range(theta_grid.size)]
    )
    if np.sum(indexes) < 2:
        idx = np.argmin(
            [np.minimum(q_diff[j] - Cthetas[j][1], Cthetas[j][0] - q_diff[j]) for j in range(theta_grid.size)])
        if idx == 0:
            ci_pp = np.array([theta_grid[0], theta_grid[1]])
        else:
            ci_pp = np.array([theta_grid[idx - 1], theta_grid[idx]])
    else:
        ci_pp = theta_grid[np.where(indexes)[0]]
        ci_pp = np.array([ci_pp.min(), ci_pp.max()])

    width_cl = ci_cl[1] - ci_cl[0]
    width_pp = ci_pp[1] - ci_pp[0]

    true_quantile = get_quantile(y_all, q)
    cov_cl = (ci_cl[0] <= true_quantile) & (ci_cl[1] >= true_quantile)
    cov_pp = (ci_pp[0] <= true_quantile) & (ci_pp[1] >= true_quantile)

    return width_cl, width_pp, cov_cl, cov_pp
