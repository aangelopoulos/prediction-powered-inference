import sys
sys.path.insert(1, '../')
import numpy as np
import scipy as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor

def get_quantile(y_n, q, y_lb: float = -np.inf, y_ub: float = np.inf):
    """
    Compared to numpy.quantile, no interpolation.
    """
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
    return lb, ub

def get_quantile_intervals(y_all, f_all, q, n, alpha, theta_grid_spacing: float = 0.01, n_train: int = 5):
    y_n, y_N, f_n, f_N = train_test_split(y_all, f_all, train_size=n)
    if n_train:
        # fit median regressor with handful of labeled data points
        predictor = QuantileRegressor(quantile=0.5, alpha=1e-6)
        train_idx = np.random.choice(n, n_train, replace=False)
        lab_idx = np.delete(np.arange(n), train_idx)  # n - n_train remaining labeled data points
        assert(lab_idx.size == n - n_train)
        predictor.fit(f_n[train_idx, None], y_n[train_idx])
        f_n = predictor.predict(f_n[lab_idx, None])
        f_N = predictor.predict(f_N[:, None])
        y_n = y_n[lab_idx]
        assert(f_n.shape == y_n.shape)
        assert(f_N.shape == y_N.shape)

    # ===== construct prediction-powered CIs on quantile =====

    # ----- prediction-powered interval -----
    theta_min = np.min([np.min(y_n), np.min(f_N), np.min(f_n)])
    theta_max = np.max([np.max(y_n), np.max(f_N), np.max(f_n)])
    theta_grid = np.arange(theta_min, theta_max, theta_grid_spacing)
    F_t = np.empty([theta_grid.size])
    rect_t = np.empty([theta_grid.size])
    w_t = np.empty([theta_grid.size])

    # construct confidence intervals on rectifier per candidate value
    z = sc.stats.norm.ppf(1 - alpha / 2)
    for t, theta in enumerate(theta_grid):
        gdiff_n = (f_n <= theta).astype(float) - (y_n <= theta).astype(float)
        rect_t[t] = gdiff_n.mean()
        fcdf_N = (f_N <= theta).astype(float)
        F_t[t] = fcdf_N.mean()
        sigma2rect = np.mean(np.square(gdiff_n - rect_t[t]))
        sigma2f = np.mean(np.square(fcdf_N - F_t[t]))
        w_t[t] = z * np.sqrt((sigma2rect / f_n.size) + (sigma2f / f_N.size))

    # include all candidate values for which rectifier value of 0 is in rectifier confidence interval
    include_idx = np.where(np.abs(F_t - rect_t - q) <= w_t)[0]
    if include_idx.size == 0:  # edge case if no candidates qualify
        include_idx = np.argmin(np.abs(F_t - rect_t - q))
        if include_idx == 0:
            ci_pp = np.array([theta_grid[0], theta_grid[1]])
        else:
            ci_pp = np.array([theta_grid[include_idx - 1], theta_grid[include_idx]])
    else:
        ci_pp = theta_grid[include_idx]
        ci_pp = np.array([ci_pp.min(), ci_pp.max()])

    # ----- classical CI on quantile -----
    ci_cl = get_classical_ci(y_n, q, alpha)

    # ===== true (finite population) quantity =====
    true_quantile = get_quantile(y_all, q)

    return true_quantile, ci_pp, ci_cl
