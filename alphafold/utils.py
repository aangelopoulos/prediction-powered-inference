import sys
sys.path.insert(1, '../')
from concentration import wsr_iid

import numpy as np
import scipy.stats as stats
import math
import multiprocess

def get_confusion_matrix(df, pred_col_name, true_column_name):
    y1_df = df.loc[df[true_column_name] == 1]
    y0_df = df.loc[df[true_column_name] == 0]
    confmat_2x2 = np.array([
        [len(y1_df.loc[y1_df[pred_col_name] >= 0.5]), len(y1_df.loc[y1_df[pred_col_name] < 0.5])],
        [len(y0_df.loc[y0_df[pred_col_name] >= 0.5]), len(y0_df.loc[y0_df[pred_col_name] < 0.5])]
    ])
    confmat_2x2 = confmat_2x2 / np.sum(confmat_2x2, axis=1, keepdims=True)
    return confmat_2x2

def B_wor(N, x_n, m, alpha, c: float = 3 / 4, theta: float = 1 / 2, convex_comb: bool = False):

    assert(np.all(x_n >= 0))
    assert(np.all(x_n <= 1))

    n = x_n.size
    muhat_n = (0.5 + np.cumsum(x_n)) / (1 + np.arange(1, n + 1))

    sigma2hat_n = (0.25 + np.cumsum(np.power(x_n - muhat_n, 2))) / (1 + np.arange(1, n + 1))
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[: -1])
    assert(np.all(sigma2hat_tminus1_n > 0))

    lambdadot_n = np.sqrt(2 * np.log(2 / alpha) / (n * sigma2hat_tminus1_n))
    lambdadot_n[np.isnan(lambdadot_n)] = 0

    sumxtminus1_n = np.append(0, np.cumsum(x_n[: -1]))
    mwor_n = (N * m - sumxtminus1_n) / (N - np.arange(0, n))

    with np.errstate(divide="ignore"):
        lambdaplus_n = np.minimum(lambdadot_n, c / mwor_n)
        lambdaplus_n = np.maximum(lambdaplus_n, -c / (1 - mwor_n))
        lambdaminus_n = np.minimum(lambdadot_n, c / (1 - mwor_n))
        lambdaminus_n = np.maximum(lambdaminus_n, -c / mwor_n)

    multiplicandpos_n = 1 + lambdaplus_n * (x_n - mwor_n)
    multiplicandpos_n[np.logical_and(lambdaplus_n == math.inf, x_n - mwor_n == 0)] = 1
    with np.errstate(invalid="ignore"):
        Kworpos_n = np.exp(np.cumsum(np.log(multiplicandpos_n)))
    # if have nans from 0 * inf, this should be 0
    Kworpos_n[np.isnan(Kworpos_n)] = 0

    multiplicandneg_n = 1 - lambdaminus_n * (x_n - mwor_n)
    multiplicandneg_n[np.logical_and(lambdaminus_n == math.inf, x_n - mwor_n == 0)] = 1
    with np.errstate(invalid="ignore"):
        Kworneg_n = np.exp(np.cumsum(np.log(multiplicandneg_n)))
    Kworneg_n[np.isnan(Kworneg_n)] = 0

    if convex_comb:
        Kwor_n = theta * Kworpos_n + (1 - theta) * Kworneg_n
    else:
        Kwor_n = np.maximum(theta * Kworpos_n, (1 - theta) * Kworneg_n)
    Kwor_n[np.logical_or(mwor_n < 0, mwor_n > 1)] = math.inf

    assert not any(np.isnan(Kwor_n))
    assert all(Kwor_n >= 0)

    return Kwor_n

def get_odds_ratio_ci_from_mu_ci(mu1_ci, mu0_ci):
    # CI on mu_1 / (1 - mu_1)
    with np.errstate(divide="ignore"):
        ratio_mu1 = (mu1_ci[0] / (1 - mu1_ci[0]), mu1_ci[1] / (1 - mu1_ci[1]))
    assert(ratio_mu1[0] <= ratio_mu1[1])

    # CI on mu_0 / (1 - mu_0)
    with np.errstate(divide="ignore"):
        ratio_mu0 = (mu0_ci[0] / (1 - mu0_ci[0]), mu0_ci[1] / (1 - mu0_ci[1]))
    assert(ratio_mu0[0] <= ratio_mu0[1])

    # combine into CI on odds ratio
    o_ci = (ratio_mu1[0] / ratio_mu0[1], ratio_mu1[1] / ratio_mu0[0])
    return o_ci

def get_logical_ci(x_n, N):
    """
    Confidence interval that would be known for sampling w/o replacement,
    regardless of concentration strategy used.

    For example, if the sum of our labeled data is 5 and N is 10, then the true mean
    cannot be any less than 0.5 (assuming all observations are between 0 and 1).
    """
    t = np.arange(1, x_n.size + 1)
    S_n = np.cumsum(x_n)
    l_n = S_n / N
    u_n = 1 - (t - S_n) / N
    return l_n[-1], u_n[-1]


def get_betting_wor_ci(x_n, N, alpha, grid_spacing, use_intersection: bool = True,
                       parallelize: bool = False, n_cores: int = None):
    candidates = np.arange(0, 1 + grid_spacing, step=grid_spacing)
    threshold = 1 / alpha
    if parallelize:
        if n_cores is None:
            n_cores = multiprocess.cpu_count()
        with multiprocess.Pool(n_cores) as p:
            Kn_list = p.map(lambda m: B_wor(N, x_n, m, alpha), candidates)
            K_mxn = np.vstack(Kn_list)
            if use_intersection:
                ci_indicators = np.max(K_mxn, axis=1) <= threshold
            else:
                ci_indicators = K_mxn[:, -1] <= threshold
            ci_values = candidates[np.where(ci_indicators)[0]]

    else:
        ci_values = []
        for m in candidates:
            K_n = B_wor(N, x_n, m, alpha)
            K = np.max(K_n) if use_intersection else K_n[-1]
            if K < threshold:
                ci_values.append(m)
            else:
                if len(ci_values):
                    break  # since interval, stop testing candidates once one is rejected

    if len(ci_values) == 0:
        l, u = 0, 1
    else:
        l = np.maximum(np.min(ci_values) - grid_spacing / 2, 0)
        u = np.minimum(np.max(ci_values) + grid_spacing / 2, 1)
    l_logical, u_logical = get_logical_ci(x_n, N)
    return np.array([np.maximum(l, l_logical), np.minimum(u, u_logical)])


# TODO for IID: fix WSR version to not double dip labeled and split error budget,
# implement CLT version where do not split error budget
def get_odds_ratio_intervals(df, ptm_name, n, alpha, delta_frac: float = 0.1, use_clt: bool = False,
                             grid_spacing: float = 1e-3, use_iid_approximation: bool = True,
                             use_intersection: bool = True, verbose: bool = True,
                             parallelize: bool = False, n_cores: int = None, n_min: int = 20):

    # Z: df[ptm_name]
    # Y: df['disordered'], binary
    # f: df['pred_disordered'], in [0, 1]

    # ===== partition n labels between examples with Z = 1 and Z = 0 =====

    # extract modified and unmodified instances from gold-standard Z
    z1_df = df.loc[df[ptm_name] == 1]
    z0_df = df.loc[df[ptm_name] == 0]
    N_z1 = len(z1_df)
    N_z0 = len(z0_df)
    assert(N_z1 + N_z0 == len(df))

    # split budget of n labels between the two groups
    n_z1 = int(N_z1 * n / len(df))
    n_z0 = n - n_z1
    n_min = np.min([n_min, N_z1, N_z0])
    if n_z1 < n_min:
        n_z1 = n_min
        n_z0 = n - n_min
    elif n_z0 < n_min:
        n_z0 = n_min
        n_z1 = n - n_min
    assert(n_z0 >= n_min)
    assert(n_z1 >= n_min)
    assert(n_z1 + n_z0 == n)
    if verbose:
        print('For Z1: N = {}, n = {}, fraction = {:.3f}'.format(N_z1, n_z1, n_z1 / N_z1))
        print('For Z0: N = {}, n = {}, fraction = {:.3f}'.format(N_z0, n_z0, n_z0 / N_z0))

    z1_lab_idx = np.random.choice(N_z1, n_z1, replace=False)
    lab_z1_df = z1_df.iloc[z1_lab_idx]
    z0_lab_idx = np.random.choice(N_z0, n_z0, replace=False)
    lab_z0_df = z0_df.iloc[z0_lab_idx]

    # only used for IID approximation (finite population version will just take all of z1_df)
    unlab_z1_df = z1_df.iloc[np.delete(np.arange(N_z1), z1_lab_idx)]
    unlab_z0_df = z0_df.iloc[np.delete(np.arange(N_z0), z0_lab_idx)]

    # ===== CIs on mu1 = P(Y = 1 | Z = 1) and mu0 = P(Y = 1 | Z = 0) =====

    # ----- prediction-powered CI -----

    rect_lab_z1_n = lab_z1_df['pred_disordered'].to_numpy() - lab_z1_df['disordered'].to_numpy()
    rect_lab_z0_n = lab_z0_df['pred_disordered'].to_numpy() - lab_z0_df['disordered'].to_numpy()

    if use_iid_approximation:
        if use_clt:
            f_unlab_z1_N = unlab_z1_df['pred_disordered'].to_numpy()
            f_unlab_z0_N = unlab_z0_df['pred_disordered'].to_numpy()

            # empirical variance of imputed estimate
            sigma2f_z1 = np.mean(np.square(f_unlab_z1_N - f_unlab_z1_N.mean()))
            sigma2f_z0 = np.mean(np.square(f_unlab_z0_N - f_unlab_z0_N.mean()))

            # empirical variance of empirical rectifier
            sigma2rect_z1 = np.mean(np.square(rect_lab_z1_n - rect_lab_z1_n.mean()))
            sigma2rect_z0 = np.mean(np.square(rect_lab_z0_n - rect_lab_z0_n.mean()))

            # CLT intervals
            w_z1 = stats.norm.ppf(1 - alpha / 4) * np.sqrt(sigma2f_z1 / f_unlab_z1_N.size + sigma2rect_z1 / rect_lab_z1_n.size)
            w_z0 = stats.norm.ppf(1 - alpha / 4) * np.sqrt(sigma2f_z0 / f_unlab_z0_N.size + sigma2rect_z0 / rect_lab_z0_n.size)
            mu1_pp = f_unlab_z1_N.mean() - rect_lab_z1_n.mean()
            mu0_pp = f_unlab_z0_N.mean() - rect_lab_z0_n.mean()
            mu1_mai_ci = (np.maximum(mu1_pp - w_z1, 0), np.minimum(mu1_pp + w_z1, 1))
            mu0_mai_ci = (np.maximum(mu0_pp - w_z0, 0), np.minimum(mu0_pp + w_z0, 1))

        else:  # use finite-sample WSR concentration
            grid = np.arange(grid_spacing, 1, step=grid_spacing)
            delta = delta_frac * alpha
            if verbose:
                print('Using delta = {:.3f} for estimating big-N terms.'.format(delta))

            # biases have values in [-1, 1], rescale to [0, 1]
            rect_z1_ci = wsr_iid((rect_lab_z1_n + 1) / 2, (alpha - delta) / 2, grid, parallelize=False)
            rect_z0_ci = wsr_iid((rect_lab_z0_n + 1) / 2, (alpha - delta) / 2, grid, parallelize=False)
            rect_z1_ci = rect_z1_ci * 2 - 1  # rescale to [-1, 1]
            rect_z0_ci = rect_z0_ci * 2 - 1

            f_z1_ci = wsr_iid(unlab_z1_df['pred_disordered'].to_numpy(), delta / 2, grid, parallelize=False)
            f_z0_ci = wsr_iid(unlab_z0_df['pred_disordered'].to_numpy(), delta / 2, grid, parallelize=False)

            mu1_mai_ci = (np.maximum(f_z1_ci[0] - rect_z1_ci[1], 0), np.minimum(f_z1_ci[1] - rect_z1_ci[0], 1))
            mu0_mai_ci = (np.maximum(f_z0_ci[0] - rect_z0_ci[1], 0), np.minimum(f_z0_ci[1] - rect_z0_ci[0], 1))

    else: # finite population version
        rect_z1_ci = get_betting_wor_ci(
            (rect_lab_z1_n + 1) / 2, N_z1, alpha / 2, grid_spacing, use_intersection=use_intersection,
            parallelize=parallelize, n_cores=n_cores
        )
        rect_z0_ci = get_betting_wor_ci(
            (rect_lab_z0_n + 1) / 2, N_z0, alpha / 2, grid_spacing, use_intersection=use_intersection,
            parallelize=parallelize, n_cores=n_cores
        )
        rect_z1_ci = rect_z1_ci * 2 - 1  # rescale to [-1, 1]
        rect_z0_ci = rect_z0_ci * 2 - 1

        f_z1 = z1_df['pred_disordered'].mean()
        f_z0 = z0_df['pred_disordered'].mean()

        mu1_mai_ci = (np.maximum(f_z1 - rect_z1_ci[1], 0), np.minimum(f_z1 - rect_z1_ci[0], 1))
        mu0_mai_ci = (np.maximum(f_z0 - rect_z0_ci[1], 0), np.minimum(f_z0 - rect_z0_ci[0], 1))

    # ----- classical CI (using only labeled data) -----

    y_z1_n = lab_z1_df['disordered'].to_numpy()
    y_z0_n = lab_z0_df['disordered'].to_numpy()
    if use_iid_approximation:
        if use_clt:
            sigma2y_z1 = np.mean(np.square(y_z1_n - y_z1_n.mean()))
            sigma2y_z0 = np.mean(np.square(y_z0_n - y_z0_n.mean()))
            w_z1 = stats.norm.ppf(1 - alpha / 4) *  np.sqrt(sigma2y_z1 / y_z1_n.size)
            w_z0 = stats.norm.ppf(1 - alpha / 4) *  np.sqrt(sigma2y_z0 / y_z0_n.size)
            mu1_cla_ci = (np.maximum(y_z1_n.mean() - w_z1, 0), np.minimum(y_z1_n.mean() + w_z1, 1))
            mu0_cla_ci = (np.maximum(y_z0_n.mean() - w_z0, 0), np.minimum(y_z0_n.mean() + w_z0, 1))
        else: # use finite-sample WSR concentration
            mu1_cla_ci = wsr_iid(y_z1_n, alpha / 2, grid, parallelize=False)
            mu0_cla_ci = wsr_iid(y_z0_n, alpha / 2, grid, parallelize=False)
    else:
        mu1_cla_ci = get_betting_wor_ci(
            y_z1_n, N_z1, alpha / 2, grid_spacing, use_intersection=use_intersection, parallelize=False
        )
        mu0_cla_ci = get_betting_wor_ci(
            y_z0_n, N_z0, alpha / 2, grid_spacing, use_intersection=use_intersection, parallelize=False
        )

    # ===== CIs on odds ratio =====

    o_mai_ci = get_odds_ratio_ci_from_mu_ci(mu1_mai_ci, mu0_mai_ci)
    o_cla_ci = get_odds_ratio_ci_from_mu_ci(mu1_cla_ci, mu0_cla_ci)

    # ----- true (finite population) quantities -----
    mu1 = z1_df['disordered'].mean()
    mu0 = z0_df['disordered'].mean()
    n_z1_y1 = len(df.loc[(df[ptm_name] == 1) & (df['disordered'] == 1)])
    n_z0_y1 = len(df.loc[(df[ptm_name] == 0) & (df['disordered'] == 1)])
    n_z1_y0 = len(df.loc[(df[ptm_name] == 1) & (df['ordered'] == 1)])
    n_z0_y0 = len(df.loc[(df[ptm_name] == 0) & (df['ordered'] == 1)])
    o = (n_z1_y1 / n_z0_y1) / (n_z1_y0 / n_z0_y0)


    if verbose:
        print('True mu1: {:.3f}.'.format(mu1))
        print('  Prediction-powered: [{:.2f}, {:.2f}] (length {:.2f})\n  Classical: [{:.2f}, {:.2f}] (length {:.2f}). '.format(
            mu1_mai_ci[0], mu1_mai_ci[1], mu1_mai_ci[1] - mu1_mai_ci[0],
            mu1_cla_ci[0], mu1_cla_ci[1], mu1_cla_ci[1] - mu1_cla_ci[0]))
        print('True mu0: {:.3f}.'.format(mu0))
        print('  Prediction-powered: [{:.2f}, {:.2f}] (length {:.2f}).\n  Classical: [{:.2f}, {:.2f}] (length {:.2f}). '.format(
            mu0_mai_ci[0], mu0_mai_ci[1], mu0_mai_ci[1] - mu0_mai_ci[0],
            mu0_cla_ci[0], mu0_cla_ci[1], mu0_cla_ci[1] - mu0_cla_ci[0]))
        print('True odds ratio: {:.3f}.'.format(o))
        print('  Prediction-powered: [{:.2f}, {:.2f}] (length {:.2f}).\n  Classical: [{:.2f}, {:.2f}] (length {:.2f}). '.format(
            o_mai_ci[0], o_mai_ci[1], o_mai_ci[1] - o_mai_ci[0],
            o_cla_ci[0], o_cla_ci[1], o_cla_ci[1] - o_cla_ci[0]))

    return mu1, mu1_mai_ci, mu1_cla_ci, mu0, mu0_mai_ci, mu0_cla_ci, o, o_mai_ci, o_cla_ci,



