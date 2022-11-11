import numpy as np
import math
import multiprocess

PTM_NAMES = [
    'p', 'p_reg', 'ub', 'ub_reg', 'sm', 'sm_reg', 'ac', 'ac_reg', 'm', 'm_reg', 'ga', 'gl', 'gl_reg',
]

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
    # If we get nans from 0 * inf, this should be 0
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
    The 1-dimensional logical confidence sequence for sampling without
    replacement. This is essentially the CS that would be
    known regardless of the underlying martingale being used.
    Specifically, if the cumulative sum at time t, S_t is equal to
    5 and N is 10, then the true mean cannot be any less than 0.5, assuming
    all observations are between 0 and 1.
    Parameters
    ----------
    x, array-like of reals between 0 and 1
        The observed bounded random variables.
    N, integer
        The size of the finite population
    Returns
    -------
    l, array-like
        Lower logical confidence sequence for the parameter
    u, array-like
        Upper logical confidence sequence for the parameter
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


# odds ratio estimation based on finite sample mean estimation
def get_odds_ratio_betting_ci(df, ptm_name, lab_idx, alpha, grid_spacing: float = 1e-3,
                              use_intersection: bool = True, verbose: bool = True,
                              parallelize: bool = False, n_cores: int = None):

    # Z: df[ptm_name]
    # Y: df['disordered']
    # f: df['pred_disordered']

    lab_df = df.iloc[lab_idx]  # labeled data
    N = len(df)                # total data
    n = len(lab_df)            # labeled data
    if verbose:
        print('N (total) = {}, n (labeled) = {}'.format(N, n))

    # ===== CIs on mu1 = P(Y = 1 | Z = 1) and mu0 = P(Y = 1 | Z = 0) =====

    # ----- true quantities
    z1_df = df.loc[df[ptm_name] == 1]
    z0_df = df.loc[df[ptm_name] == 0]
    mu1 = z1_df['disordered'].mean()
    mu0 = z0_df['disordered'].mean()

    # ----- model-assisted CI -----

    # CI on mu1
    lab_z1_df = lab_df.loc[lab_df[ptm_name] == 1]
    bias_lab_z1_n = lab_z1_df['disordered'].to_numpy() - lab_z1_df['pred_disordered'].to_numpy()
    # get WoR CI on average bias Y_i - f_i
    bias_z1_ci = get_betting_wor_ci(  # biases have values in {-1, 0, 1}, rescale to [0, 1]
        (bias_lab_z1_n + 1) / 2, N, alpha, grid_spacing, use_intersection=use_intersection,
        parallelize=parallelize, n_cores=n_cores
    )
    bias_z1_ci = bias_z1_ci * 2 - 1  # rescale to [-1, 1]

    # average prediction f_i on all data (labeled and unlabeled)
    f_mean_z1 = z1_df['pred_disordered'].mean()
    # add CI on bias to average prediction
    mu1_mai_ci = (np.maximum(f_mean_z1 + bias_z1_ci[0], 0), np.minimum(f_mean_z1 + bias_z1_ci[1], 1))

    # CI on mu0
    lab_z0_df = lab_df.loc[lab_df[ptm_name] == 0]
    bias_lab_z0_n = lab_z0_df['disordered'].to_numpy() - lab_z0_df['pred_disordered'].to_numpy()
    bias_z0_ci = get_betting_wor_ci(
        (bias_lab_z0_n + 1) / 2, N, alpha, grid_spacing, use_intersection=use_intersection,
        parallelize=parallelize, n_cores=n_cores
    )
    bias_z0_ci = bias_z0_ci * 2 - 1

    f_mean_z0 = z0_df['pred_disordered'].mean()
    mu0_mai_ci = (np.maximum(f_mean_z0 + bias_z0_ci[0], 0), np.minimum(f_mean_z0 + bias_z0_ci[1], 1))

    # ----- classical CI (using only labeled data) -----
    y_z1_n = lab_z1_df['disordered'].to_numpy()
    y_z0_n = lab_z0_df['disordered'].to_numpy()
    mu1_cla_ci = get_betting_wor_ci(
        y_z1_n, N, alpha, grid_spacing, use_intersection=use_intersection,
        parallelize=parallelize, n_cores=n_cores
    )
    mu0_cla_ci = get_betting_wor_ci(
        y_z0_n, N, alpha, grid_spacing, use_intersection=use_intersection,
        parallelize=parallelize, n_cores=n_cores
    )

    # ===== CIs on odds ratio =====

    o_mai_ci = get_odds_ratio_ci_from_mu_ci(mu1_mai_ci, mu0_mai_ci)
    o_cla_ci = get_odds_ratio_ci_from_mu_ci(mu1_cla_ci, mu0_cla_ci)

    # ----- true quantity -----
    n_z1_y1 = len(df.loc[(df[ptm_name] == 1) & (df['disordered'] == 1)])
    n_z0_y1 = len(df.loc[(df[ptm_name] == 0) & (df['disordered'] == 1)])
    n_z1_y0 = len(df.loc[(df[ptm_name] == 1) & (df['ordered'] == 1)])
    n_z0_y0 = len(df.loc[(df[ptm_name] == 0) & (df['ordered'] == 1)])
    o = (n_z1_y1 / n_z0_y1) / (n_z1_y0 / n_z0_y0)

    if verbose:
        print('True mu1: {:.3f}.'.format(mu1))
        print('  Model-assisted CI: [{:.2f}, {:.2f}] (length {:.2f})\n  Classical CI: [{:.2f}, {:.2f}] (length {:.2f}). '.format(
            mu1_mai_ci[0], mu1_mai_ci[1], mu1_mai_ci[1] - mu1_mai_ci[0],
            mu1_cla_ci[0], mu1_cla_ci[1], mu1_cla_ci[1] - mu1_cla_ci[0]))
        print('True mu0: {:.3f}.'.format(mu0))
        print('  Model-assisted CI: [{:.2f}, {:.2f}] (length {:.2f}).\n  Classical CI: [{:.2f}, {:.2f}] (length {:.2f}). '.format(
            mu0_mai_ci[0], mu0_mai_ci[1], mu0_mai_ci[1] - mu0_mai_ci[0],
            mu0_cla_ci[0], mu0_cla_ci[1], mu0_cla_ci[1] - mu0_cla_ci[0]))
        print('True odds ratio: {:.3f}.'.format(o))
        print('  Model-assisted CI: [{:.2f}, {:.2f}] (length {:.2f}).\n  Classical CI: [{:.2f}, {:.2f}] (length {:.2f}). '.format(
            o_mai_ci[0], o_mai_ci[1], o_mai_ci[1] - o_mai_ci[0],
            o_cla_ci[0], o_cla_ci[1], o_cla_ci[1] - o_cla_ci[0]))

    return o, o_mai_ci, o_cla_ci, mu1, mu1_mai_ci, mu1_cla_ci, mu0, mu0_mai_ci, mu0_cla_ci



