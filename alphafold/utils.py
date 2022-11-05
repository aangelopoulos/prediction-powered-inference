import numpy as np
import math
from scipy.stats import binomtest

def get_confusion_matrix(df, pred_col_name, true_column_name):
    y1_df = df.loc[df[true_column_name] == 1]
    y0_df = df.loc[df[true_column_name] == 0]
    confmat_2x2 = np.array([
        [len(y1_df.loc[y1_df[pred_col_name] == 1]), len(y1_df.loc[y1_df[pred_col_name] == 0])],
        [len(y0_df.loc[y0_df[pred_col_name] == 1]), len(y0_df.loc[y0_df[pred_col_name] == 0])]
    ])
    confmat_2x2 = confmat_2x2 / np.sum(confmat_2x2, axis=1, keepdims=True)
    return confmat_2x2

def B_wor(N, x_n, m, alpha: float = 0.05, c: float = 3 / 4, theta: float = 1 / 2, convex_comb: bool = False):

    n = x_n.size
    muhat_n = (0.5 + np.cumsum(x_n)) / (1 + np.arange(1, n + 1))

    sigma2hat_n = (0.25 + np.cumsum(np.square(x_n - muhat_n))) / (1 + np.arange(1, n + 1))
    sigma2hat_tminus1_n = np.hstack([0.25, sigma2hat_n[: -1]])

    lambdadot_n = np.sqrt(2 * np.log(2 / alpha) / (n * sigma2hat_tminus1_n))

    sumxtminus1_n = np.hstack([0, np.cumsum(x_n[: -1])])

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

def get_betting_wor_ci(x_n, N, alpha, grid_spacing):
    candidates = np.arange(0, 1 + grid_spacing, grid_spacing)
    ci = []
    for m in candidates:
        K_n = B_wor(N, x_n, m, alpha, convex_comb=False)
        if np.max(K_n) < 1 / alpha:
            ci.append(m)
    if len(ci) == 0:
        candidates = np.arange(grid_spacing / 2, 1, grid_spacing)
        for m in candidates:
            K_n = B_wor(N, x_n, m, alpha, convex_comb=False)
            if np.max(K_n) < 1 / alpha:
                ci.append(m)
    return np.min(ci) - grid_spacing / 2, np.max(ci) + grid_spacing / 2

# odds ratio estimation based on finite sample mean estimation
def get_odds_ratio_betting_ci(df, ptm_name, lab_idx, alpha, grid_spacing: float = 1e-2, verbose: bool = True):

    # Z: df[ptm_name]
    # Y: df['disordered']
    # f: df[pred_disordered]

    lab_df = df.iloc[lab_idx]  # labeled data
    N = len(df)                # total data
    n = len(lab_df)            # labeled data
    if verbose:
        print('N (total) = {}, n (labeled) = {}'.format(N, n))

    # ===== conditioned on Z = 1 =====

    # get sampling w.o. replacement CI on average difference Y_i - f_i
    lab_z1_df = lab_df.loc[lab_df[ptm_name] == 1]
    z1_diffs_n = lab_z1_df['disordered'] - lab_z1_df['pred_disordered']
    z1_scaleddiffs_n = (z1_diffs_n + 1) / 2  # values in {-1, 0, 1}, rescale to [0, 1]
    l, u = get_betting_wor_ci(z1_scaleddiffs_n, N, alpha, grid_spacing)
    l_rescaled = l * 2 - 1  # rescale to [-1, 1]
    u_rescaled = u * 2 - 1

    z1_df = df.loc[df[ptm_name] == 1]  # average prediction f_i
    f_z1_mean = z1_df['pred_disordered'].mean()
    mu1hat_ci = (np.maximum(f_z1_mean + l_rescaled, 0), np.minimum(f_z1_mean + u_rescaled, 1))

    # ===== conditioned on Z = 0 =====

    # get sampling w.o. replacement CI on average difference Y_i - f_i
    lab_z0_df = lab_df.loc[lab_df[ptm_name] == 0]
    z0_diffs_n = lab_z0_df['disordered'] - lab_z0_df['pred_disordered']
    z0_scaleddiffs_n = (z0_diffs_n + 1) / 2
    l, u = get_betting_wor_ci(z0_scaleddiffs_n, N, alpha, grid_spacing)
    l_rescaled = l * 2 - 1
    u_rescaled = u * 2 - 1

    z0_df = df.loc[df[ptm_name] == 0]
    f_z0_mean = z0_df['pred_disordered'].mean()
    mu0hat_ci = (np.maximum(f_z0_mean + l_rescaled, 0), np.minimum(f_z0_mean + u_rescaled, 1))

    # ===== confidence interval on odds ratio =====

    # CI on mu_1 / (1 - mu_1)
    with np.errstate(divide="ignore"):
        ratio1 = (mu1hat_ci[0] / (1 - mu1hat_ci[0]), mu1hat_ci[1] / (1 - mu1hat_ci[1]))
    assert(ratio1[0] <= ratio1[1])

    # CI on (1 - mu_0) / mu_0
    with np.errstate(divide="ignore"):
        ratio2 = ((1 - mu0hat_ci[1]) / mu0hat_ci[1], (1 - mu0hat_ci[0]) / mu0hat_ci[0])
    assert(ratio2[0] <= ratio2[1])

    # combine into CI on odds ratio
    o_ci = (ratio1[0] * ratio2[0], ratio1[1] * ratio2[1])

    # ===== compute true odds ratio =====
    # TODO: technically this should be hypergeometric test since sampling without replacement
    n_z1_y1 = len(df.loc[(df[ptm_name] == 1) & (df['disordered'] == 1)])
    n_z0_y1 = len(df.loc[(df[ptm_name] == 0) & (df['disordered'] == 1)])
    n_z1_y0 = len(df.loc[(df[ptm_name] == 1) & (df['ordered'] == 1)])
    n_z0_y0 = len(df.loc[(df[ptm_name] == 0) & (df['ordered'] == 1)])
    true_o = (n_z1_y1 / n_z0_y1) / (n_z1_y0 / n_z0_y0)

    # ===== classical CI based solely on labeled data =====
    sum_y_z1 = int(lab_z1_df['disordered'].sum())
    result = binomtest(k=sum_y_z1, n=len(lab_z1_df), p=0.5)  # dummy value for p
    classical_mu1_ci = np.array([result.proportion_ci().low, result.proportion_ci().high])

    sum_y_z0 = int(lab_z0_df['disordered'].sum())
    result = binomtest(k=sum_y_z0, n=len(lab_z0_df), p=0.5)  # dummy value for p
    classical_mu0_ci = np.array([result.proportion_ci().low, result.proportion_ci().high])

    # CI on mu_1 / (1 - mu_1)
    with np.errstate(divide="ignore"):
        classical_ratio1 = (classical_mu1_ci[0] / (1 - classical_mu1_ci[0]),
                        classical_mu1_ci[1] / (1 - classical_mu1_ci[1]))
    assert(classical_ratio1[0] <= classical_ratio1[1])

    # CI on (1 - mu_0) / mu_0
    with np.errstate(divide="ignore"):
        classical_ratio2 = ((1 - classical_mu0_ci[1]) / classical_mu0_ci[1],
                        (1 - classical_mu0_ci[0]) / classical_mu0_ci[0])
    assert(classical_ratio2[0] <= classical_ratio2[1])

    # combine into CI on odds ratio
    classical_o_ci = (classical_ratio1[0] * classical_ratio2[0], classical_ratio1[1] * classical_ratio2[1])

    if verbose:
        mu1_true = z1_df['disordered'].mean()
        mu0_true = z0_df['disordered'].mean()
        print('CI for mu1: [{:.3f}, {:.3f}]. Classical CI for mu1: [{:.3f}, {:.3f}]. True mu1: {:.3f}'.format(
            mu1hat_ci[0], mu1hat_ci[1], classical_mu1_ci[0], classical_mu1_ci[1], mu1_true))
        print('CI for mu0: [{:.3f}, {:.3f}]. Classical CI for mu0: [{:.3f}, {:.3f}]. True mu0: {:.3f}'.format(
            mu0hat_ci[0], mu0hat_ci[1], classical_mu0_ci[0], classical_mu0_ci[1], mu0_true))
        print('CI for mu1 / (1 - mu1): [{:.3f}, {:.3f}]]. Classical CI: [{:.3f}, {:.3f}]]'.format(
            ratio1[0], ratio1[1], classical_ratio1[0], classical_ratio1[1]))
        print('CI for (1 - mu0) / mu0: [{:.3f}, {:.3f}]]. Classical CI: [{:.3f}, {:.3f}]]'.format(
            ratio2[0], ratio2[1], classical_ratio2[0], classical_ratio2[1]))

    return o_ci, classical_o_ci, true_o



