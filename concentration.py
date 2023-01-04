import numpy as np
import itertools
from scipy.stats import binom, norm
from scipy.optimize import brentq
from joblib import delayed, Parallel
import pdb

"""
    IID
"""
def binomial_iid(N,alpha,muhat):
    def invert_upper_tail(mu): return binom.cdf(N*muhat, N, mu) - (alpha/2)
    def invert_lower_tail(mu): return binom.cdf(N*muhat, N, mu) - (1-alpha/2)
    u = brentq(invert_upper_tail,0,1)
    l = brentq(invert_lower_tail,0,1)
    return np.array([l,u])

def bentkus_iid(N, alpha, muhat):
    return binomial_iid(N, alpha/np.e, muhat)

def clt_iid(x, alpha):
    n = x.shape[0]
    sigmahat = x.std()
    w = norm.ppf(1-alpha/2) * sigmahat / np.sqrt(n)
    return np.array([ x.mean() - w, x.mean() + w ])

def wsr_iid_ana(x,alpha,grid,num_cpus=10,step=1): # x is a [0,1] bounded sequence
    n = x.shape[0]
    muhats = (1/2 + np.cumsum(x))/(np.arange(n)+1)
    sigmahat2s = (1/4 + np.cumsum((x-muhats)**2))/(np.arange(n)+1)
    lambdas = np.concatenate([np.array([1,]), np.sqrt(2*np.log(2/alpha)/(n*sigmahat2s))[:-1]]) # can't use last entry
    def M(m,i): return 1/2*np.maximum(
        np.prod(1+np.minimum(lambdas[:i], 0.5/m)*(x[:i]-m)),
        np.prod(1-np.minimum(lambdas[:i], 0.5/(1-m))*(x[:i]-m))
    )
    M = np.vectorize(M)
    M_list = Parallel(n_jobs=num_cpus)(delayed(M)(grid,i) for i in range(1,n+step,step))
    ci_full = grid[np.where(np.prod(np.stack(M_list, axis=1) < 1/alpha , axis=1))[0]]
    return np.array([ci_full.min(), ci_full.max()]) # only output the interval

def wsr_iid(x_n, alpha, grid, num_cpus=10, parallelize: bool = False, intersection: bool = True,
            theta: float = 0.5, c: float = 0.75):
    n = x_n.shape[0]
    t_n = np.arange(1, n + 1)
    muhat_n = (0.5 + np.cumsum(x_n)) / (1 + t_n)
    sigma2hat_n = (0.25 + np.cumsum(np.power(x_n - muhat_n, 2))) / (1 + t_n)
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[: -1])
    assert(np.all(sigma2hat_tminus1_n > 0))
    lambda_n = np.sqrt(2 * np.log(2 / alpha) / (n * sigma2hat_tminus1_n))

    def M(m):
        lambdaplus_n = np.minimum(lambda_n, c / m)
        lambdaminus_n = np.minimum(lambda_n, c / (1 - m))
        return np.maximum(
            theta * np.exp(np.cumsum(np.log(1 + lambdaplus_n * (x_n - m)))),
            (1 - theta) * np.exp(np.cumsum(np.log(1 - lambdaminus_n * (x_n - m))))
        )

    if parallelize:  # sometimes much slower
        M = np.vectorize(M)
        M_list = Parallel(n_jobs=num_cpus)(delayed(M)(m) for m in grid)
        indicators_gxn = np.array(M_list) < 1 / alpha
    else:
        indicators_gxn = np.zeros([grid.size, n])
        found_lb = False
        for m_idx, m in enumerate(grid):
            m_n = M(m)
            indicators_gxn[m_idx] = m_n < 1 / alpha
            if not found_lb and np.prod(indicators_gxn[m_idx]):
                found_lb = True
            if found_lb and not np.prod(indicators_gxn[m_idx]):
                break  # since interval, once find a value that fails, stop searching
    if intersection:
        ci_full = grid[np.where(np.prod(indicators_gxn, axis=1))[0]]
    else:
        ci_full =  grid[np.where(indicators_gxn[:, -1])[0]]
    if ci_full.size == 0:  # grid maybe too coarse
        idx = np.argmax(np.sum(indicators_gxn, axis=1))
        if idx == 0:
            return np.array([grid[0], grid[1]])
        return np.array([grid[idx - 1], grid[idx]])
    return np.array([ci_full.min(), ci_full.max()]) # only output the interval

"""
    OLS algorithm
"""
def ols(features, outcome):
    ols_coeffs = np.linalg.pinv(features).dot(outcome)
    return ols_coeffs

def standard_ols_interval(X, Y, alpha):
    n = X.shape[0]
    d = X.shape[1]
    thetahat = ols(X, Y)
    s = np.sqrt(np.diag( np.linalg.inv(X.T@X) ))
    stderr = s * np.sqrt( 1/(n-d) * ((Y - X@thetahat)**2).sum() )
    halfwidth = norm.ppf(1-alpha/2) * stderr/np.sqrt(n)
    return thetahat - halfwidth, thetahat + halfwidth

def pp_ols_interval(X_labeled, X_unlabeled, Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha):
    n = X_labeled.shape[0]
    N = n + X_unlabeled.shape[0]
    d = X_labeled.shape[1]
    thetatildef = ols(X_unlabeled, Yhat_unlabeled)
    rectifierhat = ols(X_labeled, Y_labeled - Yhat_labeled)
    pp_thetahat = thetatildef + rectifierhat
    s_unlabeled = np.sqrt(np.diag( np.linalg.inv(X_unlabeled.T@X_unlabeled) ))
    s_labeled = np.sqrt(np.diag( np.linalg.inv(X_labeled.T@X_labeled) ))
    stderr_unlabeled = s_unlabeled * np.sqrt( 1/(N-d) * ((Yhat_unlabeled - X_unlabeled@thetatildef)**2).sum() )
    stderr_labeled = s_labeled * np.sqrt( 1/(n-d) * ((Y_labeled - Yhat_labeled - X_labeled@rectifierhat)**2).sum() )
    halfwidth = norm.ppf(1-0.99*alpha/2) * stderr_labeled/np.sqrt(n)   +   norm.ppf(1-0.01*alpha/2) * stderr_unlabeled/np.sqrt(N)
    return pp_thetahat - halfwidth, pp_thetahat + halfwidth

"""
    Logistic regression algorithm
"""
def logistic(X, XTy, lr=1):
    betahat = np.array([0,0])
    for epoch in range(1000):
        p = expit(X@betahat) # This is equal to 1./(1. + np.exp(-X@betahat))
        grad = X.T@p - XTy
        betahat = betahat - lr*grad
    return betahat


def product(*args, **kwds):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def standard_logistic_interval(X, Y, alpha, num_grid=100):
    n = X.shape[0]
    d = X.shape[1]
    XTy = X.T@Y
    mean = 1/n * X.T @ Y
    sigmahat = np.sqrt(1/n * ((X * (Y - mean[None,:]))**2).sum(axis=0))
    XTy_halfwidth = norm.ppf(1-alpha/2) * sigmahat/np.sqrt(n)
    XTy_rectangle_coords = np.linspace(XTy-XTy_halfwidth, XTy+XTy_halfwidth, num_grid)
    XTy_rectangle = product(*tuple([XTy_rectangle_coords[:,j] for j in range(d)]))
    theta_ci = np.array([logistic(X, _XTy) for _XTy in XTy_rectangle])
    return [theta_ci.min(axis=0), theta_ci.max(axis=0)]

def pp_logistic_interval(X_labeled, X_unlabeled, Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha, num_grid=100):
    n = X_labeled.shape[0]
    N = n + X_unlabeled.shape[0]
    d = X_labeled.shape[1]
    rechat = 1/n * X_labeled.T@(Yhat_labeled - Y_labeled)
    sigmahat_rec = np.sqrt(1/n * ((X_labeled * (Y_labeled - rechat[None,:]))**2).sum(axis=0))

    XTy_halfwidth = norm.ppf(1-alpha/2) * sigmahat/np.sqrt(n)
    XTy_rectangle_coords = np.linspace(XTy-XTy_halfwidth, XTy+XTy_halfwidth, num_grid)
    XTy_rectangle = product(*tuple([XTy_rectangle_coords[:,j] for j in range(d)]))
    theta_ci = np.array([logistic(X, _XTy) for _XTy in XTy_rectangle])
    return [theta_ci.min(axis=0), theta_ci.max(axis=0)]

"""
    DISCRETE L_p ESTIMATION RATES
"""

def linfty_dkw(N, K, alpha):
    return np.sqrt(2/N * np.log(2 / alpha))

def linfty_binom(N, K, alpha, qhat):
    epsilon = 0
    for k in np.arange(K):
        bci = binomial_iid(N, alpha/K, qhat[k])
        epsilon = np.maximum(epsilon, np.abs(bci-qhat[k]).max())
    return epsilon

"""
	SAMPLING WITHOUT REPLACEMENT
"""
def clt_swr(x,N,alpha):
    n = x.shape[0]
    point_estimate = x.mean()
    fluctuations = x.std()*norm.cdf(1-alpha/2)*np.sqrt((N-n)/(N*n))
    return np.array([point_estimate-fluctuations, point_estimate+fluctuations])

def wsr_swr(x,N,alpha,grid,num_cpus=10, intersection=True): # x is a [0,1] bounded sequence
    n = x.shape[0]
    def mu(m,i): return (N*m - np.concatenate([np.array([0,]), np.cumsum(x[:i-1])]))/(N - (np.arange(i)+1) + 1 )
    muhats = (1/2 + np.cumsum(x))/(np.arange(n)+1)
    sigmahat2s = (1/4 + np.cumsum((x-muhats)**2))/(np.arange(n)+1)
    lambdas = np.concatenate([np.array([1,]), np.sqrt(2*np.log(2/alpha)/(n*sigmahat2s))[:-1]]) # can't use last entry
    def M(m,i): return 1/2*np.maximum(
        np.prod(1+np.minimum(lambdas[:i], 0.5/mu(m,i))*(x[:i]-mu(m,i))),
        np.prod(1-np.minimum(lambdas[:i], 0.5/(1-mu(m,i)))*(x[:i]-mu(m,i)))
    )
    M = np.vectorize(M)
    if intersection:
        M_list = Parallel(n_jobs=num_cpus)(delayed(M)(grid,i) for i in range(1,n+1))
    else:
        M_list =[M(grid, n),]
    ci_full = grid[np.where(np.prod(np.stack(M_list, axis=1) < 1/alpha , axis=1))[0]]
    return np.array([ci_full.min(), ci_full.max()]) # only output the interval

if __name__ == "__main__":
    x = np.random.choice(np.arange(10),size=(10000,), p=[0.8,0.1,0.05,0.03,0.01,0.005,0.002,0.001,0.001,0.001])
    n = 1000
    print(linfty_dkw(10000,10,0.1))
    print(linfty_binom(10000,10,0.1,x[:n]))
    x_oh = np.take(np.eye(10), x, axis=0)
    qhat = x_oh.mean(axis=0)
    print(qhat)
