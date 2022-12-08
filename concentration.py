import numpy as np
from scipy.stats import binom, norm
from scipy.optimize import brentq
from joblib import delayed, Parallel

"""
    IID
"""
def binomial_iid(N,delta,muhat):
    def invert_upper_tail(mu): return binom.cdf(N*muhat, N, mu) - (delta/2)
    def invert_lower_tail(mu): return binom.cdf(N*muhat, N, mu) - (1-delta/2)
    u = brentq(invert_upper_tail,0,1)
    l = brentq(invert_lower_tail,0,1)
    return np.array([l,u])

def bentkus_iid(N, delta, muhat):
    return binomial_iid(N, delta/np.e, muhat)

def wsr_iid_ana(x,delta,grid,num_cpus=10,step=1): # x is a [0,1] bounded sequence
    n = x.shape[0]
    muhats = (1/2 + np.cumsum(x))/(np.arange(n)+1)
    sigmahat2s = (1/4 + np.cumsum((x-muhats)**2))/(np.arange(n)+1)
    lambdas = np.concatenate([np.array([1,]), np.sqrt(2*np.log(2/delta)/(n*sigmahat2s))[:-1]]) # can't use last entry
    def M(m,i): return 1/2*np.maximum(
        np.prod(1+np.minimum(lambdas[:i], 1/m)*(x[:i]-m)),
        np.prod(1-np.minimum(lambdas[:i], 1/(1-m))*(x[:i]-m))
    )
    M = np.vectorize(M)
    M_list = Parallel(n_jobs=num_cpus)(delayed(M)(grid,i) for i in range(1,n+step,step))
    ci_full = grid[np.where(np.prod(np.stack(M_list, axis=1) < 1/delta , axis=1))[0]]
    return np.array([ci_full.min(), ci_full.max()]) # only output the interval

def wsr_iid(x_n, delta, grid, num_cpus=10, parallelize: bool = False, intersection: bool = True):
    n = x_n.shape[0]
    t_n = np.arange(1, n + 1)
    muhat_n = (0.5 + np.cumsum(x_n)) / (1 + t_n)
    sigma2hat_n = (0.25 + np.cumsum(np.power(x_n - muhat_n, 2))) / (1 + t_n)
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[: -1])
    assert(np.all(sigma2hat_tminus1_n > 0))
    lambda_n = np.sqrt(2 * np.log(2 / delta) / (n * sigma2hat_tminus1_n))

    def M(m):
        lambdaplus_n = np.minimum(lambda_n, 0.75 / m)
        lambdaminus_n = np.minimum(lambda_n, 0.75 / (1 - m))
        return 1/2 * np.maximum(
            np.exp(np.cumsum(np.log(1 + lambdaplus_n * (x_n - m)))),
            np.exp(np.cumsum(np.log(1 - lambdaminus_n * (x_n - m))))
        )

    if parallelize:  # sometimes much slower
        M = np.vectorize(M)
        M_list = Parallel(n_jobs=num_cpus)(delayed(M)(m) for m in grid)
        indicators_gxn = np.array(M_list) < 1 / delta
    else:
        indicators_gxn = np.zeros([grid.size, n])
        found_lb = False
        for m_idx, m in enumerate(grid):
            m_n = M(m)
            indicators_gxn[m_idx] = m_n < 1 / delta
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
    DISCRETE L_p ESTIMATION RATES
"""

def linfty_dkw(N, K, delta):
    return np.sqrt(2/N * np.log(2 / delta)) 

def linfty_binom(N, K, delta, qhat):
    epsilon = 0
    for k in np.arange(K):
        bci = binomial_iid(N, delta/K, qhat[k])
        epsilon = np.maximum(epsilon, np.abs(bci-qhat[k]).max())
    return epsilon

"""
	SAMPLING WITHOUT REPLACEMENT
"""
def clt_swr(x,N,delta):
    n = x.shape[0]
    point_estimate = x.mean()
    fluctuations = x.std()*norm.cdf(1-delta/2)*np.sqrt((N-n)/(N*n))
    return np.array([point_estimate-fluctuations, point_estimate+fluctuations])
	
def wsr_swr(x,N,delta,grid,num_cpus=10): # x is a [0,1] bounded sequence
    n = x.shape[0]
    def mu(m,i): return (N*m - np.concatenate([np.array([0,]), np.cumsum(x[:i-1])]))/(N - (np.arange(i)+1) + 1 )
    muhats = (1/2 + np.cumsum(x))/(np.arange(n)+1)
    sigmahat2s = (1/4 + np.cumsum((x-muhats)**2))/(np.arange(n)+1)
    lambdas = np.concatenate([np.array([1,]), np.sqrt(2*np.log(2/delta)/(n*sigmahat2s))[:-1]]) # can't use last entry
    def M(m,i): return 1/2*np.maximum(
        np.prod(1+np.minimum(lambdas[:i], 1/mu(m,i))*(x[:i]-mu(m,i))),
        np.prod(1-np.minimum(lambdas[:i], 1/(1-mu(m,i)))*(x[:i]-mu(m,i)))
    )
    M = np.vectorize(M)
    M_list = Parallel(n_jobs=num_cpus)(delayed(M)(grid,i) for i in range(1,n+1))
    ci_full = grid[np.where(np.prod(np.stack(M_list, axis=1) < 1/delta , axis=1))[0]]
    return np.array([ci_full.min(), ci_full.max()]) # only output the interval

if __name__ == "__main__":
    x = np.random.choice(np.arange(10),size=(10000,), p=[0.8,0.1,0.05,0.03,0.01,0.005,0.002,0.001,0.001,0.001])
    n = 1000
    print(linfty_dkw(10000,10,0.1))
    print(linfty_binom(10000,10,0.1,x[:n]))
    x_oh = np.take(np.eye(10), x, axis=0)
    qhat = x_oh.mean(axis=0)
    print(qhat)
