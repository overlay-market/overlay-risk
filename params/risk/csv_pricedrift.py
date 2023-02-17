import pystable
import pandas as pd
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.getcwd()+'/scripts/risk_pipeline')
from helpers import helpers  # noqa

# uncertainties
ALPHAS = np.array([0.001, 0.01, 0.025, 0.05, 0.075, 0.1])


def get_params():
    """
    Get parameters from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename', type=str,
        help='Name of the input data file'
    )
    parser.add_argument(
        '--periodicity', type=int,
        help='Cadence of input data'
    )
    parser.add_argument(
        '--long_twap', type=int,
        help='Longer TWAP'
    )

    args = parser.parse_args()
    return args.filename, args.periodicity, args.long_twap


def gaussian():
    return pystable.create(alpha=2.0, beta=0.0, mu=0.0,
                           sigma=1.0, parameterization=1)


def rescale(dist: pystable.STABLE_DIST, t: float) -> pystable.STABLE_DIST:
    """
    Rescales stable distribution using scaling property.
    For iid X_i ~ S(a, b, mu_i, sigma_i)

    sum_{i=1}^{t} X_i ~ S(a, b, mu, sigma)

    where
      mu = sum_{i=1}^{t} mu_i
      |sigma| = ( sum_{i=1}^{t} |sigma|**(a) ) ** (1/a)

    and `t` input for function is number of iids.
    """
    mu = dist.contents.mu_1 * t
    sigma = dist.contents.sigma * t**(1/dist.contents.alpha)

    return pystable.create(
        alpha=dist.contents.alpha,
        beta=dist.contents.beta,
        mu=mu,
        sigma=sigma,
        parameterization=1
    )


def mu_max_long(a: float, b: float, mu: float, sig: float,
                v: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes price change bound for the long side
    given uncertainty levels `alphas`.

    mu_max_long = (1/v) * (F^{-1}_{X_v}(1-alpha/2))
    """
    dst_x = pystable.create(alpha=a, beta=b, mu=mu*v,
                            sigma=sig*(v**(1/a)), parameterization=1)

    return np.array(pystable.q(dst_x, list(1-alphas/2), len(alphas)))/v


def mu_max_short(a: float, b: float, mu: float, sig: float,
                 v: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes price change bound for the short side
    given uncertainty levels `alphas`.

    mu_max_long = -(1/v) * (F^{-1}_{X_v}(alpha/2))
    """
    dst_x = pystable.create(alpha=a, beta=b, mu=mu*v,
                            sigma=sig*(v**(1/a)), parameterization=1)

    return np.array(pystable.q(dst_x, list(alphas/2), len(alphas)))/v


def mu_max(a: float, b: float, mu: float, sig: float,
           v: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes price change bound given uncertainty levels `alphas`.
    """
    m_l = mu_max_long(a, b, mu, sig, v, alphas)
    m_s = mu_max_short(a, b, mu, sig, v, alphas)

    # choose the max bw the long and short calibrations
    return np.maximum(m_l, m_s)


def main(filename, t, v):
    """
    Fits input csv timeseries data with pystable and generates output
    csv with market impact static spread + slippage params.
    """
    filepath = f"scripts/risk_pipeline/outputs/data/{filename}.csv"  # datafile
    resultsname = filename.replace('_treated', '')
    resultspath = helpers.get_results_dir() + resultsname
    print(f'Analyzing file {filename}')
    df = pd.read_csv(filepath)
    p = df['close'].to_numpy() if 'close' in df else df['twap']
    log_close = [np.log(p[i]/p[i-1]) for i in range(1, len(p))]

    dst = gaussian()  # use gaussian as init dist to fit from
    pystable.fit(dst, log_close, len(log_close))
    print(
        f'''
        fit params: alpha: {dst.contents.alpha}, beta: {dst.contents.beta},
        mu: {dst.contents.mu_1}, sigma: {dst.contents.sigma}
        '''
    )

    dst = rescale(dst, 1/t)
    print(
        f'''
        rescaled params (1/t = {1/t}):
        alpha: {dst.contents.alpha}, beta: {dst.contents.beta},
        mu: {dst.contents.mu_1}, sigma: {dst.contents.sigma}
        '''
    )

    # calc mu_maxs
    mus = mu_max(dst.contents.alpha, dst.contents.beta,
                 dst.contents.mu_1, dst.contents.sigma, v, ALPHAS)
    df_mus = pd.DataFrame(data=[ALPHAS, mus]).T
    df_mus.columns = ['alpha', 'mu_max']
    print('mu_maxs:', df_mus)
    df_mus.to_csv(f"{resultspath}/{resultsname}-mu_maxs.csv", index=False)
    return df_mus


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        filename, t, lt = get_params()
        main(filename, t, lt)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
