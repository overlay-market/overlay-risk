import pystable
import pandas as pd
import numpy as np
from scipy import integrate
import argparse
import os
import sys

#sys.path.insert(0, os.getcwd()+'/scripts/risk_pipeline')
import helpers  # noqa

TS = 3600 * np.arange(1, 721)  # 1h, 2h, 3h, ...., 30d

# uncertainties
ALPHAS = np.array([0.01, 0.025, 0.05, 0.075, 0.1])
ALPHA = 0.05


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

    args = parser.parse_args()
    return args.filename, args.periodicity


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


def mm_long(a: float, b: float, mu: float, sig: float,
            t: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes maintenance margin for the long side
    given uncertainty levels `alphas`.

    mm_long = [(e) ** -(F^{-1}_{X_w}(alpha))] - 1
    """
    dst_x = pystable.create(alpha=a, beta=b, mu=mu*t,
                            sigma=sig*(t**(1/a)), parameterization=1)

    return np.exp(
            -np.array(pystable.q(dst_x, list(alphas), len(alphas)))
            ) - 1


def mm_short(a: float, b: float, mu: float, sig: float,
             t: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes maintenance margin for the short side
    given uncertainty levels `alphas`.

    mm_short = 1 - [(e) ** -(F^{-1}_{X_w}(1-alpha))]
    """
    dst_x = pystable.create(alpha=a, beta=b, mu=mu*t,
                            sigma=sig*(t**(1/a)), parameterization=1)

    return 1 - np.exp(
                -np.array(pystable.q(dst_x, list(1-alphas), len(alphas)))
                )


def mm(a: float, b: float, mu: float, sig: float,
       t: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes price change bound given uncertainty levels `alphas`.
    """
    mm_l = mm_long(a, b, mu, sig, t, alphas)
    mm_s = mm_short(a, b, mu, sig, t, alphas)

    # choose the max bw the long and short calibrations
    return np.maximum(mm_l, mm_s)


def rho(a: float, b: float, mu: float, sig: float, g_inv_short: float,
        t: float, alpha: float, is_long: bool, mm: float) -> np.ndarray:

    dst_x = pystable.create(alpha=a, beta=b, mu=mu*t,
                            sigma=sig*(t**(1/a)), parameterization=1)

    def integrand(y): return pystable.pdf(dst_x, [y], 1)[0] * np.exp(y)
    if is_long:
        return (1/alpha)*integrate.quad(
                            integrand, -np.inf, -np.log(1+mm)
                            )[0]
    else:
        return (1/alpha)*integrate.quad(
                            integrand, -np.log(1-mm), g_inv_short
                            )[0]


def beta(a: float, b: float, mu: float, sig: float,
         t: float, alpha: float, mm: float) -> np.ndarray:
    """
    Computes liquidate burn constant
    """
    rho_l = rho(a, b, mu, sig, np.log(2), t, alpha, True, mm)
    beta_l = alpha * ((1-rho_l)*(1+(1/mm)) - 1)

    rho_s = rho(a, b, mu, sig, np.log(2), t, alpha, False, mm)
    beta_s = alpha * ((1-rho_s)*(1-(1/mm)) - 1)

    return np.maximum(beta_l, beta_s)


def main(filename, t):
    """
    Fits input csv timeseries data with pystable and generates output
    csv with market impact static spread + slippage params.
    """
    filepath, resultsname, resultspath = helpers.get_paths(filename)

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

    # Calibrate maintenance margin constants
    mms_l = []
    for t in TS:
        # calc MMs
        mms_l.append(
            mm(dst.contents.alpha, dst.contents.beta,
               dst.contents.mu_1, dst.contents.sigma, t, ALPHAS)
        )

    df_mms = pd.DataFrame(data=mms_l,
                          columns=[f'alpha={a}' for a in ALPHAS],
                          index=[f't={t}' for t in TS])
    print('MMs:', df_mms)
    df_mms.to_csv(f"{resultspath}/{resultsname}-mms.csv")

    # Calibrate betas
    betas = []
    for i in range(len(df_mms)):
        print(
            f'Calibrating betas: {np.round((i/len(df_mms))*100,2)}% complete',
            end="\r")
        betas.append(
            beta(dst.contents.alpha, dst.contents.beta,
                 dst.contents.mu_1, dst.contents.sigma,
                 int(df_mms[f'alpha={ALPHA}'].index[i][2:]),
                 ALPHA,
                 df_mms[f'alpha={ALPHA}'][i])
        )
    df_betas = pd.DataFrame(
                    data=betas,
                    columns=[f'alpha={ALPHA}'],
                    index=[f't={t}' for t in TS]
                    )
    # Beta is NaN when MM > 1. Drop those rows.
    df_betas.dropna(inplace=True)
    df_betas.to_csv(f"{resultspath}/{resultsname}-betas.csv")
    return df_mms, df_betas


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        filename, t = get_params()
        main(filename, t)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
