import pystable
import pandas as pd
import numpy as np


FILENAME = "ETHUSD-600-20210401-20210630"
FILEPATH = f"csv/{FILENAME}.csv"  # datafile
T = 600  # 10m candle size on datafile
TS = 3600 * np.arange(1, 721)  # 1h, 2h, 3h, ...., 30d

# uncertainties
ALPHAS = np.array([0.01, 0.025, 0.05, 0.075, 0.1])


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


def main():
    """
    Fits input csv timeseries data with pystable and generates output
    csv with market impact static spread + slippage params.
    """
    print(f'Analyzing file {FILENAME}')
    df = pd.read_csv(FILEPATH)
    p = df['c'].to_numpy() if 'c' in df else df['twap']
    log_close = [np.log(p[i]/p[i-1]) for i in range(1, len(p))]

    dst = gaussian()  # use gaussian as init dist to fit from
    pystable.fit(dst, log_close, len(log_close))
    print(
        f'''
        fit params: alpha: {dst.contents.alpha}, beta: {dst.contents.beta},
        mu: {dst.contents.mu_1}, sigma: {dst.contents.sigma}
        '''
    )

    dst = rescale(dst, 1/T)
    print(
        f'''
        rescaled params (1/T = {1/T}):
        alpha: {dst.contents.alpha}, beta: {dst.contents.beta},
        mu: {dst.contents.mu_1}, sigma: {dst.contents.sigma}
        '''
    )

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
    df_mms.to_csv(f"csv/metrics/{FILENAME}-mms.csv", index=False)


if __name__ == '__main__':
    main()
