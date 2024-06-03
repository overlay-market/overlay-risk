import pystable
import pandas as pd
import numpy as np
import typing as tp
import time

from scipy import integrate
from concurrent.futures import ProcessPoolExecutor


FILENAME = "data-1625069716_weth-usdc-twap"
FILEPATH = f"csv/{FILENAME}.csv"  # datafile

KS_FILEPATH = f"csv/metrics/{FILENAME}-ks.csv"

T = 40  # 10m candle size on datafile
TC = 40  # 10 m compounding period
CP = 4  # 5x payoff cap

# EV are projected over hour intervals in datafile
TS = 5760 * np.array([7, 15, 20, 30])  # 7d, 15d, 20d, 30d (Ti values)
ALPHA = 0.05

# 20% inflation per year total for all markets
NUM_MARKETS = 5
INFLATION_PER_YEAR = 0.2
BLOCKS_PER_YEAR = 5760*365
IS = TS * (INFLATION_PER_YEAR / NUM_MARKETS) / BLOCKS_PER_YEAR


def gaussian():
    return pystable.create(alpha=2.0, beta=0.0, mu=0.0,
                           sigma=1.0, parameterization=1)


def rescale(dist: pystable.STABLE_DIST, t: float) -> pystable.STABLE_DIST:
    mu = dist.contents.mu_1 * t
    if t > 1:
        sigma = dist.contents.sigma * \
            (t/dist.contents.alpha)**(1/dist.contents.alpha)
    else:
        sigma = dist.contents.sigma * \
            ((1/t)/dist.contents.alpha)**(-1/dist.contents.alpha)

    return pystable.create(
        alpha=dist.contents.alpha,
        beta=dist.contents.beta,
        mu=mu,
        sigma=sigma,
        parameterization=1
    )


def nexpected_value(a: float, b: float, mu: float, sigma: float,
                    k: float, v: float, g_inv_long: float, cp: float,
                    g_inv_short: float, is_long: bool, t: float) -> float:
    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t/a)**(1/a), parameterization=1)
    oi_imb = ((1-2*k)**(np.floor(t/v)))

    def integrand(y): return pystable.pdf(x, [y], 1)[0] * np.exp(y)

    if is_long:
        # expected value long
        cdf_x_ginv = pystable.cdf(x, [g_inv_long], 1)[0]
        integral_long, _ = integrate.quad(integrand, -np.inf, g_inv_long)
        nev_long = oi_imb * (integral_long - cdf_x_ginv + cp*(1-cdf_x_ginv))
        return nev_long
    else:
        # expected value short
        cdf_x_ginv_one = pystable.cdf(x, [g_inv_short], 1)[0]
        integral_short, _ = integrate.quad(integrand, -np.inf, g_inv_short)
        nev_short = oi_imb * (2*cdf_x_ginv_one - 1 - integral_short)
        return nev_short


def time_averaged_ev(args: tp.Tuple) -> float:
    (a, b, mu, sigma, k, v, g_inv_long, cp, g_inv_short, is_long, t) = args

    def integrand(tau):
        return nexpected_value(a=a, b=b, mu=mu, sigma=sigma, k=k, v=v,
                               g_inv_long=g_inv_long, cp=cp,
                               g_inv_short=g_inv_short, is_long=is_long,
                               t=tau)

    integral, integral_err = integrate.quad(integrand, 0, t)
    print(t, ', ', k, ', ', is_long)
    return {
        "tavg_ev": integral / t,
        "k": k,
        "t": t,
        "is_long": is_long
        }


def main():
    """
    Fits input csv timeseries data with pystable and generates output
    csv with funding constant params.
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

    df_ks = pd.read_csv(KS_FILEPATH)
    print('df_ks[ALPHA]', df_ks[f"alpha={ALPHA}"])

    # inverse cap
    g_inv = np.log(1+CP)
    g_inv_one = np.log(2)

    # For different k values at alpha = 0.05 level (diff n calibs),
    # look at time averages of EV for various Ti time periods into the future
    tavg_ev_calls = []
    for t in TS:
        for k in df_ks[f"alpha={ALPHA}"]:
            for is_long in [True, False]:
                tavg_ev_calls.append(
                    (
                        dst.contents.alpha, dst.contents.beta,
                        dst.contents.mu_1, dst.contents.sigma,
                        k, TC, g_inv, CP, g_inv_one,
                        is_long, t
                    )
                )

    start_time = time.time()
    tavg_ev_values = []
    with ProcessPoolExecutor() as executor:
        for item in executor.map(time_averaged_ev, tavg_ev_calls):
            tavg_ev_values.append(item)

    print('tavg_ev_values: ', tavg_ev_values)
    print('time taken: ', time.time()-start_time)


if __name__ == '__main__':
    main()
