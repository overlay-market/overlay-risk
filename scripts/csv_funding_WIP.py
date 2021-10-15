import pystable
import pandas as pd
import numpy as np
import typing as tp

from scipy import integrate
from concurrent.futures import ThreadPoolExecutor


FILENAME = "ethusd_01012020_08232021"
FILEPATH = f"csv/{FILENAME}.csv"  # datafile
T = 4  # 1m candle size on datafile
TC = 40  # 10 m compounding period
CP = 4  # 5x payoff cap

# uncertainties
ALPHAS = np.array([0.01, 0.025, 0.05, 0.075, 0.1])
# periods into the future at which we want 1/compoundingFactor to start
# exceeding VaR from priceFrame: 1/(1-2k)**n >= VaR[(P(n)/P(0) - 1)]
NS = 480 * np.arange(1, 85)  # 2h, 4h, 6h, ...., 7d

# For plotting nvars
TS = 240 * np.arange(1, 720)  # 1h, 2h, 3h, ...., 30d
ALPHA = 0.05


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


def k(a: float, b: float, mu: float, sig: float,
      n: float, v: float, g_inv: float, alphas: np.ndarray) -> np.ndarray:
    dst_y = pystable.create(alpha=a, beta=b, mu=mu*n,
                            sigma=sig*((n/a)**(1/a)), parameterization=1)

    # calc quantile accounting for cap
    cdf_y_ginv = pystable.cdf(dst_y, [g_inv], 1)[0]
    qs = pystable.q(dst_y, list(cdf_y_ginv-alphas), len(alphas))
    qs = np.array(qs)

    # factor at "infty"
    factor_long = np.exp(qs)
    # short has less risk. just needs to have a funding rate to decay
    factor_short = 1 + np.zeros(len(alphas))

    # Compare long vs short and return max of the two
    factor = np.maximum(factor_long, factor_short)

    # want (1-2k)**(n/v) = 1/factor to set k for v timeframe
    # n/v is # of compound periods that pass
    return (1 - (1/factor)**(v/n))/2.0


def nvalue_at_risk(args: tp.Tuple) -> (float, float):
    (a, b, mu, sigma, k_n, v, g_inv, alpha, t) = args

    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t/a)**(1/a), parameterization=1)

    # var long
    cdf_x_ginv = pystable.cdf(x, [g_inv], 1)[0]
    q_long = pystable.q(x, [cdf_x_ginv - alpha], 1)[0]
    nvar_long = ((1-2*k_n)**(np.floor(t/v))) * (np.exp(q_long) - 1)

    # var short
    q_short = pystable.q(x, [alpha], 1)[0]
    nvar_short = ((1-2*k_n)**(np.floor(t/v))) * (1 - np.exp(q_short))

    return nvar_long, nvar_short, k_n, t # TODO:returning an arg seems weird


def nexpected_shortfall(a: float, b: float, mu: float, sigma: float,
                        k_n: float, v: float, g_inv: float, alpha: float,
                        t: float) -> (float, float, float, float):
    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t/a)**(1/a), parameterization=1)
    oi_imb = ((1-2*k_n)**(np.floor(t/v)))

    def integrand(y): return pystable.pdf(x, [y], 1)[0] * np.exp(y)

    # expected shortfall long
    cdf_x_ginv = pystable.cdf(x, [g_inv], 1)[0]
    q_min_long = pystable.q(x, [cdf_x_ginv - alpha], 1)[0]
    integral_long, _ = integrate.quad(integrand, q_min_long, g_inv)
    nes_long = oi_imb * (integral_long/alpha - 1)

    # expected shortfall short
    q_max_short = pystable.q(x, [alpha], 1)[0]
    integral_short, _ = integrate.quad(integrand, -np.inf, q_max_short)
    nes_short = oi_imb * (1 - integral_short/alpha)

    return nes_long, nes_short, nes_long * alpha, nes_short * alpha


def nexpected_value(a: float, b: float, mu: float, sigma: float,
                    k_n: float, v: float, g_inv_long: float, cp: float,
                    g_inv_short: float, t: float) -> (float, float):
    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t/a)**(1/a), parameterization=1)
    oi_imb = ((1-2*k_n)**(np.floor(t/v)))

    def integrand(y): return pystable.pdf(x, [y], 1)[0] * np.exp(y)

    # expected value long
    cdf_x_ginv = pystable.cdf(x, [g_inv_long], 1)[0]
    integral_long, _ = integrate.quad(integrand, -np.inf, g_inv_long)
    nev_long = oi_imb * (integral_long - cdf_x_ginv + cp*(1-cdf_x_ginv))

    # expected value short
    cdf_x_ginv_one = pystable.cdf(x, [g_inv_short], 1)[0]
    integral_short, _ = integrate.quad(integrand, -np.inf, g_inv_short)
    nev_short = oi_imb * (2*cdf_x_ginv_one - 1 - integral_short)

    return nev_long, nev_short


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

    # calc k (funding constant)
    g_inv = np.log(1+CP)
    g_inv_one = np.log(2)
    ks = []
    for n in NS:
        fundings = k(dst.contents.alpha, dst.contents.beta,
                     dst.contents.mu_1, dst.contents.sigma,
                     n, TC, g_inv, ALPHAS)
        ks.append(fundings)

    df_ks = pd.DataFrame(
        data=ks,
        columns=[f"alpha={alpha}" for alpha in ALPHAS],
        index=[f"n={n}" for n in NS]
    )
    print('ks:', df_ks)
    df_ks.to_csv(f"csv/metrics/{FILENAME}-ks.csv")

    # For different k values at alpha = 0.05 level (diff n calibs),
    # plot VaR and ES at times into the future
    nvalue_at_risk_calls = []
    for t in TS:
        for k_n in df_ks[f"alpha={ALPHA}"]:
            nvalue_at_risk_calls.append(
                (dst.contents.alpha, dst.contents.beta,
                dst.contents.mu_1, dst.contents.sigma,
                k_n, TC, g_inv, ALPHA, t)
            )

    nvar_long = pd.DataFrame()
    nvar_short = pd.DataFrame()

    with ThreadPoolExecutor() as executor:
        for nvar_long_i, nvar_short_i, t_i, k_i in executor.map(nvalue_at_risk, nvalue_at_risk_calls):
            nvar_long.loc[t_i, k_i] = nvar_long_i
            nvar_short.loc[t_i ,k_i] = nvar_short_i

    print('nvar_long: ', nvar_long)
    print('nvar_short: ', nvar_short)

if __name__ == '__main__':
    main()
