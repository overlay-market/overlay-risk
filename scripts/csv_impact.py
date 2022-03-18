import pystable
import pandas as pd
import numpy as np

from scipy import integrate


FILENAME = "data-1625069716_weth-usdc-twap"
FILEPATH = f"csv/{FILENAME}.csv"  # datafile
T = 600  # 10m candle size on datafile
V = 600  # 10m shorter TWAP
CP = 10  # 10x payoff cap

# uncertainties
ALPHAS = np.array([0.01, 0.025, 0.05, 0.075, 0.1])
# target OI threshold beyond which scalp trade is negative EV
Q0S = np.array([0.005 * i for i in range(1, 11, 1)])


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


def delta_long(a: float, b: float, mu: float, sig: float,
               v: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes static spread constant calibration for the long side
    given uncertainty levels `alphas`.

    delta_long = (1/2) * F^{-1}_{X_t}(1-alpha)
    """
    dst_x = pystable.create(alpha=a, beta=b, mu=mu*v,
                            sigma=sig*(v**(1/a)), parameterization=1)

    # calc quantile accounting for cap
    qs_long = pystable.q(dst_x, list(1-alphas), len(alphas))
    qs_long = np.array(qs_long)

    d = qs_long / 2.0
    return d


def delta_short(a: float, b: float, mu: float, sig: float,
                v: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes static spread constant calibration for the short side
    given uncertainty levels `alphas`.

    delta_short = - (1/2) * F^{-1}_{X_t}(alpha)
    """
    dst_x = pystable.create(alpha=a, beta=b, mu=mu*v,
                            sigma=sig*(v**(1/a)), parameterization=1)

    # calc quantile accounting for cap
    qs_short = pystable.q(dst_x, list(alphas), len(alphas))
    qs_short = np.array(qs_short)

    d = - qs_short / 2.0
    return d


def delta(a: float, b: float, mu: float, sig: float,
          v: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computes static spread constant calibration `delta` given uncertainty
    levels `alphas`.

    delta_long = (1/2) * F^{-1}_{X_t}(1-alpha)
    delta_short = - (1/2) * F^{-1}_{X_t}(alpha)
    """
    d_l = delta_long(a, b, mu, sig, v, alphas)
    d_s = delta_short(a, b, mu, sig, v, alphas)

    # choose the max bw the long and short calibrations
    return np.maximum(d_l, d_s)


def lmbda(a: float, b: float, mu: float, sig: float, v: float, g_inv: float,
          alpha: float, q0s: np.ndarray) -> np.ndarray:
    """
    Computes market impact constant calibration `lmbda` given uncertainty
    level `alpha`, payoff cap enforced through `g_inv`, and negative
    EV for open interest cap boundaries of `q0s`.

    rho_long = (int_0^{g_inv} dy e**y * f_Y_m(y)) \
        / (alpha - (1+CP) * (1-F_Y_m(g_inv)))
    rho_short = alpha / (int_{-infty}^{0} dy e**y * f_Y_p(y))

    lmbda_long = ln(rho_long) / (2 * q0)
    lmbda_short = ln(rho_short) / (2 * q0)
    """
    alphas = np.array([alpha])

    # calc long lambda*q
    delta_l = delta_long(a, b, mu, sig, v, alphas)
    dst_y_m = pystable.create(alpha=a, beta=b, mu=mu*v - 2*delta_l,
                              sigma=sig*(v**(1/a)), parameterization=1)

    def integrand_m(y): return pystable.pdf(dst_y_m, [y], 1)[0] * np.exp(y)
    numerator_l, _ = integrate.quad(integrand_m, 0, g_inv)
    denominator_l = alpha - (1+CP)*(1-pystable.cdf(dst_y_m, [g_inv], 1)[0])
    rho_l = numerator_l / denominator_l

    # calc short lambda*q
    delta_s = delta_short(a, b, mu, sig, v, alphas)
    dst_y_p = pystable.create(alpha=a, beta=b, mu=mu*v + 2*delta_s,
                              sigma=sig*(v**(1/a)), parameterization=1)

    def integrand_p(y): return pystable.pdf(dst_y_p, [y], 1)[0] * np.exp(y)
    denominator_s, _ = integrate.quad(integrand_p, -np.inf, 0)
    numerator_s = alpha
    rho_s = numerator_s / denominator_s

    # choose the max bw the long and short calibrations
    return np.maximum(np.log(rho_l) / (2*q0s), np.log(rho_s) / (2*q0s))


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
    g_inv = np.log(1+CP)

    # calc deltas
    deltas = delta(dst.contents.alpha, dst.contents.beta,
                   dst.contents.mu_1, dst.contents.sigma, V, ALPHAS)
    df_deltas = pd.DataFrame(data=[ALPHAS, deltas]).T
    df_deltas.columns = ['alpha', 'delta']
    print('deltas:', df_deltas)
    df_deltas.to_csv(f"csv/metrics/{FILENAME}-deltas.csv", index=False)

    # calc lambda (mkt impact)
    ls = []
    for alpha in ALPHAS:
        lambdas = lmbda(dst.contents.alpha, dst.contents.beta,
                        dst.contents.mu_1, dst.contents.sigma,
                        V, g_inv, alpha, Q0S)
        ls.append(lambdas)

    df_ls = pd.DataFrame(
        data=ls,
        columns=[f"q0={q0}" for q0 in Q0S],
        index=[f"alpha={alpha}" for alpha in ALPHAS]
    )
    print('lambdas:', df_ls)
    df_ls.to_csv(f"csv/metrics/{FILENAME}-lambdas.csv")


if __name__ == '__main__':
    main()