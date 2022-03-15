import pystable
import pandas as pd
import numpy as np

from scipy import integrate


FILENAME = "data-1625069716_weth-usdc-twap"
FILEPATH = f"csv/{FILENAME}.csv"  # datafile
T = 600  # 10m candle size on datafile (in blocks)
CP = 10  # 10x payoff cap

# uncertainties
ALPHAS = np.array([0.01, 0.025, 0.05, 0.075, 0.1])
# time periods (in seconds) into the future at which we want VaR = 0
NS = 86400 * np.arange(1, 61)  # 1d, 2d, 3d, ...., 60d

# For plotting normalized var, ev, es
TS = 3600 * np.arange(1, 1441)  # 1h, 2h, 3h, ...., 60d
ALPHA = 0.05


# TODO: test
def gaussian():
    return pystable.create(alpha=2.0, beta=0.0, mu=0.0,
                           sigma=1.0, parameterization=1)


# TODO: test
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


def k(a: float, b: float, mu: float, sig: float,
      n: float, alphas: np.ndarray) -> np.ndarray:
    """
    Computed funding constant calibration `k` given uncertainty
    levels `alphas` and anchor time `n`.

    k_long = (1/(2 * n)) * F^{-1}_{X_t}(1-alpha)
    k_short = (1/(2 * n)) * ln[ 2 - F^{-1}_{X_t}(alpha) ]
    """
    dst_y = pystable.create(alpha=a, beta=b, mu=mu*n,
                            sigma=sig*(n**(1/a)), parameterization=1)

    # k long needed for VaR = 0 at n in future
    # NOTE: divide by 1/2T_alpha at return
    qs_long = pystable.q(dst_y, list(1-alphas), len(alphas))
    qs_long = np.array(qs_long)
    k_long = qs_long

    # k short needed for VaR = 0 at n in future
    # NOTE: divide by 1/2T_alpha at return
    qs_short = pystable.q(dst_y, list(alphas), len(alphas))
    qs_short = np.array(qs_short)
    k_short = np.log(2 - np.exp(qs_short))

    # Compare long vs short and return max of the two
    k_max = np.maximum(k_long, k_short)

    # calculate t_alpha: i.e. get n in seconds in the future
    # divide by 1/2T_alpha to get k value then return
    t_alpha = n
    k_max = k_max / (2 * t_alpha)
    return k_max


def nvalue_at_risk(a: float, b: float, mu: float, sigma: float,
                   k_n: float, alpha: float, t: float) -> (float, float):
    """
    Computed value at risk to the protocol at time `t` in the future
    for an initial open interest imbalance to one side, given `k=k_n`
    calibration for the market's funding constant.

    if Q_long = Q and Q_short = 0:
        VaR = Q * [ e**(F^{-1}_{X_t}(1-alpha) - 2k*t) - 1 ]

    else if Q_short = Q and Q_long = 0:
        VaR = Q * [ e**(-2k*t) * ( 2 - e**(F^{-1}_{X_t}(alpha)) ) - 1 ]
    """
    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma * (t**(1/a)), parameterization=1)

    # var long
    q_long = pystable.q(x, [1 - alpha], 1)[0]
    nvar_long = np.exp(q_long - 2 * k_n * t) - 1

    # var short
    q_short = pystable.q(x, [alpha], 1)[0]
    nvar_short = np.exp(- 2 * k_n * t) * (2 - np.exp(q_short)) - 1

    return nvar_long, nvar_short


def nexpected_shortfall(a: float, b: float, mu: float, sigma: float,
                        k_n: float, g_inv: float, cp: float, alpha: float,
                        t: float) -> (float, float, float, float):
    """
    Computed expected shortfall (conditional & unconditional) at time `t`
    in the future for an initial open interest imbalance to one side, given
    `k=k_n` calibration for the market's funding constant and payoff cap `cp`.

    if Q_long = Q and Q_short = 0:
        ES = Q * { (e**(-2k*t) / alpha) * [
            int_{F^{-1}_{X_t}(1-alpha)}^{g^{-1}(C_p)} dx e**x * f_{X_t}
            + (1 + C_p) * (1 - F_{X_t}(C_p))
        ] - 1 }

    else if Q_short = Q and Q_long = 0:
        ES = Q * { (e**(-2k*t) / alpha) * [
            2 - (1/alpha) * int_{-infty}^{F^{-1}_{X_t}(alpha)} dx e**x *f_{X_t}
        ] - 1}
    """
    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t**(1/a)), parameterization=1)

    def integrand(y): return pystable.pdf(x, [y], 1)[0] * np.exp(y)

    # expected shortfall long
    cdf_x_ginv = pystable.cdf(x, [g_inv], 1)[0]
    q_min_long = pystable.q(x, [1 - alpha], 1)[0]

    integral_long = 0
    if g_inv > q_min_long:
        integral_long, _ = integrate.quad(integrand, q_min_long, g_inv)

    nes_long = (
        np.exp(-2 * k_n * t) / alpha) * \
        (integral_long + (1+cp) * (1 - cdf_x_ginv)) - 1

    # expected shortfall short
    q_max_short = pystable.q(x, [alpha], 1)[0]
    integral_short, _ = integrate.quad(integrand, -np.inf, q_max_short)
    nes_short = np.exp(-2 * k_n * t) * (2 - integral_short / alpha) - 1

    return nes_long, nes_short, nes_long * alpha, nes_short * alpha


def nexpected_value(a: float, b: float, mu: float, sigma: float,
                    k_n: float, g_inv_long: float, cp: float,
                    g_inv_short: float, t: float) -> (float, float):
    """
    Computed expected value at time `t` in the future for an initial
    open interest imbalance to one side, given `k=k_n` calibration for
    the market's funding constant and payoff cap `cp`.

    TODO: test expressions are the same as if did integral
    TODO: with min(g(x), 1+C_p)

    if Q_long = Q and Q_short = 0:
        EV = Q * { e**(-2k*t) * [
            int_{-infty}^{g^{-1}(C_P)} dx e**x * f_{X_t}
            + (1+C_P) * (1 - F_{X_t}(g^{-1}(C_P)))
        ] - 1 }

    else if Q_short = Q and Q_long = 0:
        EV = Q * { e**(-2k*t) * [
            2 * F_{X_t}(g^{-1}(1))
            - int_{-infty}^{g^{-1}(1)} dx e**x * f_{X_t}
        ] - 1}
    """
    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t**(1/a)), parameterization=1)

    def integrand(y): return pystable.pdf(x, [y], 1)[0] * np.exp(y)

    # expected value long
    cdf_x_ginv = pystable.cdf(x, [g_inv_long], 1)[0]
    integral_long, _ = integrate.quad(integrand, -np.inf, g_inv_long)
    nev_long = np.exp(-2*k_n*t) * (integral_long + (1+cp)*(1-cdf_x_ginv)) - 1

    # expected value short
    cdf_x_ginv_one = pystable.cdf(x, [g_inv_short], 1)[0]
    integral_short, _ = integrate.quad(integrand, -np.inf, g_inv_short)
    nev_short = np.exp(-2*k_n*t) * (2 * cdf_x_ginv_one - integral_short) - 1

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

    # rescale to per second distribution
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
        fundings = k(a=dst.contents.alpha, b=dst.contents.beta,
                     mu=dst.contents.mu_1, sig=dst.contents.sigma,
                     n=n, alphas=ALPHAS)
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
    nvars_long = []
    nvars_short = []
    ness_long = []
    ness_short = []
    nevs_long = []
    nevs_short = []
    for t in TS:
        nvar_t_long = []
        nvar_t_short = []

        ness_t_long = []
        ness_t_short = []

        nevs_t_long = []
        nevs_t_short = []

        for k_n in df_ks[f"alpha={ALPHA}"]:
            # normalized value at risk
            nvar_long, nvar_short = nvalue_at_risk(
                a=dst.contents.alpha, b=dst.contents.beta,
                mu=dst.contents.mu_1, sigma=dst.contents.sigma,
                k_n=k_n, alpha=ALPHA, t=t)
            nvar_t_long.append(nvar_long)
            nvar_t_short.append(nvar_short)

            # normalized expected shortfall (conditional & unconditional)
            nes_long, nes_short, nes_long_uncond, nes_short_uncond = \
                nexpected_shortfall(
                    a=dst.contents.alpha, b=dst.contents.beta,
                    mu=dst.contents.mu_1, sigma=dst.contents.sigma,
                    k_n=k_n, g_inv=g_inv, cp=CP, alpha=ALPHA, t=t)
            ness_t_long.append(nes_long)
            ness_t_short.append(nes_short)

            # normalized expected value
            nev_long, nev_short = \
                nexpected_value(
                    a=dst.contents.alpha, b=dst.contents.beta,
                    mu=dst.contents.mu_1, sigma=dst.contents.sigma,
                    k_n=k_n, g_inv_long=g_inv, cp=CP,
                    g_inv_short=g_inv_one, t=t)
            nevs_t_long.append(nev_long)
            nevs_t_short.append(nev_short)

        nvars_long.append(nvar_t_long)
        nvars_short.append(nvar_t_short)

        print('t', t)
        print('ness_t_long', ness_t_long)
        print('ness_t_short', ness_t_short)

        ness_long.append(ness_t_long)
        ness_short.append(ness_t_short)

        print('t', t)
        print('nevs_t_long', nevs_t_long)
        print('nevs_t_short', nevs_t_short)

        nevs_long.append(nevs_t_long)
        nevs_short.append(nevs_t_short)

    # VaR dataframe to csv
    df_nvars_long = pd.DataFrame(
        data=nvars_long,
        columns=[f"k={k_n}" for k_n in df_ks[f"alpha={ALPHA}"]],
        index=[f"t={t}" for t in TS]
    )
    df_nvars_short = pd.DataFrame(
        data=nvars_short,
        columns=[f"k={k_n}" for k_n in df_ks[f"alpha={ALPHA}"]],
        index=[f"t={t}" for t in TS]
    )
    print(f'nvars long (alpha={ALPHA}):', df_nvars_long)
    df_nvars_long.to_csv(
        f"csv/metrics/{FILENAME}-nvars-long-alpha-{ALPHA}.csv")

    print(f'nvars short (alpha={ALPHA}):', df_nvars_short)
    df_nvars_short.to_csv(
        f"csv/metrics/{FILENAME}-nvars-short-alpha-{ALPHA}.csv")

    # Expected shortfall dataframe to csv
    df_ness_long = pd.DataFrame(
        data=ness_long,
        columns=[f"k={k_n}" for k_n in df_ks[f"alpha={ALPHA}"]],
        index=[f"t={t}" for t in TS]
    )
    df_ness_short = pd.DataFrame(
        data=ness_short,
        columns=[f"k={k_n}" for k_n in df_ks[f"alpha={ALPHA}"]],
        index=[f"t={t}" for t in TS]
    )
    print(f'ness long (alpha={ALPHA}):', df_ness_long)
    df_ness_long.to_csv(
        f"csv/metrics/{FILENAME}-ness-long-conditional-alpha-{ALPHA}.csv")

    print(f'ness short (alpha={ALPHA}):', df_ness_short)
    df_ness_short.to_csv(
        f"csv/metrics/{FILENAME}-ness-short-conditional-alpha-{ALPHA}.csv")

    # Expected value dataframe to csv
    df_nevs_long = pd.DataFrame(
        data=nevs_long,
        columns=[f"k={k_n}" for k_n in df_ks[f"alpha={ALPHA}"]],
        index=[f"t={t}" for t in TS]
    )
    df_nevs_short = pd.DataFrame(
        data=nevs_short,
        columns=[f"k={k_n}" for k_n in df_ks[f"alpha={ALPHA}"]],
        index=[f"t={t}" for t in TS]
    )
    print(f'nevs long (alpha={ALPHA}):', df_nevs_long)
    df_nevs_long.to_csv(
        f"csv/metrics/{FILENAME}-nevs-long-alpha-{ALPHA}.csv")

    print(f'nevs short (alpha={ALPHA}):', df_nevs_short)
    df_nevs_short.to_csv(
        f"csv/metrics/{FILENAME}-nevs-short-alpha-{ALPHA}.csv")


if __name__ == '__main__':
    main()
