import pandas as pd
import numpy as np
from scipy.stats import levy_stable
from scipy import integrate

# Constants
FILENAME = "historical_data"
FILEPATH = f"C:/Users/HP/Desktop/risk/overlay-risk/{FILENAME}.csv"  # data file path
KS_FILEPATH = f"C:/Users/HP/Desktop/risk/overlay-risk/metrics/{FILENAME}-ks.csv"

T = 300  # 5 m candle size on data file (in seconds)
TC = 600  # 10 m compounding period (in seconds)
CP = 5  # Example payoff cap

# EV are projected over hour intervals in data file
TS = 5760 * np.array([7, 15, 20, 30])  # 7d, 15d, 20d, 30d (Ti values)
ALPHA = 0.05

# 20% inflation per year total for all markets
NUM_MARKETS = 9
INFLATION_PER_YEAR = 0.2
BLOCKS_PER_YEAR = 5760 * 365
IS = TS * (INFLATION_PER_YEAR / NUM_MARKETS) / BLOCKS_PER_YEAR

def gaussian():
    return levy_stable.create(alpha=2.0, beta=0.0, loc=0.0, scale=1.0)

def rescale_params(alpha, beta, mu, sigma, t):
    """
    Rescales Levy stable distribution parameters using the scaling property.
    
    Args:
    alpha, beta, mu, sigma (float): Parameters of the Levy stable distribution.
    t (float): Scaling factor.

    Returns:
    tuple: Rescaled parameters (alpha, beta, mu_rescaled, sigma_rescaled).
    """
    mu_rescaled = mu * t
    if t > 1:
        sigma_rescaled = sigma * (t / alpha) ** (1 / alpha)
    else:
        sigma_rescaled = sigma * ((1 / t) / alpha) ** (-1 / alpha)
    return alpha, beta, mu_rescaled, sigma_rescaled

def nexpected_value(alpha, beta, mu, sigma, k, v, g_inv_long, cp, g_inv_short, is_long, t):
    x = levy_stable(alpha, beta, loc=mu * t, scale=sigma * (t ** (1 / alpha)))
    oi_imb = ((1 - 2 * k) ** np.floor(t / v))

    def integrand(y):
        return levy_stable.pdf(y, alpha, beta, loc=mu * t, scale=sigma * (t ** (1 / alpha))) * np.exp(y)

    if is_long:
        # expected value long
        cdf_x_ginv = levy_stable.cdf(g_inv_long, alpha, beta, loc=mu * t, scale=sigma * (t ** (1 / alpha)))
        integral_long, _ = integrate.quad(integrand, -np.inf, g_inv_long)
        nev_long = oi_imb * (integral_long - cdf_x_ginv + cp * (1 - cdf_x_ginv))
        return nev_long
    else:
        # expected value short
        cdf_x_ginv_one = levy_stable.cdf(g_inv_short, alpha, beta, loc=mu * t, scale=sigma * (t ** (1 / alpha)))
        integral_short, _ = integrate.quad(integrand, -np.inf, g_inv_short)
        nev_short = oi_imb * (2 * cdf_x_ginv_one - 1 - integral_short)
        return nev_short

def time_averaged_ev(alpha, beta, mu, sigma, k, v, g_inv_long, cp, g_inv_short, is_long, t):
    def integrand(tau):
        return nexpected_value(alpha, beta, mu, sigma, k, v, g_inv_long, cp, g_inv_short, is_long, tau)

    integral, _ = integrate.quad(integrand, 0, t)
    return integral / t

def main():
    print(f'Analyzing file {FILENAME}')
    df = pd.read_csv(FILEPATH)
    p = df['close'].to_numpy() if 'close' in df else df['twap']
    log_close = np.diff(np.log(p))

    # Fit Levy stable distribution
    alpha, beta, mu, sigma = levy_stable.fit(log_close)
    print(f"fit params: alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

    # Rescale distribution parameters
    alpha, beta, mu, sigma = rescale_params(alpha, beta, mu, sigma, 1 / T)
    print(f"rescaled params (1/T = {1 / T}): alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

    df_ks = pd.read_csv(KS_FILEPATH)
    print('df_ks[ALPHA]', df_ks[f"alpha={ALPHA}"])

    # inverse cap
    g_inv = np.log(1 + CP)
    g_inv_one = np.log(2)

    # For different k values at alpha = 0.05 level (diff n calibs),
    # look at time averages of EV for various Ti time periods into the future
    tavg_ev_long = []
    tavg_ev_short = []
    for t in TS:
        print('t', t)

        tavg_ev_t_long = []
        tavg_ev_t_short = []

        for k in df_ks[f"alpha={ALPHA}"]:
            print('k', k)
            # time averaged normalized expected value
            tavg_ev_l = time_averaged_ev(alpha, beta, mu, sigma, k, TC, g_inv, CP, g_inv_one, True, t)
            tavg_ev_s = time_averaged_ev(alpha, beta, mu, sigma, k, TC, g_inv, CP, g_inv_one, False, t)

            tavg_ev_t_long.append(tavg_ev_l)
            tavg_ev_t_short.append(tavg_ev_s)

        tavg_ev_long.append(tavg_ev_t_long)
        tavg_ev_short.append(tavg_ev_t_short)

        print('tavg_ev_long', tavg_ev_long)
        print('tavg_ev_short', tavg_ev_short)

    # VaR dataframe to csv
    df_tavg_ev_long = pd.DataFrame(
        data=tavg_ev_long,
        columns=[f"k={k_n}" for k_n in df_ks[f"alpha={ALPHA}"]],
        index=[f"ti={t}" for t in TS]
    )
    df_tavg_ev_short = pd.DataFrame(
        data=tavg_ev_short,
        columns=[f"k={k_n}" for k_n in df_ks[f"alpha={ALPHA}"]],
        index=[f"ti={t}" for t in TS]
    )
    print(f'tavg ev long (alpha={ALPHA}):', df_tavg_ev_long)
    df_tavg_ev_long.to_csv(f"C:/Users/HP/Desktop/risk/overlay-risk/metrics/{FILENAME}-tavg-ev-long-alpha-{ALPHA}.csv")

    print(f'tavg ev short (alpha={ALPHA}):', df_tavg_ev_short)
    df_tavg_ev_short.to_csv(f"C:/Users/HP/Desktop/risk/overlay-risk/metrics/{FILENAME}-tavg-ev-short-alpha-{ALPHA}.csv")

    # Cq estimates with respect to total supply
    df_tavg_ev = df_tavg_ev_long

    def apply_cq(col):
        return IS / col
    df_cqs = df_tavg_ev.apply(apply_cq)
    print(f'cq (alpha={ALPHA}):', df_cqs)
    df_cqs.to_csv(f"C:/Users/HP/Desktop/risk/overlay-risk/metrics/{FILENAME}-cq-alpha-{ALPHA}.csv")

if __name__ == '__main__':
    main()
