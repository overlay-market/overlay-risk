import pandas as pd
import numpy as np
from scipy.stats import levy_stable
from scipy import integrate

# Constants
FILENAME = "btcd"
FILEPATH = r"C:\Users\HP\Desktop\overlay\overlay-risk\btcd.csv"  # data file path
KS_FILEPATH = r"C:\Users\HP\Desktop\overlay\overlay-risk\btcd-ks.csv"

T =  86400 # 5 m candle size on data file (in seconds)
TC = 600  # 10 m compounding period (in seconds)
CP = 5  # Example payoff cap

# EV are projected over hour intervals in data file
#TS = 5760 * np.array([7, 15, 20, 30])  # 7d, 15d, 20d, 30d (Ti values)
TS = 5760 * np.array([30])  # 7d, 15d, 20d, 30d (Ti values)
ALPHA = 0.01

# 20% inflation per year total for all markets
NUM_MARKETS = 20
INFLATION_PER_YEAR = 0.2
BLOCKS_PER_YEAR = 5760 * 365
IS = TS * (INFLATION_PER_YEAR / NUM_MARKETS) / BLOCKS_PER_YEAR

def gaussian():
    return levy_stable.rvs(alpha=2.0, beta=0.0, loc=0.0, scale=1.0)

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
    sigma_rescaled = sigma * (t ** (1 / alpha))
    return alpha, beta, mu_rescaled, sigma_rescaled

def nexpected_value(alpha, beta, mu, sigma, k, v, g_inv_long, cp, g_inv_short, is_long, t):
    oi_imb = ((1 - 2 * k) ** np.floor(t / v))
    
    # Define the integrand function for integration
    def integrand(y):
        return levy_stable.pdf(y, alpha, beta, loc=mu * t, scale=sigma * (t ** (1 / alpha))) * np.exp(y)
    
    # Limit the integration range to +/- 4 standard deviations around the mean
    mean = mu * t
    stddev = sigma * (t ** (1 / alpha))  # Standard deviation based on scaling
    integration_limit = 4 * stddev  # +/- 4 standard deviations

    if is_long:
        # expected value long
        cdf_x_ginv = levy_stable.cdf(g_inv_long, alpha, beta, loc=mean, scale=stddev)
        integral_long, _ = integrate.quad(integrand, -integration_limit, g_inv_long, epsabs=1e-5, limit=500)
        nev_long = oi_imb * (integral_long - cdf_x_ginv + cp * (1 - cdf_x_ginv))
        return nev_long
    else:
        # expected value short
        cdf_x_ginv_one = levy_stable.cdf(g_inv_short, alpha, beta, loc=mean, scale=stddev)
        integral_short, _ = integrate.quad(integrand, -integration_limit, g_inv_short, epsabs=1e-5, limit=500)
        nev_short = oi_imb * (2 * cdf_x_ginv_one - 1 - integral_short)
        return nev_short


def time_averaged_ev(alpha, beta, mu, sigma, k, v, g_inv_long, cp, g_inv_short, is_long, t):
    def integrand(tau):
        return nexpected_value(alpha, beta, mu, sigma, k, v, g_inv_long, cp, g_inv_short, is_long, tau)

    integral, _ = integrate.quad(integrand, 0, t, limit=500)
    return integral / t

def main():
    print(f'Analyzing file {FILENAME}')
    df = pd.read_csv(FILEPATH)
    p = df['close'].to_numpy() if 'close' in df else df['twap']
    log_close = np.diff(np.log(p))

    # Remove NaNs or infs
    log_close = log_close[np.isfinite(log_close)]

    # Fit Levy stable distribution
    alpha, beta, mu, sigma = levy_stable.fit(log_close)
    print(f"fit params: alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

    # Rescale distribution parameters
    alpha, beta, mu, sigma = rescale_params(alpha, beta, mu, sigma, 1 / T)
    print(f"rescaled params (1/T = {1 / T}): alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

    # Set single k value
    k = 1.16e-7  # Example k value

    # inverse cap
    g_inv = np.log(1 + CP)
    g_inv_one = np.log(2)

    # For different k values at alpha = 0.05 level (diff n calibs),
    # look at time averages of EV for various Ti time periods into the future
    tavg_ev_long = []
    tavg_ev_short = []
    for t in TS:
        print('t', t)

        # time averaged normalized expected value
        tavg_ev_l = time_averaged_ev(alpha, beta, mu, sigma, k, TC, g_inv, CP, g_inv_one, True, t)
        tavg_ev_s = time_averaged_ev(alpha, beta, mu, sigma, k, TC, g_inv, CP, g_inv_one, False, t)

        tavg_ev_long.append(tavg_ev_l)
        tavg_ev_short.append(tavg_ev_s)

        print('tavg_ev_long', tavg_ev_long)
        print('tavg_ev_short', tavg_ev_short)

    # VaR dataframe to csv
    df_tavg_ev_long = pd.DataFrame(
        data=[tavg_ev_long],
        columns=[f"ti={t}" for t in TS]
    )
    df_tavg_ev_short = pd.DataFrame(
        data=[tavg_ev_short],
        columns=[f"ti={t}" for t in TS]
    )
    print(f'tavg ev long (alpha={ALPHA}):', df_tavg_ev_long)
    df_tavg_ev_long.to_csv(f"C:/Users/HP/Desktop/risk/overlay-risk//{FILENAME}-tavg-ev-long-alpha-{ALPHA}.csv")

    print(f'tavg ev short (alpha={ALPHA}):', df_tavg_ev_short)
    df_tavg_ev_short.to_csv(f"C:/Users/HP/Desktop/risk/overlay-risk//{FILENAME}-tavg-ev-short-alpha-{ALPHA}.csv")

    # Cq estimates with respect to total supply
    df_tavg_ev = df_tavg_ev_long.T

    def apply_cq(col):
        return IS / col
    df_cqs = df_tavg_ev.apply(apply_cq, axis=1)
    print(f'cq (alpha={ALPHA}):', df_cqs)
    df_cqs.to_csv(f"C:/Users/HP/Desktop/risk/overlay-risk//{FILENAME}-cq-alpha-{ALPHA}.csv")

if __name__ == '__main__':
    main()