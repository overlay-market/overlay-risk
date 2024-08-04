import pandas as pd
import numpy as np
from scipy.stats import levy_stable
from scipy import integrate

# Constants
TS = 3600 * np.arange(1, 721)  # 1h, 2h, 3h, ...., 30d
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05

def rescale_params(alpha, beta, mu, sigma, t):
    """
    Rescales Levy stable distribution parameters using the scaling property.

    Args:
        alpha (float): Stability parameter (0 < alpha ≤ 2).
        beta (float): Skewness parameter (-1 ≤ beta ≤ 1).
        mu (float): Location parameter.
        sigma (float): Scale parameter.
        t (float): Time scaling factor.

    Returns:
        tuple: Rescaled parameters (alpha, beta, mu_rescaled, sigma_rescaled).
    """
    mu_rescaled = mu * t
    sigma_rescaled = sigma * (t ** (1 / alpha))
    return alpha, beta, mu_rescaled, sigma_rescaled

def mm_long(alpha, beta, mu, sigma, t, alpha_level):
    """
    Computes maintenance margin for the long side given a confidence level.

    Args:
        alpha (float): Stability parameter (0 < alpha ≤ 2).
        beta (float): Skewness parameter (-1 ≤ beta ≤ 1).
        mu (float): Location parameter.
        sigma (float): Scale parameter.
        t (float): Time frame in seconds.
        alpha_level (float): Confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        float: Maintenance margin for the long side.
    """
    return np.exp(-levy_stable.ppf(alpha_level, alpha, beta, loc=mu * t, scale=sigma * (t ** (1 / alpha)))) - 1

def mm_short(alpha, beta, mu, sigma, t, alpha_level):
    """
    Computes maintenance margin for the short side given a confidence level.

    Args:
        alpha (float): Stability parameter (0 < alpha ≤ 2).
        beta (float): Skewness parameter (-1 ≤ beta ≤ 1).
        mu (float): Location parameter.
        sigma (float): Scale parameter.
        t (float): Time frame in seconds.
        alpha_level (float): Confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        float: Maintenance margin for the short side.
    """
    return 1 - np.exp(-levy_stable.ppf(1 - alpha_level, alpha, beta, loc=mu * t, scale=sigma * (t ** (1 / alpha))))

def mm(alpha, beta, mu, sigma, t, alpha_level):
    """
    Computes the maximum maintenance margin for long and short positions given a confidence level.

    Args:
        alpha (float): Stability parameter (0 < alpha ≤ 2).
        beta (float): Skewness parameter (-1 ≤ beta ≤ 1).
        mu (float): Location parameter.
        sigma (float): Scale parameter.
        t (float): Time frame in seconds.
        alpha_level (float): Confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        float: Maximum maintenance margin for the given parameters and time frame.
    """
    mm_l = mm_long(alpha, beta, mu, sigma, t, alpha_level)
    mm_s = mm_short(alpha, beta, mu, sigma, t, alpha_level)
    return np.maximum(mm_l, mm_s)

def rho(alpha, beta, mu, sigma, g_inv_short, t, alpha_level, is_long, mm):
    """
    Computes the expected loss (rho) for a given position.

    Args:
        alpha (float): Stability parameter (0 < alpha ≤ 2).
        beta (float): Skewness parameter (-1 ≤ beta ≤ 1).
        mu (float): Location parameter.
        sigma (float): Scale parameter.
        g_inv_short (float): Inverse gamma for the short position calculation.
        t (float): Time frame in seconds.
        alpha_level (float): Confidence level.
        is_long (bool): True if the position is long, False if short.
        mm (float): Maintenance margin.

    Returns:
        float: Expected loss (rho) for the given position and parameters.
    """
    def integrand(y):
        return levy_stable.pdf(y, alpha, beta, loc=mu * t, scale=sigma * (t ** (1 / alpha))) * np.exp(y)
    
    if is_long:
        return (1 / alpha_level) * integrate.quad(integrand, -np.inf, -np.log(1 + mm))[0]
    else:
        return (1 / alpha_level) * integrate.quad(integrand, -np.log(1 - mm), g_inv_short)[0]

def calculate_beta(alpha, beta, mu, sigma, t, alpha_level, mm):
    """
    Calculates the beta value for liquidation risk.

    Args:
        alpha (float): Stability parameter (0 < alpha ≤ 2).
        beta (float): Skewness parameter (-1 ≤ beta ≤ 1).
        mu (float): Location parameter.
        sigma (float): Scale parameter.
        t (float): Time frame in seconds.
        alpha_level (float): Confidence level.
        mm (float): Maintenance margin.

    Returns:
        float: Beta value indicating the risk adjustment for the position.
    """
    rho_l = rho(alpha, beta, mu, sigma, np.log(2), t, alpha_level, True, mm)
    beta_l = alpha_level * ((1 - rho_l) * (1 + (1 / mm)) - 1)

    rho_s = rho(alpha, beta, mu, sigma, np.log(2), t, alpha_level, False, mm)
    beta_s = alpha_level * ((1 - rho_s) * (1 - (1 / mm)) - 1)

    return np.maximum(beta_l, beta_s)

def main():
    """
    Main function to fit input CSV timeseries data with Levy stable distribution and generate
    output CSV with market impact, static spread, and slippage parameters.

    It reads the market data, fits the Levy stable distribution, rescales parameters,
    and calculates maintenance margins and beta values for each time frame.

    Returns:
        pd.DataFrame: Combined DataFrame with maintenance margin and beta values.
    """
    filename = r"C:\Users\HP\Desktop\risk\overlay-risk\historical_data.csv"
    t = 86400  # Example periodicity (1 day in seconds)

    print(f'Analyzing file {filename}')
    df = pd.read_csv(filename)
    p = df['close'].to_numpy() if 'close' in df else df['twap']
    log_close = [np.log(p[i] / p[i - 1]) for i in range(1, len(p))]

    alpha, beta, mu, sigma = levy_stable.fit(log_close)
    print(f"fit params: alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

    alpha, beta, mu, sigma = rescale_params(alpha, beta, mu, sigma, 1/t)
    print(f"rescaled params (1/t = {1/t}): alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

    data = []
    for t in TS:
        mm_value = mm(alpha, beta, mu, sigma, t, ALPHA)
        beta_value = calculate_beta(alpha, beta, mu, sigma, t, ALPHA, mm_value)
        data.append((t, mm_value, beta_value))

    # Adding detailed column names
    df_combined = pd.DataFrame(
        data,
        columns=[
            'time_frame_seconds',
            f'mm_alpha={ALPHA}_beta={beta}_mu={mu}_sigma={sigma}',
            f'beta_alpha={ALPHA}_beta={beta}_mu={mu}_sigma={sigma}'
        ]
    )
    print('Combined DataFrame:', df_combined)
    df_combined.to_csv(f"{filename.rsplit('.', 1)[0]}-combined.csv", index=False)

    return df_combined

if __name__ == '__main__':
    main()
