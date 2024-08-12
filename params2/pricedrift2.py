import pandas as pd
import numpy as np
from scipy.stats import levy_stable
import os

# uncertainties
ALPHAS = np.array([0.05])

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

def mu_max_long(alpha, beta, mu, sigma, v, alphas):
    """
    Computes price change bound for the long side given uncertainty levels `alphas`.
    
    Args:
    alpha, beta, mu, sigma (float): Parameters of the Levy stable distribution.
    v (float): Time horizon.
    alphas (np.ndarray): Array of uncertainty levels.

    Returns:
    np.ndarray: Array of maximum price changes for the long side.
    """
    return levy_stable.ppf(1 - alphas/2, alpha, beta, loc=mu * v, scale=sigma * (v ** (1 / alpha))) / v

def mu_max_short(alpha, beta, mu, sigma, v, alphas):
    """
    Computes price change bound for the short side given uncertainty levels `alphas`.
    
    Args:
    alpha, beta, mu, sigma (float): Parameters of the Levy stable distribution.
    v (float): Time horizon.
    alphas (np.ndarray): Array of uncertainty levels.

    Returns:
    np.ndarray: Array of maximum price changes for the short side.
    """
    return -levy_stable.ppf(alphas/2, alpha, beta, loc=mu * v, scale=sigma * (v ** (1 / alpha))) / v

def mu_max(alpha, beta, mu, sigma, v, alphas):
    """
    Computes price change bound given uncertainty levels `alphas`.
    
    Args:
    alpha, beta, mu, sigma (float): Parameters of the Levy stable distribution.
    v (float): Time horizon.
    alphas (np.ndarray): Array of uncertainty levels.

    Returns:
    np.ndarray: Array of maximum price changes.
    """
    m_l = mu_max_long(alpha, beta, mu, sigma, v, alphas)
    m_s = mu_max_short(alpha, beta, mu, sigma, v, alphas)
    return np.maximum(m_l, m_s)

def main(data_file, alpha_level):
    filename = data_file
    t = 86400
    v = 432000  # Example longer TWAP (5 days in seconds)

    print(f'Analyzing file {filename}')
    df = pd.read_csv(filename)
    p = df['close'].to_numpy() if 'close' in df else df['twap']
    log_close = np.log(p[1:] / p[:-1])

    alpha, beta, mu, sigma = levy_stable.fit(log_close)
    print(f"fit params: alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

    alpha, beta, mu, sigma = rescale_params(alpha, beta, mu, sigma, 1/t)
    print(f"rescaled params (1/t = {1/t}): alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

    mus = mu_max(alpha, beta, mu, sigma, v, ALPHAS)
    df_mus = pd.DataFrame(data={'alpha': ALPHAS, 'mu_max': mus})
    print('mu_maxs:', df_mus)

    resultspath = os.path.dirname(filename)
    resultsname = os.path.basename(filename).rsplit('.', 1)[0]
    df_mus.to_csv(f"{resultspath}/{resultsname}-mu_maxs.csv", index=False)

    return df_mus

