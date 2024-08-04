from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import levy_stable

# Constants
ALPHAS = np.array([0.01, 0.025, 0.05, 0.075, 0.1])
NS = 86400 * np.arange(1, 31)  # 1d, 2d, 3d, ...., 31
TS = 3600 * 12 * np.arange(1, 121)  # 12h, 24h, 36h, ...., 60d
ALPHA = 0.05

def fit_levy_stable(log_returns):
    return levy_stable.fit(log_returns)

def calculate_kl(alpha, beta, mu, sigma, n, alpha_level):
    # Inverse CDF (quantile) for the 1 - alpha level
    F_inv_1_minus_alpha = levy_stable.ppf(1 - alpha_level, alpha, beta, loc=mu, scale=sigma)
    # Calculate kl
    kl = 1 / (2 * (n / 86400)) * F_inv_1_minus_alpha
    return kl

def calculate_ks(alpha, beta, mu, sigma, n, alpha_level):
    # Inverse CDF (quantile) for the alpha level
    F_inv_alpha = levy_stable.ppf(alpha_level, alpha, beta, loc=mu, scale=sigma)
    # Calculate ks
    ks = 1 / (2 * (n / 86400)) * np.log(2 - np.exp(F_inv_alpha))
    return ks

def calibrate_k(alpha, beta, mu, sigma, ns, alpha_level):
    ks_values = []
    for n in ns:
        # Calculate kl and ks
        kl = calculate_kl(alpha, beta, mu, sigma, n, alpha_level)
        ks = calculate_ks(alpha, beta, mu, sigma, n, alpha_level)
        # Take the maximum of kl and ks
        k = max(kl, ks)
        ks_values.append(k)
    
    df_ks = pd.DataFrame(data=ks_values, columns=[f"alpha={alpha_level}"], index=[f"n={n/86400} days" for n in ns])
    return df_ks

def main():
    # Load historical market data from CSV file
    csv_filename = Path("C:/Users/HP/Desktop/risk/historical_data1.csv")
    try:
        df = pd.read_csv(csv_filename, index_col='datetime', parse_dates=True)
        print("Data loaded successfully.")
        print(df)

        # Process data for levy stable fitting
        p = df['close'].to_numpy()
        log_returns = np.diff(np.log(p))

        # Fit levy stable distribution
        alpha, beta, mu, sigma = fit_levy_stable(log_returns)
        print(f"Fitted params: alpha: {alpha}, beta: {beta}, mu: {mu}, sigma: {sigma}")

        # Calibrate funding constants (k) to ensure VaR decays to zero
        df_ks = calibrate_k(alpha, beta, mu, sigma, NS, ALPHA)
        print('Calibrated Funding Constants (k):')
        print(df_ks)

    except FileNotFoundError:
        print(f"CSV file not found: {csv_filename}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
