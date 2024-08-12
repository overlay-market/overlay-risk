import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import integrate
from scipy.stats import levy_stable

# Constants
ALPHAS = np.array([0.01, 0.025, 0.05, 0.075, 0.1])
Q0S = np.array([0.005 * i for i in range(1, 11, 1)])

def fit_stable(data):
    def negative_log_likelihood(params, data):
        return -np.sum(levy_stable.logpdf(data, *params))

    initial_params = [1.5, 0, np.mean(data), np.std(data)]
    bounds = [(0.1, 2), (-1, 1), (None, None), (0, None)]
    result = minimize(negative_log_likelihood, initial_params, args=(data,), bounds=bounds)
    return result.x

def rescale_params(alpha, beta, mu, sigma, time_factor):
    mu_rescaled = mu * time_factor
    sigma_rescaled = sigma * time_factor**(1/alpha)
    return alpha, beta, mu_rescaled, sigma_rescaled

def delta_long(a, b, mu, sig, v, alphas):
    dist = levy_stable(a, b, loc=mu*v, scale=sig*(v**(1/a)))
    qs_long = dist.ppf(1 - alphas)
    return qs_long / 2.0

def delta_short(a, b, mu, sig, v, alphas):
    dist = levy_stable(a, b, loc=mu*v, scale=sig*(v**(1/a)))
    qs_short = dist.ppf(alphas)
    return - qs_short / 2.0

def delta(a, b, mu, sig, v, alphas):
    d_l = delta_long(a, b, mu, sig, v, alphas)
    d_s = delta_short(a, b, mu, sig, v, alphas)
    return np.maximum(d_l, d_s)

def lmbda_long(a, b, mu, sig, g_inv, v, alpha, q0s):
    alphas = np.array([alpha])
    cp = np.exp(g_inv) - 1
    delta_l = delta_long(a, b, mu, sig, v, alphas)
    dist = levy_stable(a, b, loc=mu*v - 2*delta_l, scale=sig*(v**(1/a)))

    def integrand(y):
        return dist.pdf(y) * np.exp(y)
    
    numerator_l, _ = integrate.quad(integrand, 0, g_inv)
    denominator_l = alpha - (1+cp)*(1-dist.cdf(g_inv))
    rho_l = numerator_l / denominator_l
    return np.log(rho_l) / (2*q0s)

def lmbda_short(a, b, mu, sig, v, alpha, q0s):
    alphas = np.array([alpha])
    delta_s = delta_short(a, b, mu, sig, v, alphas)
    dist = levy_stable(a, b, loc=mu*v + 2*delta_s, scale=sig*(v**(1/a)))

    def integrand(y):
        return dist.pdf(y) * np.exp(y)

    numerator_s = alpha
    denominator_s, _ = integrate.quad(integrand, -np.inf, 0)
    rho_s = numerator_s / denominator_s
    return np.log(rho_s) / (2*q0s)

def lmbda(a, b, mu, sig, v, g_inv, alpha, q0s):
    lmbda_l = lmbda_long(a, b, mu, sig, g_inv, v, alpha, q0s)
    lmbda_s = lmbda_short(a, b, mu, sig, v, alpha, q0s)
    return np.maximum(lmbda_l, lmbda_s)

def analyze_data(csv_file_path, t, cp, st, alpha_level):
    try:
        df = pd.read_csv(csv_file_path, index_col='datetime', parse_dates=True)
        print("Data loaded successfully.")
        print(df)

        p = df['close'].to_numpy()
        log_close = np.diff(np.log(p))

        params = fit_stable(log_close)
        a, b, mu, sig = params
        print(f'Fit params: alpha: {a}, beta: {b}, mu: {mu}, sigma: {sig}')

        time_factor = 1 / t
        a, b, mu, sig = rescale_params(a, b, mu, sig, time_factor)
        print(f'Rescaled params (1/t = {time_factor}): alpha: {a}, beta: {b}, mu: {mu}, sigma: {sig}')

        g_inv = np.log(1 + cp)

        deltas = delta(a, b, mu, sig, st, ALPHAS)
        df_deltas = pd.DataFrame(data={'alpha': ALPHAS, 'delta': deltas})
        print('Deltas:', df_deltas)

        ls = []
        for alpha in ALPHAS:
            lambdas = lmbda(a, b, mu, sig, st, g_inv, alpha, Q0S)
            ls.append(lambdas)

        df_ls = pd.DataFrame(data=ls, columns=[f"q0={q0}" for q0 in Q0S], index=[f"alpha={alpha}" for alpha in ALPHAS])
        print('Lambdas:', df_ls)

        return df_deltas, df_ls

    except FileNotFoundError:
        print(f"CSV file not found: {csv_file_path}")
    except ValueError as e:
        print(f"Error: {e}")