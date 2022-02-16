import pandas as pd
import numpy as np
import os
import pystable
import typing as tp


def get_params() -> tp.Dict:
    '''
    Returns a `params` dict for parameters to use in statistical estimates.
    Generates metrics for 1h TWAP over last 30 days with VaR stats for next 7
    days.

    Outputs:
        [tp.Dict]
        points  [int]:          1 mo of data behind to estimate MLEs
        window  [int]:          1h TWAPs
        period  [int]:          60m periods [s]
        tolerance  [int]:       Tolerance within which `period` can
                                be inaccurate [minutes]
        alpha   List[float]:    alpha uncertainty in VaR calc
        n:      List[int]:      number of periods into the future over which
                                VaR is calculated
        data_start [int]:       start calculating metrics from these many days
                                ago
    '''
    return {
        "points": 90,
        "window": 60,
        "period": 10,
        "tolerance": 10,
        "alpha": [0.05, 0.01, 0.001, 0.0001],
        "n": [144, 1008, 2016, 4320],
        "data_start": 90
    }


def calc_vars(alpha: float, beta: float, sigma: float, mu: float, t: int,
              n: int, qtile: np.ndarray) -> np.ndarray:
    '''
    Calculates bracketed term:
        [e**(mu * n * t + sqrt(sig_sqrd * n * t) * Psi^{-1}(1 - alpha))]
    in Value at Risk (VaR) expressions for each alpha value in the `alphas`
    numpy array. SEE: https://oips.overlay.market/notes/note-4

    Inputs:
      alpha   [float]:       alpha parameter from fit
      beta    [float]:       beta parameter from fit
      sigma   [float]:       sigma parameter from fit
      mu      [float]:       mu parameter from fit
      t       [int]:         period
      n       [int]:
      alphas  [np.ndarray]:  array of alphas

    Outputs:
      [np.ndarray]:  Array of calculated values for each `alpha`

    '''

    sig = sigma * (t/alpha) ** (-1/alpha)
    mu = mu / t
    pow = mu * n * t + sig * (n * t / alpha) ** (1 / alpha) * np.array(qtile)
    return np.exp(pow) - 1


def get_stat(timestamp: int, sample: np.ndarray, p: tp.Dict
             ) -> pd.DataFrame:
    t = p["period"] * 60

    # mles
    rs = [np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)]

    # Gaussian Fit
    fit = {'alpha': 2, 'beta': 0, 'sigma': 1, 'mu': 0, 'parameterization': 1}

    # # Check fit validity
    fit_dist = pystable.create(fit['alpha'], fit['beta'], fit['sigma'],
                               fit['mu'], fit['parameterization'])

    pystable.fit(fit_dist, rs, len(rs))

    # VaRs for 5%, 1%, 0.1%, 0.01% alphas, n periods into the future
    alphas = np.array(p["alpha"])
    ns = np.array(p["n"])
    scale_dist = pystable.create(fit_dist.contents.alpha,
                                 fit_dist.contents.beta, 1, 0, 1)
    q = 1 - np.array(alphas)
    qtile = pystable.q(scale_dist, q, len(q))
    vars = [calc_vars(fit_dist.contents.alpha, fit_dist.contents.beta,
                      fit_dist.contents.sigma, fit_dist.contents.mu_1,
                      t, n, qtile) for n in ns]
    var_labels = [
        f'VaR alpha={alpha} n={n}'
        for n in ns
        for alpha in alphas
    ]

    data = np.concatenate(([timestamp, fit_dist.contents.alpha,
                            fit_dist.contents.beta, fit_dist.contents.sigma,
                            fit_dist.contents.mu_1], *vars), axis=None)

    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'alpha', 'beta', 'sigma', 'mu', *var_labels]
    return df


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    file_name = 'sample_r14.csv'
    data_path = os.path.join(base, file_name)
    sample = np.genfromtxt(data_path, delimiter=',')
    timestamp = 1644918170
    p = get_params()
    i = 0
    # while i < 3000:
    while True:
        results = get_stat(timestamp, sample, p)
        i += 1
        print(i)
        print(results)


if __name__ == '__main__':
    main()
