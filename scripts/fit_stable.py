import pystable
import pandas as pd
import numpy as np


BASE_DIR = "csv/coingecko-data/"
FILENAME = "coingecko_fwbeth_04012021_08232021"
INFILE = f"{BASE_DIR}{FILENAME}.csv"
OUTFILE = f"{BASE_DIR}fit/{FILENAME}-fit.csv"

PRICE_COLUMN = 'c'
T = 240  # 1h candle size on datafile
SECS_PER_BLOCK = 15


def gaussian():
    return pystable.create(alpha=2.0, beta=0.0, mu=0.0,
                           sigma=1.0, parameterization=1)


def main():
    """
    Fits input csv timeseries data with pystable
    """
    print(f'Analyzing file at {INFILE}')
    df = pd.read_csv(INFILE)
    p = df[PRICE_COLUMN].to_numpy()
    log_close = [np.log(p[i]/p[i-1]) for i in range(1, len(p))]

    dst = gaussian()  # use gaussian as init dist to fit from
    pystable.fit(dst, log_close, len(log_close))
    print(
        f'''
        fit params: alpha: {dst.contents.alpha}, beta: {dst.contents.beta},
        mu: {dst.contents.mu_1}, sigma: {dst.contents.sigma}
        '''
    )
    candle_size = T * SECS_PER_BLOCK
    df_metrics = pd.DataFrame([candle_size, dst.contents.alpha,
                               dst.contents.beta, dst.contents.mu_1,
                               dst.contents.sigma]).T
    df_metrics.columns = ['candle_size', 'alpha', 'beta', 'mu', 'sigma']
    print('df_metrics', df_metrics)
    df_metrics.to_csv(OUTFILE)


if __name__ == '__main__':
    main()
