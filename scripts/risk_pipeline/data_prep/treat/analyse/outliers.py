import pandas as pd
import numpy as np
from scipy import stats


def z_score(df, window):
    # Sort by time
    df = df.sort_values('time')
    df = df.set_index('time')

    # Get z score
    z = df.rolling(f'{window}s').apply(lambda x: np.abs(stats.zscore(x)[-1]))
    z.columns = ['z_score']

    # Reset index and return
    z.reset_index(inplace=True)
    return z


def remove_starting_rows(df, window):
    # Remove first `window` seconds of data
    df = df.set_index('time')
    df = df.loc[df.index[0] + pd.Timedelta(seconds=window):]
    return df.reset_index(inplace=True)


def interquartile_mean(df, window):
    '''
    Get mean of price price values between 25th and 75th percentile over
    the last `window` seconds
    '''

    # Sort by time
    df = df.sort_values('time')
    df = df.set_index('time')

    # Get rolling iqm
    rolling_iqm = df.rolling(f'{window}s').apply(
        lambda x: x[(x >= x.quantile(0.25)) & (x <= x.quantile(0.75))].mean(),
        raw=False
    )

    rolling_iqm.columns = ['rolling_iqm']

    # Reset index and return
    rolling_iqm.reset_index(inplace=True)
    return rolling_iqm
