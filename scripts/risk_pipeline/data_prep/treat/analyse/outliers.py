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
    df = df.join(z)

    # Reset index and return
    df.reset_index(inplace=True)
    return df


def remove_starting_rows(df, window):
    # Remove first `window` seconds of data
    df = df.set_index('time')
    df = df.loc[df.index[0] + pd.Timedelta(seconds=window):]
    return df.reset_index(inplace=True)

