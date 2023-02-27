
def twap(df, time_col, price_col, tf):
    # TWAP and periodicity
    df.index = df[time_col]
    df.drop(time_col, axis=1, inplace=True)
    df = df[[price_col]]
    df = df.rolling(f'{tf}s', min_periods=1).mean()
    df.reset_index(inplace=True)
    return df


def set_periodicity(df, time_col, price_col, tf):
    df.index = df[time_col]
    df = df[[price_col]]
    df = df.resample(f"{tf}S").ohlc()
    df = df[price_col][[price_col]]
    df.reset_index(inplace=True)
    return df
