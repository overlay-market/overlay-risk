import pandas as pd
import numpy as np


BASE_DIR = "csv/univ3-data/"
FILENAME = "univ3-usdceth-03"
INFILE = f"{BASE_DIR}raw/{FILENAME}.csv"
OUTFILE = f"{BASE_DIR}clean/{FILENAME}-cleaned.csv"

DECIMALS_0 = 6
DECIMALS_1 = 18

SECS_PER_BLOCK = 15  # 15s per block
WINDOW_10M = 40  # 15s intervals
WINDOW_1H = 240  # 15s intervals


def get_quote(sqrt_price_x96: float, is_y_x: bool, amount_in: int) -> int:
    if is_y_x:
        return int(sqrt_price_x96)**2 * amount_in / (1 << 192)
    else:
        return (1 << 192) * amount_in / int(sqrt_price_x96)**2


def get_reserve(sqrt_price_x96: float, liquidity: float,
                is_x: bool) -> int:
    if is_x:
        return (int(liquidity) << 96) / int(sqrt_price_x96)
    else:
        return (int(liquidity) * int(sqrt_price_x96)) / (1 << 96)


def get_geo_avg(w: pd.Series, t: float) -> float:
    return w[0]*np.prod(w/w[0])**(1/t) if len(w) != 0 else np.nan


def include_prices(df: pd.DataFrame) -> pd.DataFrame:
    p0s = df['sqrtPriceX96'].apply(
        lambda x: get_quote(x, True, 10**(DECIMALS_0)))
    p1s = df['sqrtPriceX96'].apply(
        lambda x: get_quote(x, False, 10**(DECIMALS_1)))
    df['y/x'] = p0s
    df['x/y'] = p1s
    return df


def include_reserves(df: pd.DataFrame) -> pd.DataFrame:
    xs = df.filter(items=['sqrtPriceX96', 'liquidity']).apply(
        lambda args: get_reserve(*args, is_x=True), axis=1)
    ys = df.filter(items=['sqrtPriceX96', 'liquidity']).apply(
        lambda args: get_reserve(*args, is_x=False), axis=1)
    df['x'] = xs
    df['y'] = ys
    return df


def include_twaps(df: pd.DataFrame, t: int) -> pd.DataFrame:
    twap0s = df['y/x'].rolling(window=t)\
        .apply(lambda w: get_geo_avg(w, t), raw=True)
    twap1s = df['x/y'].rolling(window=t)\
        .apply(lambda w: get_geo_avg(w, t), raw=True)
    df[f'y/x twap {SECS_PER_BLOCK*t}s'] = twap0s
    df[f'x/y twap {SECS_PER_BLOCK*t}s'] = twap1s
    return df


def include_twars(df: pd.DataFrame, t: int) -> pd.DataFrame:
    twaxs = df['x'].rolling(window=t)\
        .apply(lambda w: get_geo_avg(w, t), raw=True)
    tways = df['y'].rolling(window=t)\
        .apply(lambda w: get_geo_avg(w, t), raw=True)
    df[f'x twar {SECS_PER_BLOCK*t}s'] = twaxs
    df[f'y twar {SECS_PER_BLOCK*t}s'] = tways
    return df


def main():
    """
    Assumes raw data from csv file taken from Swap event emit.

    Columns should include:
    (evt_block_time,sqrtPriceX96,tick,liquidity)
    """
    dtype = {
        'sqrtPriceX96': np.longdouble,
        'tick': int,
        'liquidity': np.longdouble
    }
    print(f"Reading csv {INFILE}...")
    df = pd.read_csv(INFILE, parse_dates=['evt_block_time'], dtype=dtype)
    print("df raw", df)

    df = include_prices(df)
    df = include_reserves(df)
    print("df with prices and reserves", df)

    # resample to 15 seconds
    df.set_index('evt_block_time', inplace=True)
    df = df.resample(f'{SECS_PER_BLOCK}s').mean()
    df.ffill(inplace=True)
    print("df resampled to 15s", df)

    df = include_twaps(df, WINDOW_10M)
    df = include_twaps(df, WINDOW_1H)
    print("df with twaps", df)

    df = include_twars(df, WINDOW_10M)
    print("df with twars", df)

    # resample to 1m
    df = df.resample('60s').mean()
    print("df resampled to 60s", df)

    print(f"Writing csv to {OUTFILE}...")
    df.to_csv(OUTFILE)


if __name__ == '__main__':
    main()
