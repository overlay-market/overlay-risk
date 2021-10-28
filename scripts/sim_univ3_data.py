import pandas as pd
import numpy as np
import typing as tp
import json


START_TIMESTAMP = 1635206696
SECS_PER_BLOCK = 15
DECIMALS_Y = 18
DECIMALS_X = 18

BASE_DIR = "csv/coingecko-data/"
FILENAME = "coingecko_shibeth_04012021_10262021-mc100"

# TODO: SIM_COLS loop
SIM_COLS = ["sim-1", "sim-8", "sim-17", "sim-29", "sim-37", "sim-48",
            "sim-53", "sim-57", "sim-60", "sim-62", "sim-63", "sim-84",
            "sim-85", "sim-88"]
SIM_COL = "sim-88"

INDIR = f"{BASE_DIR}sims/shibeth/"
INFILE = f"{INDIR}{FILENAME}.csv"  # datafile
OUTDIR = f"{BASE_DIR}sims/shibeth/"
OUTFILE = f"{OUTDIR}{FILENAME}-{SIM_COL}_raw_uni.json"

TICK_BASE = 1.0001
TEN_MIN = int(600/SECS_PER_BLOCK)
ONE_HR = int(3600/SECS_PER_BLOCK)


def get_tick_from_price(p: float) -> float:
    return np.log(p) / np.log(TICK_BASE)


def get_price_from_tick(t: float) -> float:
    return np.power(TICK_BASE, t)


def format_for_json(tks: pd.Series, tkcs: pd.Series) -> tp.List:
    obs = [{
        "observation": [
            time.timestamp(),
            int(tkcs[time]),
            0,  # TODO: liquidity
            True
        ],
        "shim": [
            time.timestamp(),
            0,  # TODO: liquidity
            int(tks[time]),
            i+1,
        ]
    } for i, time in enumerate(tkcs.index)]

    # ret reversed list given typical raw_uni formatting
    obs.reverse()
    return obs


def main():
    df = pd.read_csv(INFILE)

    # Create datetime index
    df['unix'] = df['timestamp'] + START_TIMESTAMP
    df['time'] = pd.to_datetime(df['unix'], unit='s')
    df.set_index('time', inplace=True)

    df = df.resample(f'{SECS_PER_BLOCK}s').mean()
    df.ffill(inplace=True)

    # create tick cumulatives
    ticks = get_tick_from_price(df[SIM_COL] * 10**(DECIMALS_Y-DECIMALS_X))
    tick_cums = (ticks*SECS_PER_BLOCK).cumsum()

    obs_json = format_for_json(ticks, tick_cums)

    # write to json file
    with open(OUTFILE, 'w') as f:
        json.dump(obs_json, f)


if __name__ == '__main__':
    main()
