from Historic_Crypto import HistoricalData  # Uses coinbase pro API
import argparse
import os
import sys
import pandas as pd
sys.path.insert(0, os.getcwd()+'/scripts/risk_pipeline')
from helpers import helpers  # noqa: E402


def get_params():
    """
    Get parameters from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pair', type=str,
        help='Name of the token pair. Ex ETH-USD'
    )
    parser.add_argument(
        '--periodicity', type=int,
        help='Cadence of data required from Coinbase (in secs). Ex: 300'
    )
    parser.add_argument(
        '--start_time', type=str,
        help='Start time for data pull. Ex: 2016-06-01-00-00'
    )
    parser.add_argument(
        '--end_time', type=str,
        help='End time for data pull. Ex: 2016-06-01-00-00'
    )

    args = parser.parse_args()
    return args.pair, args.periodicity, args.start_time, args.end_time


def get_file_name(pair, t, start, end):
    return f"{pair}_{start}_{end}_{t}secs.csv"


def get_data(pair, t, start, end):
    file_name = get_file_name(pair, t, start, end)
    file_path = os.getcwd() + '/scripts/risk_pipeline/outputs/data/'
    full_path = file_path + file_name
    if helpers.file_exists(full_path):
        print(f"Data {file_name} already exists")
        return pd.read_csv(full_path), full_path
    else:
        df = HistoricalData(pair, t, start, end).retrieve_data()
        helpers.csv(df, full_path)
        return df, full_path


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        pair, t, start, end = get_params()
        get_data(pair, t, start, end)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
