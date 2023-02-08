import argparse
import pandas as pd
import os
import sys
sys.path.insert(0, os.getcwd()+'/scripts/risk_pipeline')
from data_prep.treat.analyse import missing_values  # noqa
from data_prep.treat.treatment import missing_value_treatment, twap  # noqa
from helpers import helpers  # noqa
from helpers import visualizations  # noqa


def get_params():
    """
    Get parameters from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename', type=str,
        help='Name of the file with price data'
    )
    parser.add_argument(
        '--pull_periodicity', type=int,
        help='Cadence of data required from Coinbase (in secs). Ex: 300'
    )
    parser.add_argument(
        '--final_periodicity', type=int,
        help='Cadence of data required for risk analysis (in secs). Ex: 600'
    )
    args = parser.parse_args()
    return args.filename, args.pull_periodicity, args.final_periodicity


def treatment(file_name, t, tf):
    file_path = os.getcwd() + '/scripts/risk_pipeline/outputs/data/'
    results_path = helpers.create_dir(file_name)

    # Read data
    if helpers.file_exists(file_path + file_name + '.csv'):
        df = pd.read_csv(file_path + file_name + '.csv')
    else:
        print(f"File {file_name} doesn't exist at {file_path}")
        return

    # Plot initial price data
    title = "Price feed pull from coinbase"
    chartname = f"{file_name}_raw"
    xcol = 'time'
    ycol = 'close'
    visualizations.time_series(df, title, chartname, xcol, ycol, results_path)

    # Get report on missing values
    df_m = missing_values.missing_candlesticks(
        df, t, 'time', 'close', file_name, results_path)

    # Treat missing values
    df_f = missing_value_treatment.forward_fill(df_m, 'close')

    # Get TWAP and convert to desired periodicity
    df_f = twap.twap(df_f, 'time', 'close', tf)
    df_f = twap.set_periodicity(df, 'time', 'close', tf)

    # Plot final price data (visualizations)
    title = "Price feed pull from coinbase - Treated"
    chartname = f"{file_name}_final"
    xcol = 'time'
    ycol = 'close'
    visualizations.time_series(df_f, title, chartname, xcol,
                               ycol, results_path)

    # Save data
    final_file_name = file_name + "_treated.csv"
    file_path = os.getcwd() + '/scripts/risk_pipeline/outputs/data/'
    helpers.csv(df_f, file_path + final_file_name)
    return df_f


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        filename, t, tf = get_params()
        treatment(filename, t, tf)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")