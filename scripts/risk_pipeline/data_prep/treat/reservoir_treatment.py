import argparse
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd()+'/scripts/risk_pipeline')
from data_prep.treat.analyse import outliers  # noqa: E402
from data_prep.treat.treatment import missing_value_treatment  # noqa: E402
from data_prep.treat.treatment import twap  # noqa: E402
from helpers import helpers  # noqa: E402
import visualizations.line_chart as lc  # noqa: E402


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
        '--final_periodicity', type=int,
        help='Cadence of data required for risk analysis (in secs). Ex: 600'
    )
    args = parser.parse_args()
    return args.filename, args.final_periodicity


def treatment(file_name, tf):
    file_path = os.getcwd() + '/scripts/risk_pipeline/outputs/data/'
    results_path = helpers.create_dir(file_name)

    # Read data
    if helpers.file_exists(file_path + file_name + '.csv'):
        df = pd.read_csv(file_path + file_name + '.csv', index_col=0)
        df.time = pd.to_datetime(df.time)
    else:
        print(f"File {file_name} doesn't exist at {file_path}")
        return

    # Plot initial price data
    title = "Price feed pull from reservoir API"
    chartname = f"{file_name}_raw"
    xcol = 'time'
    ycol = 'close'
    fig = lc.LineChart(df, title, xcol, ycol).create_chart()
    fig.write_html(f"{results_path}/{chartname}.html")

    # Get z scores to treat outliers
    window = 86400*5
    z_df = outliers.z_score(df, window)
    iqm_df = outliers.interquartile_mean(df, window)
    treat_df = df.merge(z_df, on='time').merge(iqm_df, on='time')

    # Replace close price with rolling iqm if z score is above threshold
    treat_df['treat_close'] = np.where(
        treat_df['z_score'] > 1.5,
        treat_df['rolling_iqm'],
        treat_df['close']
    )

    # Plot price after outlier treatment
    title = "Outliers treated"
    chartname = f"{file_name}_outliers_treated"
    xcol = 'time'
    ycol = 'treat_close'
    fig = lc.LineChart(treat_df, title, xcol, ycol).create_chart()
    fig.write_html(f"{results_path}/{chartname}.html")


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        filename, tf = get_params()
        treatment(filename, tf)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
