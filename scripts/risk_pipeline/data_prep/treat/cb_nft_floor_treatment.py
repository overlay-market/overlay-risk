import argparse
import pandas as pd
import os
import sys
sys.path.insert(0, os.getcwd()+'/scripts/risk_pipeline')
from data_prep.treat.treatment import twap  # noqa: E402
from data_prep.treat.analyse import missing_values  # noqa: E402
from data_prep.treat.treatment import missing_value_treatment  # noqa: E402
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
    '''
    Convert NFT floor price data from Coinbase algo to desired periodicity
    '''

    file_path = os.getcwd() + '/scripts/risk_pipeline/outputs/data/'
    results_path = helpers.create_dir(file_name)

    # Read data
    if helpers.file_exists(file_path + file_name + '.csv'):
        df = pd.read_csv(file_path + file_name + '.csv', index_col=0)
        df['time'] = pd.to_datetime(df.unix_timestamp, unit='s')
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

    # Convert to desired periodicity
    df = twap.set_periodicity(df, 'time', 'close', tf)

    # Nulls likely generated during `set_periodicity`.
    # These are not neccessarily "missing values". But worth checking.
    m_df, null_report = missing_values.missing_candlesticks(
        df, tf, 'time', 'close')
    null_report.to_csv(f"{results_path}/{file_name}_missing_values.csv")

    # Treat missing values
    f_df = missing_value_treatment.forward_fill(m_df, 'close')

    # Plot final price data
    title = "Price - final"
    chartname = f"{file_name}_final"
    xcol = 'time'
    ycol = 'close'
    fig = lc.LineChart(f_df, title, xcol, ycol).create_chart()
    fig.write_html(f"{results_path}/{chartname}.html")

    # Save data
    final_file_name = file_name + f'_{tf}_secs_treated.csv'
    file_path = os.getcwd() + '/scripts/risk_pipeline/outputs/data/'
    helpers.csv(f_df, file_path + final_file_name)
    return f_df


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        filename, tf = get_params()
        treatment(filename, tf)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
