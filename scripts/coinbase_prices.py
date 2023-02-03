import pandas as pd
from Historic_Crypto import HistoricalData  # Uses coinbase pro API
import argparse
import plotly.express as px
from datetime import timedelta
import numpy as np


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
        '--pull_periodicity', type=int,
        help='Cadence of data required from Coinbase (in secs). Ex: 300'
    )
    parser.add_argument(
        '--final_periodicity', type=int,
        help='Cadence of data required for risk analysis (in secs). Ex: 600'
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
    return args.pair, args.pull_periodicity, args.final_periodicity,\
        args.start_time, args.end_time


def chart(df, title, label, t, pair, start, end):
    # Plot feed
    fig = px.line(df[['time', 'close']], x='time', y='close')
    fig.update_layout(title=title)
    fig.update_layout(xaxis_title="Time", yaxis_title="Price")
    fig.write_html(f"csv/metrics/{pair}-{label}-{t}-{start}-{end}.html")


def main():
    pair, t, tf, start, end = get_params()
    # Pull and save data
    # TODO: Check if file exists
    df = HistoricalData(pair, t, start, end).retrieve_data()
    df.to_csv(f"csv/{pair}-SPOT-{t}-{start}-{end}.csv")
    df = pd.read_csv(f"csv/{pair}-SPOT-{t}-{start}-{end}.csv",
                     index_col=0)
    df.reset_index(inplace=True)
    df['time'] = pd.to_datetime(df['time'])

    # Plot feed
    chart(df, 'Price feed pull from Coinbase', 'initial', t, pair, start, end)

    # Create df such that there is a null row for missing timestamps
    start_date = df.time.min()
    end_date = df.time.max()

    timestamp_list = []
    current = start_date
    while current < end_date:
        timestamp_list.append(current)
        current += timedelta(minutes=t/60)

    time_df = pd.DataFrame({'time': np.array(timestamp_list)})
    df = time_df.merge(df, how='left', on='time')

    # Report info on missing values
    df_nulls = df.join(df.close.notnull().astype(int).cumsum(), rsuffix='_grp')
    null_report = df_nulls.groupby('close_grp').agg(
        {'time': ['min', 'max', 'count'],
         'close': ['min', 'max']})
    null_report = null_report[null_report.time['count'] > 1]
    null_report.reset_index(inplace=True)

    for i in range(null_report.shape[0]):
        idx = df[df.time == null_report.time['max'].iloc[i]].index + 1
        null_report.at[i, ('close', 'max')] = df.iloc[idx, :]['close']
    null_report.columns = null_report.columns.get_level_values(1)
    null_report.drop('', axis=1, inplace=True)
    null_report.columns = ['start_time', 'end_time', 'null_row_count',
                           'pre_null_price', 'post_null_price']
    null_report.start_time += pd.Timedelta(minutes=t/60)
    null_report.null_row_count -= 1
    null_report['percentage_change'] =\
        abs((null_report.post_null_price/null_report.pre_null_price) - 1) * 100
    null_report.sort_values('percentage_change', ascending=False, inplace=True)
    null_report.to_csv(f'csv/metrics/{pair}-null_report-{t}-{start}-{end}.csv')

    # Ffill nulls by default
    # But reconsider based on insights from null report
    # TODO: Print warnings
    df.close.ffill(inplace=True)

    # TWAP and periodicity
    df.index = df.time
    df.drop('time', axis=1, inplace=True)
    df = df[['close']]
    df = df.rolling(f'{tf}s', min_periods=1).mean()
    final_df = df.resample(f"{tf}S").ohlc()
    final_df = final_df.close[['close']]
    final_df.reset_index(inplace=True)
    final_df.to_csv(f'csv/{pair}-twap-{t}-{start}-{end}.csv')
    chart(final_df, 'Price feed pull from Coinbase', 'final',
          tf, pair, start, end)


if __name__ == '__main__':
    main()
