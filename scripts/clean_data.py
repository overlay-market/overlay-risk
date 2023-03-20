import pandas as pd
import argparse


def get_params() -> str:
    '''
    Returns the file name with data
    '''
    parser = argparse.ArgumentParser(
        description='Transform data to any periodicty'
    )

    parser.add_argument(
        '--source', type=str,
        help='CSV file with data'
    )

    parser.add_argument(
        '--periodicity', type=str,
        help='Refer: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases'  # noqa
    )

    parser.add_argument(
        '--timestamp_column', type=str,
        help='Name of column with timestamp in data file'
    )
    args = parser.parse_args()

    return {
        "source": args.source,
        "periodicity": args.periodicity,
        "timestamp_column": args.timestamp_column
    }


def file_location() -> str:
    return 'csv/'


def main():
    p = get_params()
    fl = file_location()

    file_path = fl + p['source']

    df = pd.read_csv(file_path)
    df[p['timestamp_column']] = pd.to_datetime(df[p['timestamp_column']])
    df.set_index(p['timestamp_column'], inplace=True)
    df = df.resample(p['periodicity']).mean()
    df.ffill(inplace=True)

    final_name =\
        fl + p['source'].replace('.csv', '') + '_' + p['periodicity'] + '.csv'
    df.to_csv(final_name)


if __name__ == '__main__':
    main()
