import pandas as pd
import typing as tp
import argparse


def get_params() -> tp.Dict:

    parser = argparse.ArgumentParser(
        description='Convert price cumulatives to TWAP'
    )
    parser.add_argument(
        '--name', type=str,
        help='Name of CSV file in `csv` directory (without ".csv")'
    )
    parser.add_argument(
        '--decimal1', type=int,
        help='Number of decimals of token1'
    )
    parser.add_argument(
        '--decimal2', type=int,
        help='Number of decimals of token2'
    )
    parser.add_argument(
        '--length', type=int,
        help='TWAP length in minutes (shoud be multiple of 10)'
    )

    args = parser.parse_args()

    return {
        "name": args.name,
        "decimal1": args.decimal1,
        "decimal2": args.decimal2,
        "length": args.length
    }


def get_subset(df):
    df = df[df['_field'] == 'tick_cumulative']
    df = df[['_time', '_value', 'id']]
    return df


def get_data(name):
    return pd.read_csv(f'csv/{name}.csv')


def main():
    params = get_params()
    name = params['name']
    decimal1 = params['decimal1']
    decimal2 = params['decimal2']
    lag = int(params['length']/10)

    df = get_data(params['name'])
    df = get_subset(df)
    df.loc[:, 'lag_time'] = df.loc[:, '_time'].shift(lag)
    df.loc[:, 'lag_value'] = df.loc[:, '_value'].shift(lag)
    df.loc[:, 'dt'] = (pd.to_datetime(df._time) - pd.to_datetime(df.lag_time))\
        .dt.total_seconds()
    df.loc[:, 'dp'] = (df['_value'] - df['lag_value'])
    df.loc[:, 'log_price'] = df.loc[:, 'dp']/df.loc[:, 'dt']
    df.loc[:, 'price'] = (1.0001**df.loc[:, 'log_price'])\
        / (10**(decimal1 - decimal2))
    df.loc[:, 'inv_price'] = 1/df.loc[:, 'price']
    df.to_csv(f'csv/{name}_twap.csv')


if __name__ == '__main__':
    main()
