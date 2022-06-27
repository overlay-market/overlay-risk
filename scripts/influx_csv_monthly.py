import pandas as pd
import os
import argparse
import typing as tp
import logging

import calendar
from influxdb_client import InfluxDBClient


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'],
                          debug=False)


def get_params() -> tp.Dict:
    source = "ovl_sushi"  # Default to ovl_sushi InfluxDB bucket

    parser = argparse.ArgumentParser(
        description='Fetch data from InfluxDB bucket and save to local csv'
    )
    parser.add_argument(
        '--bucket', type=str,
        help='name of the influx bucket to query'
    )

    args = parser.parse_args()
    if args.bucket:
        source = args.bucket

    return {
        "source": source
    }


def get_data(query_api, cfg: tp.Dict,
             p: tp.Dict, month: tp.Tuple) -> pd.DataFrame:
    bucket = p['source']
    org = cfg['org']

    max_date = calendar.monthrange(month[0], month[1])[1]
    start_date = f'{month[0]}-{str(month[1]).zfill(2)}-01T00:00:00.00Z'
    end_date = f'{month[0]}-{str(month[1]).zfill(2)}-{max_date}T23:59:59.00Z'
    print(f'Fetching data from {bucket} ...')
    query = f'''
        from(bucket:"{bucket}")
        |> range(start: {start_date}, stop: {end_date})
        |> filter(fn: (r) => r["id"] == "UniswapV3: WETH / USDC .3%")
    '''
    df = query_api.query_data_frame(query=query, org=org)
    if type(df) == list:
        df = pd.concat(df, ignore_index=True)
    return df


def create_csv(df: pd.DataFrame, ym: str):
    if not os.path.exists('csv'):
        os.makedirs('csv')
    # df.to_csv(f"csv/data-{int(datetime.now().timestamp())}.csv", index=False)
    df.to_csv(f"csv/WETH-USDC-{ym}.csv", index=False)


def main():
    """
    Fetch from an influx bucket and write data locally to csv
    """
    config = get_config()
    params = get_params()
    client = create_client(config)
    query_api = client.query_api()

    try:
        for month in [(2021, 9),
                      (2021, 10),
                      (2021, 11),
                      (2021, 12),
                      (2022, 1),
                      (2022, 2),
                      (2022, 3),
                      (2022, 4),
                      (2022, 5),
                      (2022, 6)]:
            print(month)
            df = get_data(query_api, config, params, month)
            create_csv(df, f'{str(month[0])}{str(month[1]).zfill(2)}')
    except Exception as e:
        print("Failed to write data from influx to csv")
        logging.exception(e)

    client.close()


if __name__ == '__main__':
    main()
