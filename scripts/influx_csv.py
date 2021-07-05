import pandas as pd
import os
import argparse
import typing as tp
import logging

from datetime import datetime
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
    points = 30  # Default to 30d in past for influx start range
    source = "ovl_sushi"  # Default to ovl_sushi InfluxDB bucket

    parser = argparse.ArgumentParser(
        description='Fetch data from InfluxDB bucket and save to local csv'
    )
    parser.add_argument(
        '--bucket', type=str,
        help='name of the influx bucket to query'
    )
    parser.add_argument(
        '--points', type=int,
        help='number of days in the past to use as start range of influx query'
    )

    args = parser.parse_args()
    if args.points:
        points = args.points
    if args.bucket:
        source = args.bucket

    return {
        "points": points,
        "source": source
    }


def get_data(query_api, cfg: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    points = p['points']
    bucket = p['source']
    org = cfg['org']

    print(f'Fetching data from {bucket} for the last {points} days ...')
    query = f'''
        from(bucket:"{bucket}") |> range(start: -{points}d)
    '''
    df = query_api.query_data_frame(query=query, org=org)
    if type(df) == list:
        df = pd.concat(df, ignore_index=True)
    return df


def create_csv(df: pd.DataFrame):
    if not os.path.exists('csv'):
        os.makedirs('csv')
    df.to_csv(f"csv/data-{int(datetime.now().timestamp())}.csv", index=False)


def main():
    """
    Fetch from an influx bucket and write data locally to csv
    """
    config = get_config()
    params = get_params()
    client = create_client(config)
    query_api = client.query_api()

    try:
        df = get_data(query_api, config, params)
        create_csv(df)
    except Exception as e:
        print("Failed to write data from influx to csv")
        logging.exception(e)

    client.close()


if __name__ == '__main__':
    main()
