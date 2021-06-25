import pandas as pd
import os
import typing as tp
import logging

from datetime import datetime
from influxdb_client import InfluxDBClient


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "source": os.getenv('INFLUXDB_SOURCE', "ovl_sushi"),
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'],
                          debug=False)


def get_params() -> tp.Dict:
    return {
        "points": 30,
    }


def get_data(query_api, cfg: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    points = p['points']
    bucket = cfg['source']
    org = cfg['org']

    print(f'Fetching data from {bucket} ...')
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
