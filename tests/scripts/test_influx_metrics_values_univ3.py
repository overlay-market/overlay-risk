from datetime import datetime
import pytest
import os
from influxdb_client import InfluxDBClient
import typing as tp
import pandas as pd

START = 1634301023
STOP = 1634733023
ID = "UniswapV3: WETH / DAI .3%"


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "url": os.getenv("INFLUXDB_URL")
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(
        url=config['url'],
        token=config['token'],
        debug=False
    )


def get_metrics_data(bucket):
    config = get_config()
    client = create_client(config)
    query_api = client.query_api()

    query = f'''
        from(bucket:"{bucket}")
            |> range(start: {START}, stop: {STOP})
    '''
    print(f'Fetching data from {bucket}')
    df = query_api.query_data_frame(query=query, org=config['org'])

    return df


def get_cumulatives_data(bucket):
    config = get_config()
    client = create_client(config)
    query_api = client.query_api()
    points = 30
    secs = points*24*60*60

    query = f'''
        from(bucket:"{bucket}")
            |> filter(fn: (r) => r["_field"] == "tick_cumulative")
            |> range(start: {START-secs}, stop: {STOP})
    '''
    print(f'Fetching data from {bucket}')
    df = query_api.query_data_frame(query=query, org=config['org'])

    return df


def get_twap():
    cumus = get_cumulatives_data('ovl_univ3_1h')
    breakpoint()
    filter = cumus['_time'] == "2021-10-15 12:35:50+00:00"
    cumus = cumus[filter]


def test_check():
    metrics = get_metrics_data('ovl_metrics_univ3')
    cumus = get_cumulatives_data('ovl_univ3_1h')
    get_twap()

    assert 1 == 1