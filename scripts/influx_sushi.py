import pandas as pd
import numpy as np
import os
import json
import typing as tp
import logging

from brownie import network, Contract
from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "bucket": os.getenv('INFLUXDB_BUCKET', 'ovl_sushi_dev'),
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'],
                          debug=False)


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-sushi", "ingest-data-frame")
    return point_settings


def PAIR(addr: str) -> Contract:
    # UniswapV2Pair contract
    return Contract.from_explorer(addr)


def get_quote_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    qp = 'constants/quotes.json'
    return os.path.join(base, qp)


def get_quotes() -> tp.List:
    quotes = []
    p = get_quote_path()
    with open(p) as f:
        data = json.load(f)
        quotes = data.get('quotes', [])
    return quotes


# Get last price cumulatives for timeseries data
def get_prices(q: tp.Dict) -> pd.DataFrame:
    pair = PAIR(q['pair'])

    # uniswapv2 pair cumulative data views
    p0c = pair.price0CumulativeLast()
    p1c = pair.price1CumulativeLast()
    _, _, timestamp = pair.getReserves()

    # Put into a dataframe
    data = np.array([timestamp, p0c, p1c])
    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'price0Cumulative', 'price1Cumulative']
    return df


def main():
    print(f"You are using the '{network.show_active()}' network")
    config = get_config()
    quotes = get_quotes()
    client = create_client(config)
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    for q in quotes:
        try:
            print('id', q['id'])
            prices = get_prices(q)
            print('prices', prices)

            point = Point("mem")\
                .tag("id", q['id'])\
                .tag('token0_name', q['token0_name'])\
                .tag('token1_name', q['token1_name'])\
                .time(
                    datetime.utcfromtimestamp(float(prices['timestamp'])),
                    WritePrecision.NS
                )

            for col in prices.columns:
                if col != 'timestamp':
                    point = point.field(col, float(prices[col]))

            print(f"Writing {q['id']} to api ...")
            write_api.write(config['bucket'], config['org'], point)
        except Exception as e:
            print("Failed to write quote prices to influx")
            logging.exception(e)

    client.close()
