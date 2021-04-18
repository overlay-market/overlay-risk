import pandas as pd
import numpy as np
import os
import json
import typing as tp

from brownie import network, Contract
from datetime import datetime
from scipy.stats import norm

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "bucket": os.getenv('INFLUXDB_BUCKET', 'ovl_sushi'),
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'], debug=False)


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-sushi", "ingest-data-frame")
    return point_settings


def PAIR(addr: str) -> Contract:
    # UniswapV2Pair contract
    return Contract.from_explorer(addr)


def get_params() -> tp.Dict:
    return {
        "period": 1800,  # in seconds
    }


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


# Only write to influxdb every `period` seconds
def can_write(pair, read_api, q: tp.Dict, p: tp.Dict) -> bool:
    last_pc = pair.blockTimestampLast()

    # TODO: read_api read last datapoint written to influx
    last_write = read_api.read(q["id"])
    return (last_pc - p["period"]) > last_write


# Get last price cumulatives for timeseries data
def get_prices(pair, q: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    # uniswapv2 pair cumulative data views
    p0c = pair.price0CumulativeLast();
    p1c = pair.price1CumulativeLast();
    timestamp = pair.blockTimestampLast()

    # Put into a dataframe
    data = np.array([timestamp, p0c, p1c])
    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'price0Cumulative', 'price1Cumulative']
    return df


def main():
    print(f"You are using the '{network.show_active()}' network")
    config = get_config()
    params = get_params()
    quotes = get_quotes()
    client = create_client(config)
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    # TODO: implement read_api init properly
    read_api = client.read_api()

    for q in quotes:
        print('id', q['id'])
        if not can_write(pair, read_api, q, params):
            continue

        try:
            _, prices = get_prices(pair, q, params)
            print('prices', prices)

            stats.to_csv(
                f"csv/pc/{q['id']}-{int(datetime.now().timestamp())}.csv",
                index=False,
            )
            point = Point("mem")\
                .tag("id", q['id'])\
                .time(
                    datetime.fromtimestamp(float(prices['timestamp'])),
                    WritePrecision.NS
                )

            for col in prices.columns:
                if col != 'timestamp':
                    point = point.field(col, float(prices[col]))

            print(f"Writing {q['id']} to api ...")
            write_api.write(config['bucket'], config['org'], point)
        except:
            print("Failed to write quote stats to influx")

    client.close()
