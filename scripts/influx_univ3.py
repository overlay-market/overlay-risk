import pandas as pd
import numpy as np
import os
import json
import typing as tp
import logging
import time
import math
import asyncio
import brownie
import sys
import requests

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor 
from concurrent.futures import ProcessPoolExecutor 
import concurrent.futures

from brownie import web3, chain, network, Contract
from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings

block_subgraph = 'https://api.thegraph.com/subgraphs/name/decentraland/blocks-ethereum-mainnet'

def get_b_q (timestamp: int) -> str:
    return """query {
                blocks(
                    first: 1,
                    orderBy: timestamp,
                    orderDirection: desc,
                    where: { timestamp_lt: %s }
                ) {
                    timestamp
                    number
                }
            } """ % (str(timestamp))

def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "bucket": os.getenv('INFLUXDB_BUCKET', 'ovl_sushi_dev'),
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(
        url=config['url'], 
        token=config['token'],
        debug=False
    )


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-sushi", "ingest-data-frame")
    return point_settings


def POOL(addr: str) -> Contract:
    # UniswapV3Pool contract
    return Contract.from_explorer(addr)


def get_quote_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    qp = 'constants/univ3_quotes.json'
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
    pool = POOL(q['pair'])

    # uniswapv2 pair cumulative data views
    p0c = pair.price0CumulativeLast()
    p1c = pair.price1CumulativeLast()
    _, _, timestamp = pair.getReserves()

    # Put into a dataframe
    data = np.array([timestamp, p0c, p1c])
    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'price0Cumulative', 'price1Cumulative']
    return df

def get_b_t (timestamp: int) -> tp.Tuple:

    q = { 'query': get_b_q(timestamp) }

    result = json.loads(
        requests.post( block_subgraph, json=q).text
    )['data']['blocks'][0]

    return ( int(result['number']), int(result['timestamp']) )


def get_calls(pair: Contract, t_from: int, t_to: int, t_period: int) -> tp.List:

    calls = []

    while t_to < t_from:
        calls.append((pair, t_from, t_period)) 
        t_from -= t_period

    return calls

def read_tick_set( args: tp.Tuple ) -> tp.Tuple:

    ( pair, t_at, t_p ) = args

    ( b, b_t ) = get_b_t(t_at)

    ( ticks_cum, liqs_cum ) = pair.observe([0, t_p], block_identifier=b)

    return ( b_t, ticks_cum, liqs_cum ) 

def main():

    print(f"You are using the '{network.show_active()}' network")

    config = get_config()
    client = create_client(config)

    quotes = get_quotes()
    pair = POOL(quotes[0]['pair'])

    t_c = math.floor(time.time()) - 600

    calls = get_calls(pair, t_c, t_c - 86400, 600)

    columns = ['timestamp', 'tick_cumulatives', 'liquidity_per_second_cumulatives']

    p_df = pd.DataFrame(columns=columns)
    with ThreadPoolExecutor() as executor:
        for i in executor.map(read_tick_set, calls):
            print(i)
            i_df = pd.DataFrame([i], columns=columns)
            # print(i_df)
            p_df = p_df.append(i_df, ignore_index=True)

    print(p_df)



