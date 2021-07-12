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
import datetime

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
        "bucket": os.getenv('INFLUXDB_BUCKET', 'ovl_univ3_james'),
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
    point_settings.add_default_tag("influx-univ3-james", "ingest-data-frame")
    return point_settings

def POOL(addr: str, abi) -> Contract:
    # UniswapV3Pool contract
    return Contract.from_abi("unipool", addr, abi)

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

def get_uni_abi_path () -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'constants/univ3.abi.json')

def get_uni_abi () -> tp.Dict:
    with open(get_uni_abi_path()) as f:
        data = json.load(f)
    return data


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

def find_start(api, quote, config) -> int:

    r = api.query_data_frame(org=config['org'], query=f'''
        from(bucket:"{config['bucket']}") 
            |> range(start:0, stop: now())
            |> filter(fn: (r) => r["id"] == "{quote['id']}")
            |> last()
    ''')

    if (len(r.index) > 0):
        return math.floor(datetime.timestamp(
            r.iloc[0]['_time'] ))
    else:
        return quote['time_deployed']

def index_pair(args: tp.Tuple):

    ( write_api, quote, calls, config ) = args

    print("indexing")

    columns = ['timestamp', 'tick_cumulatives', 'liquidity_per_second_cumulatives']

    with ThreadPoolExecutor() as executor:
        for i in executor.map(read_tick_set, calls):
            print(i)
            i_df = pd.DataFrame([i], columns=columns)
            try:
                point = Point("mem")\
                    .tag("id", quote['id'])\
                    .tag("token0_name", quote['token0_name'])\
                    .tag("token1_name", quote['token1_name'])\
                    .time(
                        datetime.utcfromtimestamp(float(i[0])),
                        WritePrecision.NS
                    )
                
                point = point.field('tickCumulative', float(i[1][0]))
                point = point.field('tickCumulativeMinusPeriod', float(i[1][1]))

                write_api.write(config['bucket'], config['org'], point)
                print("written")
            except Exception as e:
                raise e

def main():
    print(f"You are using the '{network.show_active()}' network")
    config = get_config()
    quotes = get_quotes()
    client = create_client(config)
    q_api = client.query_api()
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings()
    )

    abi = get_uni_abi()
    print(type(abi))

    index_pair_calls = []

    for q in quotes:
        # print(q)
        pool = POOL(q['pair'], abi)
        t_cur = math.floor(time.time())
        t_strt = find_start(q_api, q, config)
        index_pair_calls.append( 
            ( write_api, q, get_calls(pool, t_cur, t_strt, 600) , config ) 
        )

    print(len(index_pair_calls))

    with ThreadPoolExecutor() as executor:
        print("hello")
        for i in executor.map(index_pair, index_pair_calls):
            print("i", i)

    client.close()

    # pool = POOL(quotes[0]['pair'], abi)
    # t_c = math.floor(time.time()) - 600
    # t_s = find_start(q_api, quotes[0], config)
    # calls = get_calls(pool, t_c, t_s, 600)
    # columns = ['timestamp', 'tick_cumulatives', 'liquidity_per_second_cumulatives']

    # p_df = pd.DataFrame(columns=columns)
    # with ThreadPoolExecutor() as executor:
    #     for i in executor.map(read_tick_set, calls):
    #         print(i)
    #         i_df = pd.DataFrame([i], columns=columns)

    #         try:
    #             point = Point("mem")\
    #                 .tag("id", quotes[0]['id'])\
    #                 .tag("token0_name", quotes[0]['token0_name'])\
    #                 .tag("token1_name", quotes[0]['token1_name'])\
    #                 .time(
    #                     datetime.utcfromtimestamp(float(i[0])),
    #                     WritePrecision.NS
    #                 )
                
    #             point = point.field('tickCumulative', float(i[1][0]))
    #             point = point.field('tickCumulativeMinusPeriod', float(i[1][1]))

    #             write_api.write(config['bucket'], config['org'], point)
    #             print("written")
    #         except Exception as e:
    #             raise e

    #         p_df = p_df.append(i_df, ignore_index=True)

    # client.close()

