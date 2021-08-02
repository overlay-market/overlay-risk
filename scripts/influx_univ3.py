from numpy.core.numeric import NaN
from numpy.lib.type_check import nan_to_num
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
import atexit

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor 
from concurrent.futures import ProcessPoolExecutor 
import concurrent.futures

from brownie import web3, chain, network, Contract
from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings

block_subgraph = 'https://api.thegraph.com/subgraphs/name/decentraland/blocks-ethereum-mainnet'

obs_json = {
    'timestamps': [],
    'tick_cumulatives': [],
    'liquidity_cumulatives': []
}


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
    qp = 'constants/univ3_quotes_simple.json'
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

    while t_from < t_to:
        calls.append((pair, t_from, t_period)) 
        t_from += t_period

    return calls

def read_tick_set( args: tp.Tuple ) -> tp.Tuple:

    ( pair, t_at, ts ) = args
    ( b, b_t ) = get_b_t(t_at)

    try:
        ( ticks_cum, liqs_cum ) = pair.observe(ts, block_identifier=b)
        return ( b_t, ticks_cum, liqs_cum )
    except Exception as e:
        return ( NaN, NaN, NaN )

def find_cardinality_increase_time (pair: Contract, t_strt: int) -> int:

    cardinality = 1

    while True:
        ( b, b_t ) = get_b_t(t_strt)
        print("b", b)
        slot0 = pair.slot0(block_identifier=b)
        print("slot0[4]", slot0[4], "cardinality", cardinality, slot0[4] == cardinality)
        if (slot0[4] == cardinality):
            cardinality = slot0[4]
            t_strt += 86400
        else: 
            print("break")
            break

    return t_strt



def find_start(api, quote, config) -> int:

    r = api.query_data_frame(org=config['org'], query=f'''
        from(bucket:"{config['bucket']}") 
            |> range(start:0, stop: now())
            |> filter(fn: (r) => r["id"] == "{quote['id'] + ' ' + quote['variant']}")
            |> last()
    ''')

    if (len(r.index) > 0):
        return math.floor(datetime.timestamp(
            r.iloc[0]['_time'] ))
    else:
        return quote['time_deployed'] + 1200

def index_pair(args: tp.Tuple):
    global obs_json

    ( write_api, quote, calls, config ) = args

    columns = [ 'timestamp', 'tick_cumulative', 'liquidity_cumulative' ]

    with ThreadPoolExecutor() as executor:
        for item in executor.map(read_tick_set, calls):
            print("item", item)
            obs_json['timestamps'].append(item[0])
            obs_json['tick_cumulatives'].append(list(item[1]))
            obs_json['liquidity_cumulatives'].append(list(item[2]))
            # if not math.isnan(item[0]):
                # try:
                #     point = Point("mem")\
                #         .tag("id", quote['id'] + ' ' + quote['variant'])\
                #         .tag("token0_name", quote['token0_name'])\
                #         .tag("token1_name", quote['token1_name'])\
                #         .time(
                #             datetime.utcfromtimestamp(float(item[0])),
                #             WritePrecision.NS
                #         )

                #     for i, f in enumerate(quote['fields']):
                #         point = point.field(f, float(item[1][i]))
                #         point = point.field(f, float(item[1][i]))

                #     write_api.write(config['bucket'], config['org'], point)
                #     print("written", 
                #         quote['id'], 
                #         datetime.fromtimestamp(item[0]).strftime("%m/%d/%Y, %H:%M:%S")
                #     )
                # except Exception as e:
                #     raise e

def main():
    print(f"You are using the '{network.show_active()}' network")

    config = get_config()
    quotes = get_quotes()
    client = create_client(config)
    query_api = client.query_api()
    delete_api = client.delete_api()
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings()
    )

    abi = get_uni_abi()

    index_pair_calls = []

    delete_api.delete( "1970-01-01T00:00:00Z", "2021-12-12T00:00:00Z", '', bucket=config['bucket'], org=config['org'] )

    get_uni_prices_mock(quotes, query_api, write_api, config)

def get_uni_prices_mock (quotes, query_api, write_api, config):

    abi = get_uni_abi()
    index_pair_calls = []
    for q in quotes:
        q['variant'] = 'mocks'
        q['fields'] = ['tick_cumulative_now', 'tick_cumulative_then']
        pool = POOL(q['pair'], abi)
        t_cur = math.floor(time.time())
        t_start = find_start(query_api, q, config)
        t_cardinality = find_cardinality_increase_time(pool, q['time_deployed']+1)
        t_start = t_start if t_start > t_cardinality else t_cardinality
        read_tick_calls = [ (pool,x,[0,600]) for x in np.arange(t_start, t_cur, 600 )]
        times = [ x for x in np.arange(t_start, t_cur, 600) ]
        # print("times", times)
        index_pair_calls.append(( write_api, q, read_tick_calls, config))

    with ThreadPoolExecutor() as executor:
        for i in executor.map(index_pair, index_pair_calls):
            print("done", i)

    pass

def get_uni_prices_metrics (quotes, query_api, write_api, config):
    abi = get_uni_abi()

    index_pair_calls = []
    for q in quotes:

        q['variant'] = 'metrics'
        q['fields'] = ['tick_cumulative']
        pool = POOL(q['pair'], abi)
        t_cur = math.floor(time.time())
        t_start = find_start(query_api, q, config)

        read_tick_calls = [ ( pool, x, [0] ) for x in np.arange(t_start, t_cur, 600) ]
        index_pair_calls.append(( write_api, q, read_tick_calls, config))

    with ThreadPoolExecutor() as executor:
        for i in executor.map(index_pair, index_pair_calls):
            print("done", i)

    pass



def hello():
    print("obs_json", obs_json)

atexit.register(hello)