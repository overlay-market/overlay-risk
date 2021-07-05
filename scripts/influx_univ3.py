import pandas as pd
import numpy as np
import os
import json
import typing as tp
import logging
import time
import math

from brownie import chain, network, Contract
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

def read_ticks_and_liquidity(pool, snapshots, block):

    i = 0
    while True:
        if (i == 10): break
        slot0 = pool.slot0(block_identifier=block)
        liquidity = pool.liquidity(block_identifier=block)

        snapshots.append({
            block: {
                'tick': slot0[1],
                'liqudity': liquidity
            }
        })

        block = get_block_by_timestamp(block, chain[block].timstamp - 600)
        i += i


    slot0 = pool.slot0(block_identifier=block)

    block = get_block_by_timestamp

def read_observations(pool, time_from):

    try:

        block = get_block_by_timestamp(len(chain) - 1, 1, time_from)

        slot0 = pool.slot0(block_identifier=block)

        index = slot0[2]
        cardinality = slot0[3]

        new_observations = []

        for i in range(cardinality):
            i = ( index + i + 1 ) % cardinality
            observation = pool.observations(i, block_identifier=block)
            new_observations.append(observation)

        new_observations.reverse()

        time_from = new_observations[len(new_observations) - 1][0]

        new_obs_df = pd.DataFrame(
            new_observations, 
            columns = [ 'timestamp', 'tick', 'secondsPerLiquidity', 'initialized' ]
        )

        return time_from, new_obs_df

    except Exception as e:
        raise(e)

def get_block_by_timestamp (block_upper: int, block_lower: int, time_target: int) -> int:

    if block_upper == block_lower: return block_upper

    time_upper = chain[block_upper].timestamp
    if time_upper == time_target: return block_upper

    time_lower = chain[block_lower].timestamp
    if time_lower == time_target: return block_lower

    block_time = math.ceil( (time_upper-time_lower) / (block_upper-block_lower) )

    block_approx = math.ceil( block_upper - ( (time_upper-time_target) / block_time ))
    
    time_approx_actual = chain[block_approx].timestamp

    time_delta = time_approx_actual - time_target
    block_delta = math.ceil( time_delta / block_time )

    if time_delta > 0:
        return get_block_by_timestamp(block_approx, block_approx - block_delta, time_target)
    else:
        return get_block_by_timestamp(block_approx + block_delta, block_approx, time_target)

def main():

    print(f"You are using the '{network.show_active()}' network")

    config = get_config()
    client = create_client(config)

    quotes = get_quotes()
    pool = POOL(quotes[0]['pair'])
    
    try:
        obs_df = pd.read_csv('observations.csv', index_col=0)
        time_from = obs_df.iloc[-1]['timestamp']
    except Exception as e:
        obs_df = pd.DataFrame([])
        time_from = chain[len(chain) - 1].timestamp


    while True:

        try:
            time_from, new_obs_df = read_observations(pool, time_from)
            obs_df = obs_df.append(new_obs_df, ignore_index=True)
            obs_df.to_csv('observations.csv')
        except Exception as e:
            print(e)
