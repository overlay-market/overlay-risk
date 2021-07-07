import pandas as pd
import numpy as np
import os
import json
import typing as tp
import logging
import time
import math
import asyncio

from multiprocessing import Pool

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

def get_b_by_timestamp (b_upper: int, b_lower: int, t_target: int) -> int:

    t_upper = chain[b_upper].timestamp
    t_lower = chain[b_lower].timestamp

    t_diff_upper = t_upper - t_target
    t_diff_lower = t_target - t_lower 

    # print("t diff upper " + str(t_diff_upper))
    # print("t diff lower " + str(t_diff_lower))

    if abs(t_diff_upper) < 100 or abs(t_diff_lower) < 100:
        if t_diff_upper < t_diff_lower and t_diff_upper > 0:
            for i in range(100):
                if t_target > chain[b_upper - i].timestamp: 
                    print("0 a")
                    return b_upper - i
        elif t_diff_upper < t_diff_lower and t_diff_upper < 0:
            for i in range(100):
                if t_target < chain[b_upper + i].timestamp:
                    print("0 b")
                    return b_upper + i - 1
        elif t_diff_lower < 0:
            for i in range(100):
                if t_target < chain[b_lower - i].timestamp:
                    print("0 c")
                    return b_upper - i 
        else:
            for i in range(100):
                if t_target < chain[b_lower + i].timestamp:
                    print("0 d")
                    return b_lower + i - 1


    if t_upper == t_target: print("TERMINATING upper"); return b_upper
    if t_lower == t_target: print("TERMINATING lower"); return b_lower

    t_diff = t_upper - t_lower
    b_diff = b_upper - b_lower

    b_time = math.floor( t_diff / b_diff )

    # print("~~~~ target " + str(t_target))
    # print("  block \n    diff: " + str(b_diff) 
    #     + "\n    upper: " + str(b_upper) 
    #     + "\n    lower: " + str(b_lower)
    #     + "\n    time: " + str(b_time))
    # print("  time  \n    diff: " + str(t_diff) 
    #     + "\n    upper: " + str(t_upper)
    #     + "\n    lower: " + str(t_lower)
    #     + "\n    diff target upper: " + str(t_upper - t_target) 
    #     + "\n    diff target lower: " + str(t_target - t_lower))

    if b_upper - b_lower <= 1: 
        # print("3")
        i = 0
        if t_target > t_upper: 
            while True:
                if t_target < chain[b_upper + i].timestamp: return b_upper + i - 1
                i += 1
        elif t_target < t_lower: 
            # print("ZING")
            while True:
                print(chain[b_lower - i].timestamp)
                if t_target > chain[b_lower - i].timestamp: return b_lower - i 
                i = i + 1
        else: return b_lower if abs(t_target - t_lower) < abs(t_upper - t_target) else b_upper

    if t_target > t_upper:
        t_delta = t_target - t_upper
        b_delta = math.ceil( t_delta / b_time )
        # print("~@~@~ t_delta: " + str(t_delta))
        # print("~@~@~ b_delta: " + str(b_delta))
        # print("1")
        return get_b_by_timestamp(b_upper + b_delta, b_upper, t_target)

    if t_target < t_lower:
        t_delta = t_lower - t_target
        b_delta = math.floor( t_delta / b_time )
        # print("2")
        return get_b_by_timestamp(b_lower, b_lower - b_delta, t_target)


    # print("time target         " + str(t_target))
    # print("block delta         " + str(b_delta))
    # print("time delta          " + str(t_delta))
    # print("block time          " + str(b_time))

    t_delta_upper = t_upper - t_target
    t_delta_lower = t_target - t_lower

    if t_delta_upper < t_delta_lower:
        b_delta = t_delta_upper / b_time
        b_approx = math.floor( b_upper - b_delta )
        t_approx = chain[b_approx].timestamp
    else:
        b_delta = t_delta_lower / b_time
        b_approx = math.ceil( b_lower + b_delta )
        t_approx = chain[b_approx].timestamp

    t_delta = t_approx - t_target 

    if t_delta == 0: return b_approx

    b_t_approx = abs(t_delta) / b_delta

    # print("b delta " + str(b_delta))
    # print("t delta " + str(t_delta))

    b_guess = b_approx + int( t_delta / b_t_approx )

    # print("~#~#~ t_delta   " + str(t_delta))
    # print("~#~#~ bt_approx " + str(b_t_approx))
    # print("~#~#~ b_approx  " + str(b_approx))
    # print("~#~#~ b_guess   " + str(b_guess))

    if b_guess == b_approx: print("4"); return get_b_by_timestamp(b_guess, b_guess - 1, t_target)
    elif b_guess > b_approx: print("5"); return get_b_by_timestamp(b_guess, b_approx, t_target)
    else: print("6"); return get_b_by_timestamp(b_approx, b_guess, t_target)

    # b_delta = abs(math.ceil( ( t_delta / b_time )))

    # print("b_upper - b_lower    " + str(b_diff) + " (" + str(b_upper) + " - " + str(b_lower) + ")")
    # print("t_upper - t_lower    " + str(t_diff) + " (" + str(t_upper) + " - " + str(t_lower) + ")")
    # print("b_time               " + str(b_time) + " (" + str(b_diff) + "/" + str(t_diff) + ")")
    # print("b_t_approx        " + str(b_t_approx) + " (" + str(t_delta) + "/" + str(b_upper - b_approx) + ")")
    # print("b_guess              " + str(b_guess))
    # print("b_approx             " + str(b_approx))
    # print("t_delta              " + str(t_delta))
    # print("b_time          " + str(b_time))
    # print("b_approx         " + str(b_approx))
    # print("t_approx          " + str(t_approx))
    # print("t_delta          " + str(t_delta))

    # print("time approx                 " + str(t_approx))
    # print("time target - time approx   " + str(t_target - t_approx))
    # print("time delta / block time     " + str(t_delta / b_time))
    # print("block time                  " + str(b_time) 
    #     + "     ----> t_upper (" + str(t_upper) + ") - t_lower (" + str(t_lower) + ") --> " + str(t_upper - t_lower) 
    #     + " / " + str(b_upper-b_lower) + " b_upper (" + str(b_upper) + ") - b_lower (" + str(b_lower) + ")" )
    # print("block approx                " + str(b_approx))



    if t_delta > 0:
        return get_b_by_timestamp(b_approx, b_approx - b_delta, t_target)
    else:
        return get_b_by_timestamp(b_approx + b_delta, b_approx, t_target)

    if t_delta > 0:
        return get_b_by_timestamp(b_approx, b_approx - b_delta, t_target)
    else:
        return get_b_by_timestamp(b_approx + b_delta, b_approx, t_target)

def get_tick_sets(pool: any, b_upper: int, b_lower: int, period: int) -> tp.List:

    t_upper = chain[b_upper].timestamp
    t_lower = chain[b_lower].timestamp

    time = t_upper
    calls = list([])

    while t_lower < time:
        calls.append( [pool, b_upper, b_lower, time] ) 
        # calls.append( [b_upper, b_lower, time] ) 
        time -= period

    # return []
    return calls

# 
def four (a,b,c,d):
    print(a,b,c,d)

def read_tick_set(pool: any, b_upper: int, b_lower: int, t_at: int) -> tp.List:

    print("read tick set", pool, b_upper, b_lower, t_at)
    # ticks = [ 0, 0 ]

    try:
        block = get_b_by_timestamp(b_upper, b_lower, t_at)
    except RecursionError as re:
        raise Exception("RECURSION ERROR WITH " + str(b_upper) + " " + str(b_lower) + " " + str(t_at))
    except ZeroDivisionError as ze:
        print("ZE!!!!!!!")
        raise Exception("ZERO DIVISON WITH " + str(b_upper) + " " + str(b_lower) + " " + str(t_at) )
    try:
        ticks = pool([0, t_at], b_identifier=block)
    except Exception as e:
        ticks = [ 0, 0 ] 

    return ticks

def f ():
    # block = get_b_by_timestamp(12776692, 12369854, 1625481694)
    # block = get_b_by_timestamp(12777308, 12369854, 1625617252)
    # block = get_b_by_timestamp(12777667, 12369854, 1625615769)
    # block = get_b_by_timestamp(12781006, 12369854, 1625661961)
    # block = get_b_by_timestamp(12781516, 12369854, 1625665544)
    block = get_b_by_timestamp(12781964, 12369854, 1625631408)

    print("BLOCK! " + str(block))
    print("time " + str(chain[block].timestamp))

def initf(a): 
    print("init " + str(a))

def main():
    global pair

    print(f"You are using the '{network.show_active()}' network")

    config = get_config()
    # client = create_client(config)

    quotes = get_quotes()
    pair = POOL(quotes[0]['pair'])

    print(type(pair.observe))

    p = Pool(processes=15)
    calls = get_tick_sets(pair.observe, len(chain) - 1, len(chain) - 2, 60)

    print(calls)

    tick_sets = p.starmap(read_tick_set, calls)

    print("before wait")

    tick_sets.wait()
    print("before close")
    p.close()
    print("before join")
    p.join()
    print("before get")
    vals = tick_sets.get()

    print("vals", vals)

    # print("tick sets ", vals)
    # print("ready ", tick_sets.ready())
    # print("successful ", tick_sets.successful())

    # print("tick sets \/ \/ \/")
    # print(tick_sets

    # asyncio.run(f())

    print(os.cpu_count())

