from numpy.core.numeric import NaN
import os
import json
import typing as tp
import math
import requests
import atexit
import json

from concurrent.futures import ThreadPoolExecutor 

from brownie import network, Contract

block_subgraph = 'https://api.thegraph.com/subgraphs/name/decentraland/blocks-ethereum-mainnet'

mock_feeds = { }

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

        (b,_) = get_b_t(t_strt)
        slot0 = pair.slot0(block_identifier=b)

        print("b", b)
        print("slot0[4]", slot0[4], "cardinality", cardinality, slot0[4] == cardinality)
        if (slot0[4] == cardinality):
            cardinality = slot0[4]
            t_strt += 86400
        else: 
            print("break")
            break

    return t_strt

def POOL(addr: str, abi) -> Contract:
    # UniswapV3Pool contract
    return Contract.from_abi("unipool", addr, abi)

def find_start(quote) -> int:
    try:
        timestamps = mock_feeds[quote['id']]['timestamps']
        print("START", mock_feeds)
        return timestamps[len(timestamps) - 1]
    except ( KeyError, IndexError ) as e:
        print("ERROR", quote['time_deployed'], e)
        return quote['time_deployed']

def index_pair(args: tp.Tuple):

    ( quote, calls ) = args

    with ThreadPoolExecutor() as executor:
        for item in executor.map(read_tick_set, calls):
            print("item", item)
            if not math.isnan(item[0]):
                mock_feeds[quote['id']]['timestamps'].append(item[0])
                mock_feeds[quote['id']]['tick_cumulatives'].append(list(item[1]))
                mock_feeds[quote['id']]['liquidity_cumulatives'].append(list(item[2]))

def mock_feeds_path():
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, '../univ3_feed_output/univ3_mock_feeds.json')
    return os.path.normpath(p)

def main():
    print(f"You are using the '{network.show_active()}' network")

    try:
        with open(mock_feeds_path()) as f:
            mock_feeds = json.load(f)
    except Exception as e:
        pass

    quotes = get_quotes()

    get_uni_prices_mock(quotes)

def get_uni_prices_mock (quotes):
    global mock_feeds

    abi = get_uni_abi()
    index_pair_calls = []
    for q in quotes:
        pool = POOL(q['pair'], abi)
    #     t_cur = math.floor(time.time())
    #     print("b4 start")
    #     t_start = find_start(q)
    #     t_cardinality = find_cardinality_increase_time(pool, q['time_deployed']+1)
    #     t_start = t_start if t_start > t_cardinality else t_cardinality
    #     read_tick_calls = [ (pool,x,[0,600]) for x in np.arange(t_start, t_cur, 600 )]
    #     index_pair_calls.append(( q, read_tick_calls ))
        slot0 = pool.slot0(block_identifier=12463427)
        print(slot0)
        pool.observe([0, 600], block_identifier=12463427)

    # with ThreadPoolExecutor() as executor:
    #     for i in executor.map(index_pair, index_pair_calls):
    #         print("done", i)

    # pass

def hello():
    print("obs_json", mock_feeds)
    with open(mock_feeds_path(), 'w', encoding='utf-8') as file:
        json.dump(mock_feeds, file, ensure_ascii=False, indent=4)

atexit.register(hello)