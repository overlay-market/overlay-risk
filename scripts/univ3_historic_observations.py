import json
import math
import requests
import typing as tp
import os

from brownie import chain, network, Contract
from concurrent.futures import ThreadPoolExecutor
from numpy.core.numeric import NaN

BLOCK_SUBGRAPH_ROOT = "https://api.thegraph.com/"
BLOCK_SUBGRAPH_PATH = "/subgraphs/name/decentraland/blocks-ethereum-mainnet"
BLOCK_SUBGRAPH_ENDPOINT = os.path.join(
    BLOCK_SUBGRAPH_ROOT, BLOCK_SUBGRAPH_PATH)

mock_feeds = {}


def get_b_q(timestamp: int) -> str:
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


def get_uni_abi_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'constants/univ3.abi.json')


def get_uni_abi() -> tp.Dict:
    with open(get_uni_abi_path()) as f:
        data = json.load(f)
    return data


def get_b_t(timestamp: int) -> tp.Tuple:

    q = {'query': get_b_q(timestamp)}

    result = requests.post(BLOCK_SUBGRAPH_ENDPOINT, json=q)
    result = result.text
    result = json.loads(result)['data']['blocks'][0]

    return int(result['number']), int(result['timestamp'])


def read_tick_set(args: tp.Tuple) -> tp.Tuple:

    (pair, t_at, ts) = args
    (b, b_t) = get_b_t(t_at)

    try:
        (ticks_cum, liqs_cum) = pair.observe(ts, block_identifier=b)
        return (b_t, ticks_cum, liqs_cum)
    except Exception as e:
        return (NaN, NaN, NaN)


def read_observations(pool, time_from):

    try:

        block, _ = get_b_t(time_from)

        slot0 = pool.slot0(block_identifier=block)

        index = slot0[2]
        cardinality = slot0[3]

        new_obs = []

        for x in range(cardinality):
            i = (index + 1 + x) % cardinality
            o = pool.observations(i, block_identifier=block)
            b, _ = get_b_t(o[0])
            liq = pool.liquidity(block_identifier=b)
            tick = pool.slot0(block_identifier=b)[1]

            item = {
                'observation': [
                    o[0],
                    o[1],
                    o[2],
                    True
                ],
                'shim': [
                    o[0],
                    liq,
                    tick
                ]
            }

            new_obs.append(item)

        new_obs.reverse()

        time_from = new_obs[-1]['observation'][0]

        return time_from, new_obs

    except Exception as e:

        raise(e)


def prep_observations(pool, observations):

    prepped = []

    for v in observations:

        block = get_b_t(v[0])

        tick = pool.slot0(block_identifier=block)[1]
        liq = pool.liquidity(block_identifier=block)

        item = {
            'timestamp': v[0],
            'tick': tick,
            'tickCumalitive': v[1],
            'liq': liq,
            'liqCumulative': v[2]
        }

        prepped.append(item)

    return prepped


def find_cardinality_increase_time(pair: Contract, t_strt: int) -> int:

    cardinality = 1

    while True:

        b, _ = get_b_t(t_strt)
        slot0 = pair.slot0(block_identifier=b)
        if (slot0[4] == cardinality):
            t_strt += 86400
        else:
            break

    return t_strt


def POOL(addr: str, abi) -> Contract:
    # UniswapV3Pool contract
    return Contract.from_abi("unipool", addr, abi)


def find_start(quote) -> int:
    try:
        timestamps = mock_feeds[quote['id']]['timestamps']
        return timestamps[len(timestamps) - 1]
    except (KeyError, IndexError) as e:
        return quote['time_deployed']


def index_pair(args: tp.Tuple):

    (quote, calls) = args

    with ThreadPoolExecutor() as executor:
        for item in executor.map(read_tick_set, calls):
            if not math.isnan(item[0]):
                mock_feeds[quote['id']]['timestamps'].append(item[0])
                mock_feeds[quote['id']]['tick_cumulatives'].append(
                    list(item[1]))
                mock_feeds[quote['id']]['liquidity_cumulatives'].append(
                    list(item[2]))


def mock_feeds_path():
    p = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(p, 'univ3_mock_feeds.json')


def obs_json_path(q):

    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, '../univ3_feed_output',
                     'univ3_'
                     + q['token0_name'].lower() + '_'
                     + q['token1_name'].lower() + '.json')
    p = os.path.normpath(p)

    if not os.path.isfile(p):
        with open(p, 'a') as f:
            json.dump([], f, ensure_ascii=False)

    return p


def load_obs(q):
    p = obs_json_path(q)
    with open(p) as f:
        return json.load(f)


def save_obs(q, obs_json):
    print("save obs")
    p = obs_json_path(q)
    with open(p, 'w', encoding='utf-8') as file:
        json.dump(obs_json, file, ensure_ascii=False, indent=4)


def main():
    print(f"You are using the '{network.show_active()}' network")

    quotes = get_quotes()

    for q in quotes:

        pool = POOL(q['pair'], get_uni_abi())
        obs = load_obs(q)

        time_from = obs[-1]['observation'][0] if len(
            obs) else chain[-1].timestamp
        time_stop = find_cardinality_increase_time(
            pool, q['time_deployed'] + 1)

        while True:

            time_from, new_obs = read_observations(pool, time_from)

            obs.extend(new_obs)

            for i in range(len(obs)):
                if len(obs[i]['shim']) == 4:
                    obs[i]['shim'][3] = len(obs) - i
                else:
                    obs[i]['shim'].append(len(obs) - i)

            save_obs(q, obs)

            if time_from < time_stop:
                break
