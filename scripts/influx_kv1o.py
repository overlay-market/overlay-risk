import pandas as pd
import numpy as np
import os
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
        "bucket": "ovl_kv1o",
        "url": "https://us-central1-1.gcp.cloud2.influxdata.com",
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'], debug=False)


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-kv1o", "ingest-data-frame")
    return point_settings


def KV1O() -> str:
    # SushiswapV1Oracle
    return Contract.from_explorer("0xf67Ab1c914deE06Ba0F264031885Ea7B276a7cDa")


def get_params() -> tp.Dict:
    return {
        "points": 24*30,  # 1 mo of data behind to estimate mles
        "window": 2,  # 1h TWAPs
        "n": 24*7,  # n days forward to calculate VaR * a^n
    }


def quotes(params: tp.Dict) -> tp.List:
    return [
        {
            "id": "SushiswapV1Oracle: WBTC-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
            "amount_in": 1e18,
            "points": params["points"],
            "window": params["window"],
        },
        {
            "id": "SushiswapV1Oracle: USDC-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
            "amount_in": 1e18,
            "points": params["points"],
            "window": params["window"],
        },
        {
            "id": "SushiswapV1Oracle: AAVE-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",  # AAVE
            "amount_in": 1e18,
            "points": params["points"],
            "window": params["window"],
        },
        {
            "id": "SushiswapV1Oracle: SUSHI-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2",  # SUSHI
            "amount_in": 1e18,
            "points": params["points"],
            "window": params["window"],
        },
        {
            "id": "SushiswapV1Oracle: SNX-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f",  # SNX
            "amount_in": 1e18,
            "points": params["points"],
            "window": params["window"],
        },
        {
            "id": "SushiswapV1Oracle: YFI-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e",  # YFI
            "amount_in": 1e18,
            "points": params["points"],
            "window": params["window"],
        },
        {
            "id": "SushiswapV1Oracle: COMP-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0xc00e94cb662c3520282e6f5717214004a7f26888",  # COMP
            "amount_in": 1e18,
            "points": params["points"],
            "window": params["window"],
        },
        {
            "id": "SushiswapV1Oracle: UNI-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",  # UNI
            "amount_in": 1e18,
            "points": params["points"],
            "window": params["window"],
        }
        # TODO: OVL-WETH ...
    ]


# Calcs Normalized VaR * a^n
# See: https://oips.overlay.market/notes/note-4
def calc_vars(mu: float,
              sig_sqrd: float,
              n: int, t: int, alphas: np.ndarray) -> np.ndarray:
    sig = np.sqrt(sig_sqrd)
    q = 1-alphas
    pow = mu*n*t + sig*np.sqrt(n*t)*norm.ppf(q)
    return np.exp(pow) - 1


def get_stats(kv1o, q: tp.Dict, p: tp.Dict) -> (str, pd.DataFrame):
    t = kv1o.periodSize() * q["window"]  # 30m * 2 = 1h
    pair = kv1o.pairFor(q["token_in"], q["token_out"])
    sample = kv1o.sample(
        q["token_in"],
        q["amount_in"],
        q["token_out"],
        q["points"],
        q["window"]
    )

    # TODO: Add in q-q plot to see deviation from GBM given sample

    # timestamp: so don't add duplicates
    timestamp, _, _ = kv1o.lastObservation(pair)

    # mles
    rs = [np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)]
    mu = float(np.mean(rs) / t)
    ss = float(np.var(rs) / t)

    # VaRs for 5%, 1%, 0.1%, 0.01% alphas, n periods into the future
    n = p["n"]
    alphas = np.array([0.05, 0.01, 0.001, 0.0001])
    vars = calc_vars(mu, ss, n, t, alphas)
    data = np.concatenate(([timestamp, mu, ss], vars), axis=None)

    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'mu', 'sigSqrd', 'VaR 5% * a^n',
                  'VaR 1% * a^n', 'VaR 0.1% * a^n',
                  'VaR 0.01% * a^n']

    return pair, df


def main():
    print(f"You are using the '{network.show_active()}' network")
    config = get_config()
    params = get_params()
    kv1o = KV1O()
    client = create_client(config)
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    # TODO: Populate with historical stats as long as observations.length
    # prior to the cron update over the past month

    for q in quotes(params):
        _, stats = get_stats(kv1o, q, params)
        print('id', q['id'])
        print('stats', stats)
        stats.to_csv(
            f"csv/{q['id']}-{int(datetime.now().timestamp())}.csv",
            index=False,
        )
        point = Point("mem")\
            .tag("id", q['id'])\
            .time(
                datetime.fromtimestamp(float(stats['timestamp'])),
                WritePrecision.NS
            )

        for col in stats.columns:
            if col != 'timestamp':
                point = point.field(col, float(stats[col]))

        print(f"Writing {q['id']} to api ...")
        write_api.write(config['bucket'], config['org'], point)

    client.close()
