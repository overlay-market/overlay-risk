import pandas as pd
import numpy as np
import os
import typing as tp

from brownie import network, Contract
from scipy.stats import norm

from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "bucket": "ovl_kv1o",
        "url": "https://us-central1-1.gcp.cloud2.influxdata.com",
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'])


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    # point_settings.add_default_tag("example-name", "ingest-data-frame")
    return point_settings


def KV1O() -> str:
    # SushiswapV1Oracle
    return Contract.from_explorer("0xf67Ab1c914deE06Ba0F264031885Ea7B276a7cDa")


def quotes() -> tp.List:
    return [
        {
            "id": "SushiswapV1Oracle: USDT-WETH",
            "token_in": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "token_out": "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
            "amount_in": 1e18,
            "points": 24*30,  # 30 days
            "window": 2,  # 30min window
        }
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


def get_stats(kv1o, q) -> (str, pd.DataFrame):
    t = kv1o.periodSize() * q["window"]  # 30m * 2 = 1h
    pair = kv1o.pairFor(q["token_in"], q["token_out"])
    sample = kv1o.sample(
        q["token_in"],
        q["amount_in"],
        q["token_out"],
        q["points"],
        q["window"],
    )

    # timestamp: so don't add duplicates
    timestamp, _, _ = kv1o.lastObservation(pair)

    # mles
    rs = [np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)]
    mu = float(np.mean(rs) / t)
    ss = float(np.var(rs) / t)

    # VaRs for 5%, 1%, 0.1%, 0.01% alphas, 1 day into the future
    n = 24  # 24 hrs = 1 day
    alphas = np.array([0.05, 0.01, 0.001, 0.0001])
    vars = calc_vars(mu, ss, n, t, alphas)
    data = np.concatenate(([timestamp, mu, ss], vars), axis=None)

    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'mu', 'sigSqrd', 'VaR 5% * a^n',
                  'Var 1% * a^n', 'Var 0.1% * a^n',
                  'Var 0.01% * a^n']

    return pair, df


def main():
    print(f"You are using the '{network.show_active()}' network")
    config = get_config()
    client = create_client(config)
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    for q in quotes():
        _, stats = get_stats(KV1O(), q)
        write_api.write(bucket=config['bucket'],
                        org=config['org'],
                        record=stats,
                        data_frame_measurement_name=q['id'])

    client.close()
