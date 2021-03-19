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
        "bucket": os.getenv('INFLUXDB_BUCKET'),
        "url": os.getenv('INFLUXDB_URL'),
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
            "points": 48*30,  # 30 days
            "window": 2,  # 1 h rolling
        }
    ]


# Calcs VaR * a^n
def calc_vars(mu: float,
              sigma_sqrd: float,
              n: int, t: int, alphas: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(sigma_sqrd)
    q = 1-alphas
    pow = mu*n*t + sigma*np.sqrt(n*t)*norm.ppf(q)
    return np.exp(pow) - 1


def get_stats(kv1o, q) -> (str, pd.DataFrame):
    t = kv1o.periodSize()
    pair = kv1o.pairFor(q["token_in"], q["token_in"])
    sample = kv1o.sample(
        q["token_in"],
        q["amount_in"],
        q["token_in"],
        q["points"],
        q["window"],
    )
    # TODO: pd data frame ... for mu, ss, VaR
    mu = float(np.mean([
        np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)
    ]) / t)
    ss = float(np.var([
        np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)
    ]) / t)

    # Get the VaRs for 5%, 1%, 0.1% over 1h (T = periodSize, n = # of periodSizes into the future)
    n = q["window"]*24  # VaRs 1 day into the future
    alphas = np.array([0.05, 0.01, 0.001])
    vars = calc_vars(mu, ss, n, t, alphas)
    data = np.concatenate(([mu, ss], vars), axis=None)
#    print('mu', mu)
#    print('ss', ss)
#    print('VaRs', vars)
#    print('data', data)
    df = pd.DataFrame(data=data,
                      columns=['mu', 'sigmaSqrd', 'VaR 5% * a^n',
                               'Var 1% * a^n', 'Var 0.1% * a^n'])
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
        write_api.write(bucket=config['bucket'], record=stats,
                        data_frame_measurement_name=q['id'])

    client.close()
