import pandas as pd
import numpy as np
import os
import json
import typing as tp
import logging

from brownie import network, Contract
from datetime import datetime
from scipy.stats import norm

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "bucket": os.getenv('INFLUXDB_BUCKET', "ovl_kv1o"),
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'], debug=False)


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-kv1o", "ingest-data-frame")
    return point_settings


def KV1O() -> Contract:
    # SushiswapV1Oracle
    return Contract.from_explorer("0xf67Ab1c914deE06Ba0F264031885Ea7B276a7cDa")


def get_params() -> tp.Dict:
    return {
        "points": 24*30,  # 1 mo of data behind to estimate mles
        "window": 2,  # 1h TWAPs
        "n": 24*7,  # n days forward to calculate VaR * a^n
    }


def get_quote_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    qp = 'constants/quotes.json'
    return os.path.join(base, qp)


def get_quotes() -> tp.List:
    quotes = []
    p = get_quote_path()
    with open(p) as f:
        data = json.load(f)
        quotes = data.get('quotes', [])
    return quotes


def get_points_total(kv1o, q: tp.Dict) -> int:
    pair = kv1o.pairFor(q["token_in"], q["token_out"])
    n = kv1o.observationLength(pair)
    return int(n / q["window"])


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
    quotes = get_quotes()
    kv1o = KV1O()
    client = create_client(config)
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    for q in quotes:
        print('id', q['id'])
        try:
            _, stats = get_stats(kv1o, q, params)
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
        except Exception as e:
            print("Failed to write quote stats to influx")
            logging.exception(e)

    client.close()
