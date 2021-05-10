import pandas as pd
import numpy as np
import os
import json
import typing as tp
import logging

from datetime import datetime
from scipy.stats import norm

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "bucket": os.getenv('INFLUXDB_BUCKET', "ovl_metrics"),
        "source": os.getenv('INFLUXDB_SOURCE', "ovl_sushi"), # source bucket to query
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'], debug=False)


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-metrics", "ingest-data-frame")
    return point_settings


def get_params() -> tp.Dict:
    return {
        "points": 30,  # 1 mo of data behind to estimate mles
        "window": 6,  # 1h TWAPs (assuming ovl_sushi ingested every 10m)
        "n": 24*7,  # n days forward to calculate VaR * a^n
        "period": 10*60, # 10m periods
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


def get_price_fields() -> (str, str):
    return 'price0Cumulative', 'price1Cumulative'


# Fetches historical timeseries of priceCumulatives from influx
def get_price_cumulatives(query_api, cfg: tp.Dict, q: tp.Dict, p: tp.Dict) -> (int, tp.List[pd.DataFrame]):
    qid = q['id']
    points = p["points"]
    bucket = cfg['source']
    org = cfg['org']

    print(f"Fetching prices for {qid} ...")
    query = f'''
        from(bucket:"{bucket}") |> range(start: -{points}d)
            |> filter(fn: (r) => r["id"] == "{qid}")
    '''
    df = query_api.query_data_frame(query=query, org=cfg['org'])

    # Filter then separate the df into p0c and p1c dataframes
    df_filtered = df.filter(items=['_time', '_field', '_value'])
    p0c_field, p1c_field = get_price_fields()
    df_p0c = df_filtered[df_filtered._field == p0c_field]
    df_p1c = df_filtered[df_filtered._field == p1c_field]

    # Get the last timestamp
    timestamp = datetime.timestamp(
        df_filtered['_time'][len(df_filtered['_time'])-1]
    )

    return timestamp, [df_p0c, df_p1c]


def get_twap(df: pd.DataFrame, q: tp.Dict, p: tp.Dict) -> np.ndarray:
    window = p['window']

    dp = df.filter(items=['_value'])\
        .rolling(window=window)\
        .apply(lambda w : w[-1] - w[0], raw=True)
    # for time, need to map to timestamp first! then apply delt
    dt = df.filter(items=['_time'])\
        .applymap(datetime.timestamp)\
        .rolling(window=window)\
        .apply(lambda w : w[-1] - w[0], raw=True)
    twap = (dp['_value'] / dt['_time']).to_numpy()

    # Return with NaNs filtered out
    return twap[np.logical_not(np.isnan(twap))]


def get_twaps(dfs: tp.List[pd.DataFrame], q: tp.Dict, p: tp.Dict) -> tp.List[np.ndarray]:
    return [ get_twap(df, q, p) for df in dfs ]


# Calcs Normalized VaR * a^n
# See: https://oips.overlay.market/notes/note-4
def calc_vars(mu: float,
              sig_sqrd: float,
              n: int, t: int, alphas: np.ndarray) -> np.ndarray:
    sig = np.sqrt(sig_sqrd)
    q = 1-alphas
    pow = mu*n*t + sig*np.sqrt(n*t)*norm.ppf(q)
    return np.exp(pow) - 1


def get_stat(timestamp: int, sample: np.ndarray, q: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    t = p["period"] * p["window"]

    # mles
    rs = [
        np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)
        if sample[i]/sample[i-1] > 0 # TODO: investigate why twap calc was giving negative bad data (influx ingest?)
    ]
    mu = float(np.mean(rs) / t)
    ss = float(np.var(rs) / t)

    # VaRs for 5%, 1%, 0.1%, 0.01% alphas, n periods into the future
    n = p["n"]
    alphas = np.array([0.05, 0.01, 0.001, 0.0001])
    vars = calc_vars(mu, ss, n, t, alphas)
    data = np.concatenate(([timestamp, mu, ss], vars), axis=None)

    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'mu', 'sigSqrd', 'VaR 5',
                  'VaR 1', 'VaR 0.1',
                  'VaR 0.01']
    return df


def get_stats(timestamp: int, samples: tp.List[np.ndarray], q: tp.Dict, p: tp.Dict) -> tp.List[pd.DataFrame]:
    return [get_stat(timestamp, sample, q, p) for sample in samples]


def main():
    print(f"You are using data from the mainnet network")
    config = get_config()
    params = get_params()
    quotes = get_quotes()
    client = create_client(config)
    query_api = client.query_api()
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    for q in quotes:
        print('id', q['id'])
        try:
            timestamp, pcs = get_price_cumulatives(query_api, config, q, params)
            twaps = get_twaps(pcs, q, params)

            # Calc stats for each twap (NOT inverse of each other)
            stats = get_stats(timestamp, twaps, q, params)
            print('stats[0]', stats[0])
            print('stats[1]', stats[1])

            for i, stat in enumerate(stats):
                point = Point("mem")\
                    .tag("id", q['id'])\
                    .tag("_type", f"price{i}Cumulative")\
                    .time(
                        datetime.utcfromtimestamp(float(stat['timestamp'])),
                        WritePrecision.NS
                    )

                for col in stat.columns:
                    if col != 'timestamp':
                        point = point.field(col, float(stat[col]))

                print(f"Writing {q['id']} for price{i}Cumulative to api ...")
                write_api.write(config['bucket'], config['org'], point)

        except Exception as e:
            print("Failed to write quote stats to influx")
            logging.exception(e)

    client.close()

if __name__ == '__main__':
    main()