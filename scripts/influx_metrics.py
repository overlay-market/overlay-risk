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


# Fixed point resolution of price cumulatives
PC_RESOLUTION = 112


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


# Will generate metrics for 1h TWAP over last 30 days with VaR stats for next 7 days
def get_params() -> tp.Dict:
    return {
        "points": 30,  # 1 mo of data behind to estimate mles
        "window": 6,  # 1h TWAPs (assuming ovl_sushi ingested every 10m)
        "period": 600, # 10m periods
        "alpha": [0.05, 0.01, 0.001, 0.0001], # alpha uncertainty in VaR calc
        "n": [144, 1008, 2016, 4320],  # number of periods into the future over which VaR is calculated
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
    if type(df) == list:
        df = pd.concat(df, ignore_index=True)

    # Filter then separate the df into p0c and p1c dataframes
    df_filtered = df.filter(items=['_time', '_field', '_value'])
    p0c_field, p1c_field = get_price_fields()
    df_p0c = df_filtered[df_filtered['_field'] == p0c_field].sort_values(by='_time', ignore_index=True)
    df_p1c = df_filtered[df_filtered['_field'] == p1c_field].sort_values(by='_time', ignore_index=True)

    # Get the last timestamp
    timestamp = datetime.timestamp(df_p0c['_time'][len(df_p0c['_time'])-1])

    return timestamp, [df_p0c, df_p1c]


def compute_amount_out(twap_112: np.ndarray, amount_in: int) -> np.ndarray:
    rshift = np.vectorize(lambda x: int(x * amount_in) >> PC_RESOLUTION)
    return rshift(twap_112)


def get_twap(pc: pd.DataFrame, q: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    window = p['window']

    dp = pc.filter(items=['_value'])\
        .rolling(window=window)\
        .apply(lambda w : w[-1] - w[0], raw=True)

    # for time, need to map to timestamp first then apply delta
    dt = pc.filter(items=['_time'])\
        .applymap(datetime.timestamp)\
        .rolling(window=window)\
        .apply(lambda w : w[-1] - w[0], raw=True)

    # with NaNs filtered out
    twap_112 = (dp['_value'] / dt['_time']).to_numpy()
    twap_112 = twap_112[np.logical_not(np.isnan(twap_112))]
    twaps = compute_amount_out(twap_112, q['amount_in'])

    # window times
    window_times = dt['_time'].to_numpy()
    window_times = window_times[np.logical_not(np.isnan(window_times))]

    # window close timestamps
    t = pc.filter(items=['_time'])\
        .applymap(datetime.timestamp)\
        .rolling(window=window)\
        .apply(lambda w : w[-1], raw=True)
    ts = t['_time'].to_numpy()
    ts = ts[np.logical_not(np.isnan(ts))]

    df = pd.DataFrame(data=[ts, window_times, twaps]).T
    df.columns = ['timestamp', 'window', 'twap']

    # filter out any twaps that are less than or equal to 0; TODO: why? injestion from sushi?
    df = df[df['twap'] > 0]
    return df


def get_twaps(pcs: tp.List[pd.DataFrame], q: tp.Dict, p: tp.Dict) -> tp.List[pd.DataFrame]:
    return [ get_twap(pc, q, p) for pc in pcs ]


def get_samples_from_twaps(twaps: tp.List[pd.DataFrame]) -> tp.List[np.ndarray]:
    return [ twap['twap'].to_numpy() for twap in twaps ]


# Calcs VaR * d^n normalized for initial imbalance
# See: https://oips.overlay.market/notes/note-4
def calc_vars(mu: float,
              sig_sqrd: float,
              t: int,
              n: int, alphas: np.ndarray) -> np.ndarray:
    sig = np.sqrt(sig_sqrd)
    q = 1-alphas
    pow = mu*n*t + sig*np.sqrt(n*t)*norm.ppf(q)
    return np.exp(pow) - 1


def get_stat(timestamp: int, sample: np.ndarray, q: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    t = p["period"]

    # mles
    rs = [
        np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)
    ]
    mu = float(np.mean(rs) / t)
    ss = float(np.var(rs) / t)

    # VaRs for 5%, 1%, 0.1%, 0.01% alphas, n periods into the future
    alphas = np.array(p["alpha"])
    ns = np.array(p["n"])
    vars = [ calc_vars(mu, ss, t, n, alphas) for n in ns ]
    var_labels = [
        f'VaR alpha={alpha} n={n}'
        for n in ns
        for alpha in alphas
    ]

    data = np.concatenate(([timestamp, mu, ss], *vars), axis=None)

    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'mu', 'sigSqrd', *var_labels]
    return df


def get_stats(timestamp: int, samples: tp.List[np.ndarray], q: tp.Dict, p: tp.Dict) -> tp.List[pd.DataFrame]:
    return [get_stat(timestamp, sample, q, p) for sample in samples]

def get_token_name(i: int, id: str):
    if i == 0:
        token_name = id.split(' / ')[0].split(': ')[1]
    else:
        token_name = id.split(' / ')[1]

    return token_name


# SEE: get_params() for more info on setup
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
            print('timestamp', timestamp)
            print('twaps', twaps)

            # Calc stats for each twap (NOT inverse of each other)
            samples = get_samples_from_twaps(twaps)
            stats = get_stats(timestamp, samples, q, params)
            print('stats', stats)

            # TODO: remove
            continue

            for i, stat in enumerate(stats):
                token_name = get_token_name(i, q['id'])
                point = Point("mem")\
                    .tag("id", q['id'])\
                    .tag('token_name', token_name)\
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
