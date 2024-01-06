import pandas as pd
import numpy as np
import os
import json
import pystable
import typing as tp
import logging
import math
import time

from datetime import datetime, timedelta

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings

# Display all columns on print
pd.set_option('display.max_columns', None)

# Fixed point resolution of price cumulatives
PC_RESOLUTION = 112


def get_config() -> tp.Dict:
    '''
    Returns a `config` dict containing InfluxDB configuration parameters

    Outputs:
        [tp.Dict]
        token   [str]:  INFLUXDB_TOKEN env, InfluxDB token
        org     [str]:  INFLUXDB_ORG env, InfluxDB organization
        bucket  [str]:  INFLUXDB_BUCKET env, InfluxDB bucket
        source  [str]:  INFLUXDB_SOURCE env, InfluxDB source bucket
        url     [str]:  INFLUXDB_URL env, InfluxDB url
    '''
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "bucket": os.getenv('INFLUXDB_BUCKET', "ovl_metrics_bal"),
        "source": os.getenv('INFLUXDB_SOURCE', "ovl_bal_1h"),
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    '''
    Returns an InfluxDBClient initialized with config `url` and `token` params
    returned by `get_config`

    Inputs:
        [tp.Dict]
        token   [str]:  INFLUXDB_TOKEN env representing an InfluxDB token
        url     [str]:  INFLUXDB_URL env representing an InfluxDB url

    Outputs:
        [InfluxDBClient]: InfluxDB client connection instance
    '''
    return InfluxDBClient(
            url=config['url'],
            token=config['token'],
            debug=False)


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-metrics", "ingest-data-frame")
    return point_settings


def get_params() -> tp.Dict:
    '''
    Returns a `params` dict for parameters to use in statistical estimates.
    Generates metrics for 1h TWAP over last 30 days with VaR stats for next 7
    days.

    Outputs:
        [tp.Dict]
        points  [int]:          1 mo of data behind to estimate MLEs
        window  [int]:          1h TWAPs
        period  [int]:          60m periods [s]
        tolerance  [int]:       Tolerance within which `period` can
                                be inaccurate [minutes]
        alpha   List[float]:    alpha uncertainty in VaR calc
        n:      List[int]:      number of periods into the future over which
                                VaR is calculated
        data_start [int]:       start calculating metrics from these many days
                                ago
    '''
    return {
        "points": 90,
        "window": 60,
        "period": 10,
        "tolerance": 10,
        "alpha": [0.05, 0.01, 0.001, 0.0001],
        "n": [144, 1008, 2016, 4320],
        "data_start": 90
    }


def get_quote_path() -> str:
    '''
    Returns full path to `quotes.json` file.

    Outputs:
        [str]:  Full path to `quotes.json` file

    '''
    base = os.path.dirname(os.path.abspath(__file__))
    qp = 'constants/balancer_quotes.json'
    return os.path.join(base, qp)


def get_quotes() -> tp.List:
    '''
    Loads from `scripts/constants/quotes.json` and return a List
    of quote dicts for quote data fetched from SushiSwap.

    Output:
        [tp.List[dict]]
        id         [str]:   Name of swap pair
        pair       [str]:   Contract address of swap pair
        token0     [str]:   Contract address of token 0 in swap pair
        token1     [str]:   Contract address of token 1 in swap pair
        is_price0  [bool]:  If true, use the TWAP value calculated from the
                            `priceCumulative0` storage variable:
                            `price0 = num_token_1 / num_token_0`

                            If false, use the TWAP value calculated from the
                            `priceCumulative1` storage variable
        amount_in  [float]:  Swap input amount
    '''
    quotes = []
    p = get_quote_path()
    with open(p) as f:
        data = json.load(f)
        quotes = data.get('quotes', [])
    return quotes


def get_price_fields() -> (str, str):
    return 'twap'


def find_start(api, quote, config, params) -> int:

    retries = 1
    success = False
    while not success and retries < 5:
        try:
            r = api.query_data_frame(org=config['org'], query=f'''
                from(bucket:"{config['bucket']}")
                    |> range(start:0, stop: now())
                    |> filter(fn: (r) => r["id"] == "{quote['id']}")
                    |> last()
            ''')
            success = True
        except Exception as e:
            wait = retries * 10
            err_cls = e.__class__
            err_msg = str(e)
            msg = f'''
            Error type = {err_cls}
            Error message = {err_msg}
            Wait {wait} secs
            '''
            print(msg)
            time.sleep(wait)
            retries += 1

    if (len(r.index) > 0):
        return math.floor(datetime.timestamp(
            r.iloc[0]['_time']))
    else:
        return math.floor(
                int(
                    datetime.timestamp(
                        datetime.now() - timedelta(days=params['data_start'])
                        )
                    )
                )


def list_of_timestamps(api, quote, config, start_ts) -> tp.List:
    retries = 1
    success = False
    now_ts = math.floor(int(datetime.timestamp(datetime.now())))
    while not success and retries < 5:
        try:
            r = api.query_data_frame(org=config['org'], query=f'''
                from(bucket:"{config['source']}")
                    |> range(start:{start_ts+1}, stop: {now_ts})
                    |> filter(fn: (r) => r["id"] == "{quote['id']}")
                    |> filter(fn: (r) => r["_field"] == "twap")
                    |> keep(columns: ["_time"])
            ''')
            success = True
        except Exception as e:
            wait = retries * 10
            err_cls = e.__class__
            err_msg = str(e)
            msg = f'''
            Error type = {err_cls}
            Error message = {err_msg}
            Wait {wait} secs
            '''
            print(msg)
            time.sleep(wait)
            retries += 1

    if r.shape[0] == 0:
        return [0]
    else:
        return list(r['_time'])


def get_twaps(
        query_api,
        cfg: tp.Dict,
        q: tp.Dict,
        p: tp.Dict,
        start_time: int,
        end_time: int) -> (int, tp.List[pd.DataFrame]):
    '''
    Fetches `historical time series of priceCumulative` values for the last
    `params['points']` number of days for id `quote['id']` from the config
    bucket `source` in `org`.

    Inputs:
        query_api  [QueryApi]:  InfluxDB client QueryApi instance
        cfg        [tp.Dict]:   Contains InfluxDB configuration parameters
          token   [str]:  INFLUXDB_TOKEN env, InfluxDB token
          org     [str]:  INFLUXDB_ORG env, InfluxDB organization
          bucket  [str]:  INFLUXDB_BUCKET env, InfluxDB bucket
          source  [str]:  INFLUXDB_SOURCE env, InfluxDB source bucket
          url     [str]:  INFLUXDB_URL env, InfluxDB url
        q          [tp.Dict]:   Quote pair entry fetched from SushiSwap
          id         [str]:   Name of swap pair
          pair       [str]:   Contract address of swap pair
          token0     [str]:   Contract address of token 0 in swap pair
          token1     [str]:   Contract address of token 1 in swap pair
          is_price0  [bool]:  If true, use the TWAP value calculated from the
                              `priceCumulative0` storage variable:
                              `price0 = num_token_1 / num_token_0`

                              If false, use the TWAP value calculated from the
                              `priceCumulative1` storage variable
          amount_in  [float]:  Swap input amount
        p          [tp.Dict]:  Parameters to use in statistical estimates
          points  [int]:          1 mo of data behind to estimate mles
          window  [int]:          1h TWAPs (assuming ovl_sushi ingested every
                                  10m) [s]
          period  [int]:          10m periods [s]
          alpha   List[float]:    alpha uncertainty in VaR calc
          n:      List[int]:      number of periods into the future over which
                                  VaR is calculated

    Outputs:
        [tuple]: Assembled from query
          timestamp          [int]:               Most recent timestamp of data
                                                  in `priceCumulative`
                                                  dataframes
          priceCumulatives0  [pandas.DataFrame]:
            _time  [int]:  Unix timestamp
            _field [str]:  Price field, `price0Cumulative`
            _value [int]:  `priceCumulative0` at unix timestamp `_time`
          priceCumulatives1  [pandas.DataFrame]:
            _time  [int]:  Unix timestamp
            _field [str]:  Price field, `price1Cumulative`
            _value [int]:  `priceCumulative1` at unix timestamp `_time`
    '''
    qid = q['id']
    points = p['points']
    bucket = cfg['source']
    org = cfg['org']
    start_time = start_time - timedelta(seconds=(points*24*60*60))
    start_time = int(datetime.timestamp(start_time))
    end_time = int(datetime.timestamp(end_time))
    print(f'Fetching prices for {qid} ...')
    query = f'''
        from(bucket:"{bucket}")
            |> range(start: {start_time}, stop: {end_time + 1})
            |> filter(fn: (r) => r["_measurement"] == "mem")
            |> filter(fn: (r) => r["_field"] == "twap")
            |> filter(fn: (r) => r["id"] == "{qid}")
            |> keep(columns: ["_time", "_field", "_value"])
    '''
    df = query_api.query_data_frame(query=query, org=org)
    if type(df) == list:
        df = pd.concat(df, ignore_index=True)

    # Filter then separate the df into p0c and p1c dataframes
    df_filtered = df.filter(items=['_time', '_field', '_value'])
    pc_field = get_price_fields()

    df_p0c = df_filtered[df_filtered['_field'] == pc_field]
    df_p0c.loc[:, '_field'] = 'twap0'
    df_p0c = df_p0c.sort_values(by='_time', ignore_index=True)

    df_p1c = df_p0c.copy()
    df_p1c.loc[:, '_field'] = 'twap1'
    df_p1c.loc[:, '_value'] = 1/df_p0c.loc[:, '_value']

    # Get the last timestamp
    timestamp = datetime.timestamp(df_p0c['_time'][len(df_p0c['_time'])-1])

    return timestamp, [df_p0c, df_p1c]


def get_samples_from_twaps(
        twaps: tp.List[pd.DataFrame]) -> tp.List[np.ndarray]:
    return [twap['_value'].to_numpy() for twap in twaps]


# Calcs VaR * d^n normalized for initial imbalance
# See: https://oips.overlay.market/notes/note-4
def calc_vars(alpha: float, beta: float, sigma: float, mu: float, t: int,
              n: int, alphas: np.ndarray) -> np.ndarray:
    '''
    Calculates bracketed term:
        [e**(mu * n * t + sqrt(sig_sqrd * n * t) * Psi^{-1}(1 - alpha))]
    in Value at Risk (VaR) expressions for each alpha value in the `alphas`
    numpy array. SEE: https://oips.overlay.market/notes/note-4

    Inputs:
      alpha   [float]:       alpha parameter from fit
      beta    [float]:       beta parameter from fit
      sigma   [float]:       sigma parameter from fit
      mu      [float]:       mu parameter from fit
      t       [int]:         period
      n       [int]:
      alphas  [np.ndarray]:  array of alphas

    Outputs:
      [np.ndarray]:  Array of calculated values for each `alpha`

    '''
    q = 1 - np.array(alphas)
    scale_dist = pystable.create(alpha, beta, 1, 0, 1)
    qtile = pystable.q(scale_dist, q, len(q))

    sig = sigma * (t/alpha) ** (-1/alpha)
    mu = mu / t
    pow = mu * n * t + sig * (n * t / alpha) ** (1 / alpha) * np.array(qtile)
    return np.exp(pow) - 1


def get_stat(timestamp: int, sample: np.ndarray, p: tp.Dict
             ) -> pd.DataFrame:
    t = p["period"] * 60

    # mles
    rs = [np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)]

    # Gaussian Fit
    fit = {'alpha': 2, 'beta': 0, 'sigma': 1, 'mu': 0, 'parameterization': 1}

    # # Check fit validity
    fit_dist = pystable.create(fit['alpha'], fit['beta'], fit['sigma'],
                               fit['mu'], fit['parameterization'])

    pystable.fit(fit_dist, rs, len(rs))

    # VaRs for 5%, 1%, 0.1%, 0.01% alphas, n periods into the future
    alphas = np.array(p["alpha"])
    ns = np.array(p["n"])
    vars = [calc_vars(fit_dist.contents.alpha, fit_dist.contents.beta,
                      fit_dist.contents.sigma, fit_dist.contents.mu_1,
                      t, n, alphas) for n in ns]
    var_labels = [
        f'VaR alpha={alpha} n={n}'
        for n in ns
        for alpha in alphas
    ]

    data = np.concatenate(([timestamp, fit_dist.contents.alpha,
                            fit_dist.contents.beta, fit_dist.contents.sigma,
                            fit_dist.contents.mu_1], *vars), axis=None)

    df = pd.DataFrame(data=data).T
    df.columns = ['timestamp', 'alpha', 'beta', 'sigma', 'mu', *var_labels]
    return df


def get_stats(
        timestamp: int,
        samples: tp.List[np.ndarray],
        p: tp.Dict) -> tp.List[pd.DataFrame]:
    return [get_stat(timestamp, sample, p) for sample in samples]


# SEE: get_params() for more info on setup
def main():
    config = get_config()
    params = get_params()
    quotes = get_quotes()
    client = create_client(config)
    query_api = client.query_api()
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    while True:
        for q in quotes:
            print('id', q['id'])
            start_ts = find_start(query_api, q, config, params)
            ts_list = list_of_timestamps(query_api, q, config, start_ts)
            if ts_list[0] == 0:
                continue
            first_ts, last_ts = ts_list[0], ts_list[len(ts_list)-1]
            _, twaps_all = get_twaps(query_api,
                                     config,
                                     q,
                                     params,
                                     first_ts,
                                     last_ts)

            try:
                # Calculate difference between max and min date.
                data_days = twaps_all[0]['_time'].max()\
                            - twaps_all[0]['_time'].min()

                if data_days < timedelta(days=params['points']-1):
                    print(
                        f"The pair has less than {params['points']-1}d of"
                        f"data, therefore it is not being ingested"
                        f"to {config['bucket']}"
                    )
                    continue

            except Exception as e:
                print("Failed to generate TWAPs")
                logging.exception(e)

            try:
                for ts in ts_list:
                    timestamp = int(ts.timestamp())
                    print('timestamp: ', datetime.fromtimestamp(timestamp))
                    end_twap = ts
                    lookb_wndw = params['points']*24*60*60
                    start_twap = (ts-timedelta(seconds=lookb_wndw))
                    twaps =\
                        [
                         twaps_all[0][(twaps_all[0]._time >= start_twap)
                                      & (twaps_all[0]._time <= end_twap)],
                         twaps_all[1][(twaps_all[1]._time >= start_twap)
                                      & (twaps_all[1]._time <= end_twap)]
                        ]

                    samples = get_samples_from_twaps(twaps)
                    stats = get_stats(timestamp, samples, params)
                    for i, stat in enumerate(stats):
                        token_name = q[f'token{i}_name']
                        point = Point("mem")\
                            .tag("id", q['id'])\
                            .tag('token_name', token_name)\
                            .tag("_type", f"price{i}Cumulative")\
                            .time(
                                datetime.utcfromtimestamp(
                                        float(stat['timestamp'])
                                        ),
                                WritePrecision.NS
                            )

                        for col in stat.columns:
                            if col != 'timestamp':
                                point = point.field(col, float(stat[col]))

                        print(f"Writing {q['id']} for price{i}Cumulative...")
                        write_api.write(config['bucket'], config['org'], point)

            except Exception as e:
                print("Failed to write quote stats to influx")
                logging.exception(e)

        print("Metrics are up to date. Wait 5 mins.")
        time.sleep(300)


if __name__ == '__main__':
    main()
