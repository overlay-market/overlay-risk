import pandas as pd
import numpy as np
import os
import json
import pystable
import typing as tp
import logging
import math
import time
import gc
from concurrent.futures import ProcessPoolExecutor

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
        "bucket": os.getenv('INFLUXDB_BUCKET', "ovl_metrics_univ3_parallel"),
        "source": os.getenv('INFLUXDB_SOURCE', "ovl_univ3_1h"),
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
    qp = 'constants/univ3_quotes.json'
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
    return 'tick_cumulative'


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
    # else:
    #     return math.floor(
    #             int(
    #                 datetime.timestamp(
    #                     datetime.now() - timedelta(days=params['data_start'])
    #                     )
    #                 )
    #             )
    else:
        return 1630800000


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
                    |> filter(fn: (r) => r["_field"] == "tick_cumulative")
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


def get_price_cumulatives(
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
            |> filter(fn: (r) => r["_field"] == "tick_cumulative")
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
    df_p0c.loc[:, '_field'] = 'tick_cumulative0'
    df_p0c = df_p0c.sort_values(by='_time', ignore_index=True)

    df_p1c = df_p0c.copy()
    df_p1c.loc[:, '_field'] = 'tick_cumulative1'
    df_p1c.loc[:, '_value'] = df_p0c.loc[:, '_value']

    # Get the last timestamp
    timestamp = datetime.timestamp(df_p0c['_time'][len(df_p0c['_time'])-1])

    return timestamp, [df_p0c, df_p1c]


def dynamic_window(
        df: pd.DataFrame,
        max_rows: int,
        window: int
        ) -> pd.DataFrame:
    '''
    Computes the window size in terms of rows such that there is as much data
    as there are seconds specified in the `window` variable.
    '''

    for i in range(1, int(max_rows+1)):
        df.loc[:, 'lag_time'] = df.loc[:, '_time'].shift(i)
        df.loc[:, i] =\
            (pd.to_datetime(df.loc[:, '_time'])
             - pd.to_datetime(df.loc[:, 'lag_time']))\
            .dt.total_seconds()
        df.loc[:, i] = abs(df.loc[:, i] - (window * 60))

        df.drop(['lag_time'], axis=1, inplace=True)

    min_df = df[[i for i in range(1, int(max_rows+1))]]\
        .idxmin(axis="columns")

    df.dropna(inplace=True)
    df = df.join(pd.DataFrame(min_df, columns=['dynamic_window']))
    df['dynamic_window'] = df['dynamic_window'].astype(int)
    return df


def delta_window(
        row: pd.Series,
        values: pd.Series,
        lookback: pd.Series
        ) -> pd.Series:
    '''
    Computes difference based on window sizes specified in `lookback`
    '''

    loc = values.index.get_loc(row.name)
    lb = lookback.loc[row.name]
    return values.iloc[loc] - values.iloc[loc-lb]


def get_twap(pc: pd.DataFrame, q: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    window = p['window']
    period = p['period']
    tolerance = p['tolerance']
    upper_limit = (window + tolerance) * 60
    lower_limit = (window - tolerance) * 60

    max_rows = ((window/period)+1) * 2

    pc = dynamic_window(pc, int(max_rows), int(window))
    pc['dp'] = pc.apply(
            delta_window,
            values=pc['_value'],
            lookback=pc['dynamic_window'],
            axis=1)
    pc['dt'] = pc.apply(
            delta_window,
            values=pc['_time'],
            lookback=pc['dynamic_window'],
            axis=1).dt.total_seconds()

    pc = pc[(pc['dt'] > 0)]
    pc = pc[((pc['dt'] <= upper_limit) & (pc['dt'] >= lower_limit))]
    pc.reset_index(inplace=True)
    # with NaNs filtered out
    log_p = pc['dp'] / pc['dt']
    twap_112 = (log_p.apply(lambda x: np.power(1.0001, x))).to_numpy()
    twaps = twap_112[np.logical_not(np.isnan(twap_112))]
    if pc.loc[0, '_field'] == 'tick_cumulative1':
        twaps = 1/twaps

    # window times
    window_times = pc['dt'].to_numpy()
    window_times = window_times[np.logical_not(np.isnan(window_times))]

    # window close timestamps
    t = pc.filter(items=['_time'])\
        .applymap(datetime.timestamp)\
        .rolling(window=1)\
        .apply(lambda w: w[-1], raw=True)
    ts = t['_time'].to_numpy()
    ts = ts[np.logical_not(np.isnan(ts))]

    df = pd.DataFrame(data=[ts, window_times, twaps]).T
    df.columns = ['timestamp', 'window', 'twap']

    # filter out any twaps that are less than or equal to 0;
    # TODO: why? injestion from sushi?
    df = df[df['twap'] > 0]
    return df


def get_twaps(
        pcs: tp.List[pd.DataFrame],
        q: tp.Dict,
        p: tp.Dict) -> tp.List[pd.DataFrame]:
    return [get_twap(pc, q, p) for pc in pcs]


def get_samples_from_twaps(
        twaps: tp.List[pd.DataFrame]) -> tp.List[np.ndarray]:
    return [twap['twap'].to_numpy() for twap in twaps]


# Calcs VaR * d^n normalized for initial imbalance
# See: https://oips.overlay.market/notes/note-4
def calc_vars(alpha: float, beta: float, sigma: float, mu: float, t: int,
              n: int, qtile: np.ndarray) -> np.ndarray:
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
    scale_dist = pystable.create(fit_dist.contents.alpha,
                                 fit_dist.contents.beta, 1, 0, 1)
    q = 1 - np.array(alphas)
    qtile = pystable.q(scale_dist, q, len(q))
    vars = [calc_vars(fit_dist.contents.alpha, fit_dist.contents.beta,
                      fit_dist.contents.sigma, fit_dist.contents.mu_1,
                      t, n, qtile) for n in ns]
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


def get_stats(args: tp.Tuple) -> tp.List[pd.DataFrame]:
    (timestamp, samples, p) = args
    print('timestamp: ', datetime.fromtimestamp(timestamp))
    return [get_stat(timestamp, sample, p) for sample in samples]


def get_stats_calls(tlist: tp.List,
                    twaps_all: tp.List[pd.DataFrame],
                    p: tp.Dict) -> tp.List[tp.Tuple]:
    calls = []
    for ts in tlist:
        timestamp = int(ts.timestamp())
        end_twap = ts.timestamp()
        lookb_wndw = p['points']*24*60*60
        start_twap = (ts-timedelta(seconds=lookb_wndw)).timestamp()
        twaps =\
            [
                twaps_all[0][(twaps_all[0].timestamp >= start_twap)
                             & (twaps_all[0].timestamp <= end_twap)],
                twaps_all[1][(twaps_all[1].timestamp >= start_twap)
                             & (twaps_all[1].timestamp <= end_twap)]
            ]
        samples = get_samples_from_twaps(twaps)
        calls.append((timestamp, samples, p))
    return calls


def get_final_df(stats):
    stats_0 = [stats[i][0] for i in range(len(stats))]
    stats_1 = [stats[i][1] for i in range(len(stats))]

    def final_df(lst, i):
        stats_df = pd.concat(lst)
        stats_df.reset_index(inplace=True)
        stats_df.drop('index', axis=1, inplace=True)
        stats_df.loc['_type'] = f"price{i}Cumulative"
        return stats_df

    return final_df(stats_0, 0)\
        .append(
            final_df(stats_1, 1),
            ignore_index=True
            )


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
            _, pcs_all = get_price_cumulatives(query_api,
                                               config,
                                               q,
                                               params,
                                               first_ts,
                                               last_ts)

            try:
                # Calculate difference between max and min date.
                data_days = pcs_all[0]['_time'].max()\
                            - pcs_all[0]['_time'].min()

                if data_days < timedelta(days=params['points']-1):
                    print(
                        f"The pair has less than {params['points']-1}d of"
                        f"data, therefore it is not being ingested"
                        f"to {config['bucket']}"
                    )
                    continue

                twaps_all = get_twaps(pcs_all, q, params)

            except Exception as e:
                print("Failed to generate TWAPs")
                logging.exception(e)

            batch_size = 125
            ts_list_batches = [ts_list[i:i+batch_size] 
                               for i in range(0, len(ts_list), batch_size)]
            for sub_ts_list in ts_list_batches:
                print(f"Calculations started for new batch of {q['id']}")
                get_stats_vals = []
                with ProcessPoolExecutor() as executor:
                    for item in executor.map(
                            get_stats,
                            get_stats_calls(sub_ts_list, twaps_all, params)
                            ):
                        get_stats_vals.append(item)

                df = get_final_df(get_stats_vals)

                with InfluxDBClient(
                        url=config['url'],
                        token=config['token'],
                        org=config['org']
                        ) as client:

                    print("Start ingestion")

                    write_api = client.write_api(
                        write_options=SYNCHRONOUS,
                        point_settings=get_point_settings())

                    j = 1
                    while j == 1:
                        try:
                            write_api.write(bucket=config['bucket'],
                                            record=df,
                                            data_frame_measurement_name="mem",
                                            data_frame_tag_columns=[
                                                'id', 'token_name', '_type'
                                                ]
                                            )
                            j = 0
                            print("Ingested to influxdb")
                        except Exception as e:
                            err_cls = e.__class__
                            err_msg = str(e)
                            msg = f'''
                            Error type = {err_cls}
                            Error message = {err_msg}
                            Wait 5 secs
                            '''
                            print(msg)
                            continue

        print("Metrics are up to date. Wait 5 mins.")
        time.sleep(300)


if __name__ == '__main__':
    main()