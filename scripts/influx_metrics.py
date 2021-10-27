import json
import logging
import numpy as np
import os
import pandas as pd
import pystable
import typing as tp

from datetime import datetime, timedelta

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


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
        "bucket": os.getenv('INFLUXDB_BUCKET', "ovl_metrics"),
        "source": os.getenv('INFLUXDB_SOURCE', "ovl_sushi"),
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
        window  [int]:          1h TWAPs (assuming `ovl_sushi` ingested every
                                10m)
        period  [int]:          10m periods [s]
        alpha   List[float]:    alpha uncertainty in VaR calc
        n:      List[int]:      number of periods into the future over which
                                VaR is calculated
    '''
    return {
        "points": 30,
        "window": 6,
        "period": 600,
        "alpha": [0.05, 0.01, 0.001, 0.0001],
        "n": [144, 1008, 2016, 4320],
    }


def get_quote_path() -> str:
    '''
    Returns full path to `quotes.json` file.

    Outputs:
        [str]:  Full path to `quotes.json` file

    '''
    base = os.path.dirname(os.path.abspath(__file__))
    qp = 'constants/quotes.json'
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
    return 'price0Cumulative', 'price1Cumulative'


def get_price_cumulatives(query_api, cfg: tp.Dict, q: tp.Dict, p: tp.Dict
                          ) -> (int, tp.List[pd.DataFrame]):
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

    print(f'Fetching prices for {qid} ...')
    query = f'''
        from(bucket:"{bucket}") |> range(start: -{points}d)
            |> filter(fn: (r) => r["id"] == "{qid}")
    '''
    df = query_api.query_data_frame(query=query, org=org)
    if type(df) == list:
        df = pd.concat(df, ignore_index=True)

    # Filter then separate the df into p0c and p1c dataframes
    df_filtered = df.filter(items=['_time', '_field', '_value'])
    p0c_field, p1c_field = get_price_fields()

    df_p0c = df_filtered[df_filtered['_field'] == p0c_field]
    df_p0c = df_p0c.sort_values(by='_time', ignore_index=True)

    df_p1c = df_filtered[df_filtered['_field'] == p1c_field]
    df_p1c = df_p1c.sort_values(by='_time', ignore_index=True)

    # Get the last timestamp
    timestamp = datetime.timestamp(df_p0c['_time'][len(df_p0c['_time'])-1])

    return timestamp, [df_p0c, df_p1c]


def compute_amount_out(twap_112: np.ndarray, amount_in: int) -> np.ndarray:
    '''
    Converts `FixedPoint.qu112x112` price average values of `twap_112` into
    integer values.
    SEE: e.g. https://github.com/overlay-market/overlay-v1-core/blob/master/contracts/OverlayV1MirinMarket.sol#L55 # noqa: E501
    Inputs:
      twap_112  [np.ndarray]:
      amount_in [int]:         Unit value for the quote currency in pair
                               e.g. WETH in SushiSwap YFI/WETH uses
                               `amount_in = 1e18` (18 decimals)

    Outputs:
      [np.ndarray]:
    '''
    rshift = np.vectorize(lambda x: int(x * amount_in) >> PC_RESOLUTION)
    return rshift(twap_112)


def get_twap(pc: pd.DataFrame, q: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    '''
    Calculates the rolling Time Weighted Average Price (TWAP) values for each
    (`_time`, `_value`) row in the `priceCumulatives` dataframe. Rolling TWAP
    values are calculated with a window size of `params['window']`.

    Inputs:
      pc  [pf.DataFrame]:  `priceCumulatives`
        _time  [int]:  Unix timestamp
        _field [str]:  Price cumulative field, `price0Cumulative` or
                       `price1Cumulatives`
        _value [int]:  Price cumulative field at unix timestamp `_time`

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
      p          [tp.Dict]:  Parameters to use in statistical estimates
        points  [int]:          1 mo of data behind to estimate mles
        window  [int]:          1h TWAPs (assuming ovl_sushi ingested every
                                10m)
        period  [int]:          10m periods [s]
        alpha   List[float]:    alpha uncertainty in VaR calc
        n:      List[int]:      number of periods into the future over which
                                VaR is calculated

    Outputs:
        [pd.DataFrame]:
    '''
    window = p['window']

    dp = pc.filter(items=['_value'])\
        .rolling(window=window)\
        .apply(lambda w: w[-1] - w[0], raw=True)

    # for time, need to map to timestamp first then apply delta
    dt = pc.filter(items=['_time'])\
        .applymap(datetime.timestamp)\
        .rolling(window=window)\
        .apply(lambda w: w[-1] - w[0], raw=True)

    # Filter out NaNs
    twap_112 = (dp['_value'] / dt['_time']).to_numpy()
    twap_112 = twap_112[np.logical_not(np.isnan(twap_112))]
    twaps = compute_amount_out(twap_112, q['amount_in'])

    window_times = dt['_time'].to_numpy()
    window_times = window_times[np.logical_not(np.isnan(window_times))]

    # Window close timestamps
    t = pc.filter(items=['_time'])\
        .applymap(datetime.timestamp)\
        .rolling(window=window)\
        .apply(lambda w: w[-1], raw=True)

    ts = t['_time'].to_numpy()
    ts = ts[np.logical_not(np.isnan(ts))]

    df = pd.DataFrame(data=[ts, window_times, twaps]).T
    df.columns = ['timestamp', 'window', 'twap']

    # Filter out any TWAPs that are less than or equal to 0;
    # TODO: Why? Ingestion from sushi?
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
    pystable.q(scale_dist, q, len(q))

    sig = sigma * (t/alpha) ** (-1/alpha)
    mu = mu / t
    pow = mu * n * t + sig * (n * t / alpha) ** (1 / alpha) * q
    return np.exp(pow) - 1


def get_stat(timestamp: int, sample: np.ndarray, p: tp.Dict
             ) -> pd.DataFrame:
    t = p["period"]

    # mles
    rs = [np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)]

    # Gaussian Fit
    fit = {'alpha': 2, 'beta': 0, 'sigma': 1, 'mu': 0, 'parameterization': 1}

    # Check fit validity
    print(pystable.checkparams(fit['alpha'], fit['beta'], fit['sigma'],
                               fit['mu'], fit['parameterization']))
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
    print("You are using data from the mainnet network")
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
            timestamp, pcs = get_price_cumulatives(query_api, config, q,
                                                   params)
            # Calculate difference between max and min date.
            data_days = pcs[0]['_time'].max() - pcs[0]['_time'].min()
            print(
                f"Number of days between latest and first "
                f"data point: {data_days}"
            )

            if data_days < timedelta(days=params['points']-1):
                print(
                    f"This pair has less than {params['points']-1} days of "
                    f"data, therefore it is not being ingested "
                    f"to {config['bucket']}"
                )
                continue

            twaps = get_twaps(pcs, q, params)
            print('timestamp', timestamp)
            print('twaps', twaps)

            # Calc stats for each twap (NOT inverse of each other)
            samples = get_samples_from_twaps(twaps)
            stats = get_stats(timestamp, samples, params)
            print('stats', stats)

            for i, stat in enumerate(stats):
                token_name = q[f'token{i}_name']
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
