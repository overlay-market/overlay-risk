from datetime import datetime

from pystable.pystable import q
import pytest
import os
from influxdb_client import InfluxDBClient
import typing as tp
import pandas as pd
import numpy as np
import pystable

START = 1634301023
STOP = 1634733023
ID = "UniswapV3: WETH / DAI .3%"


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "url": os.getenv("INFLUXDB_URL")
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(
        url=config['url'],
        token=config['token'],
        debug=False
    )


def get_metrics_data(bucket):
    config = get_config()
    client = create_client(config)
    query_api = client.query_api()

    query = f'''
        from(bucket:"{bucket}")
            |> range(start: {START}, stop: {STOP})
            |> filter(fn: (r) => r["id"] == "{ID}")
    '''
    print(f'Fetching data from {bucket}')
    df = query_api.query_data_frame(query=query, org=config['org'])

    return df


def get_cumulatives_data(bucket):
    config = get_config()
    client = create_client(config)
    query_api = client.query_api()
    points = 30
    secs = points*24*60*60

    query = f'''
        from(bucket:"{bucket}")
            |> range(start: {START-secs}, stop: {STOP})
            |> filter(fn: (r) => r["id"] == "{ID}")
            |> filter(fn: (r) => r["_field"] == "tick_cumulative")
    '''
    print(f'Fetching data from {bucket}')
    df = query_api.query_data_frame(query=query, org=config['org'])

    return df


CUMULS = get_cumulatives_data('ovl_univ3_1h')
METRICS = get_metrics_data('ovl_metrics_univ3')


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
        df['lag_time'] = df[['_time']].shift(i)
        df[i] =\
            (pd.to_datetime(df['_time']) - pd.to_datetime(df['lag_time']))\
            .dt.total_seconds()
        df[i] = abs(df[i] - (window * 60))
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


def get_twap(pc: pd.DataFrame) -> pd.DataFrame:
    window = 60
    period = 10
    tolerance = 10
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
    # if pc.loc[0, '_field'] == 'tick_cumulative1':
        # twaps = 1/twaps

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


def get_samples_from_twaps(
        twaps: pd.DataFrame) -> np.ndarray:
    return twaps['twap'].to_numpy()


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


def get_stats(timestamp: int, sample: np.ndarray) -> pd.DataFrame:
    t = 10 * 60
    p_alpha = [0.05, 0.01, 0.001, 0.0001]
    p_n = [144, 1008, 2016, 4320]

    # mles
    rs = [np.log(sample[i]/sample[i-1]) for i in range(1, len(sample), 1)]

    # Gaussian Fit
    fit = {'alpha': 2, 'beta': 0, 'sigma': 1, 'mu': 0, 'parameterization': 1}

    # # Check fit validity
    # print(pystable.checkparams(fit['alpha'], fit['beta'], fit['sigma'],
    #                            fit['mu'], fit['parameterization']))
    fit_dist = pystable.create(fit['alpha'], fit['beta'], fit['sigma'],
                               fit['mu'], fit['parameterization'])

    pystable.fit(fit_dist, rs, len(rs))

    # VaRs for 5%, 1%, 0.1%, 0.01% alphas, n periods into the future
    alphas = np.array(p_alpha)
    ns = np.array(p_n)
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

    return data
    # df = pd.DataFrame(data=data).T
    # df.columns = ['timestamp', 'alpha', 'beta', 'sigma', 'mu', *var_labels]
    # return df


def labels():
    p_alpha = [0.05, 0.01, 0.001, 0.0001]
    p_n = [144, 1008, 2016, 4320]

    alphas = np.array(p_alpha)
    ns = np.array(p_n)
    return [
        f'VaR alpha={alpha} n={n}'
        for n in ns
        for alpha in alphas
    ]


def test_check():
    twap = get_twap(CUMULS)
    twap = twap.reset_index().drop(columns='index')
    stats_list = []
    # for i in range(0, len(twap)):
    for i in range(4477, 4480):
        end_time = twap.loc[i, 'timestamp']
        start_time = end_time - 30*24*60*60
        filter = ((twap['timestamp'] >= start_time)
                  & (twap['timestamp'] <= end_time))
        twap_subset = twap[filter]
        samples = get_samples_from_twaps(twap_subset)
        stats = get_stats(end_time, samples)
        stats_list.append(stats)

    stats_df = pd.DataFrame(np.row_stack(stats_list))
    var_labels = labels()
    stats_df.columns = ['timestamp', 'alpha', 'beta', 'sigma',
                        'mu', *var_labels]
    stats_df = stats_df.melt(id_vars='timestamp')
    stats_df.columns = ['_time', '_field', '_value']
    stats_df._time = pd.to_datetime(stats_df._time, utc=True, unit='s')
    token1_metrics = METRICS[METRICS._type == 'price1Cumulative']
    merged_df = stats_df.merge(token1_metrics, how='inner',
                               on=['_time', '_field'])


    assert 1 == 1
