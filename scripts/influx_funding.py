import pystable
import pandas as pd
import numpy as np
import typing as tp
import time
import datetime
import os
from scipy import integrate
from concurrent.futures import ProcessPoolExecutor
import json

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


FILENAME = "ethusd_01012020_08232021"
FILEPATH = f"csv/{FILENAME}.csv"  # datafile
T = 1  # No rescaling
TC = 40  # 10 m compounding period
CP = 4  # 5x payoff cap

# uncertainties
ALPHAS = np.array([0.01, 0.025, 0.05, 0.075, 0.1])
# periods into the future at which we want 1/compoundingFactor to start
# exceeding VaR from priceFrame: 1/(1-2k)**n >= VaR[(P(n)/P(0) - 1)]
NS = 480 * np.arange(1, 85)  # 2h, 4h, 6h, ...., 7d

# For plotting nvars
TS = 240 * np.arange(1, 720)  # 1h, 2h, 3h, ...., 30d
ALPHA = 0.05


def gaussian():
    return pystable.create(alpha=2.0, beta=0.0, mu=0.0,
                           sigma=1.0, parameterization=1)


def rescale(dist: pystable.STABLE_DIST, t: float) -> pystable.STABLE_DIST:
    mu = dist.contents.mu_1 * t
    if t > 1:
        sigma = dist.contents.sigma * \
            (t/dist.contents.alpha)**(1/dist.contents.alpha)
    if t < 1:
        sigma = dist.contents.sigma * \
            ((1/t)/dist.contents.alpha)**(-1/dist.contents.alpha)
    if t == 1:
        sigma = dist.contents.sigma

    return pystable.create(
        alpha=dist.contents.alpha,
        beta=dist.contents.beta,
        mu=mu,
        sigma=sigma,
        parameterization=1
    )


def k(a: float, b: float, mu: float, sig: float,
      n: float, v: float, g_inv: float, alphas: np.ndarray) -> np.ndarray:
    dst_y = pystable.create(alpha=a, beta=b, mu=mu*n,
                            sigma=sig*((n/a)**(1/a)), parameterization=1)

    # calc quantile accounting for cap
    cdf_y_ginv = pystable.cdf(dst_y, [g_inv], 1)[0]
    qs = pystable.q(dst_y, list(cdf_y_ginv-alphas), len(alphas))
    qs = np.array(qs)

    # factor at "infty"
    factor_long = np.exp(qs)
    # short has less risk. just needs to have a funding rate to decay
    factor_short = 1 + np.zeros(len(alphas))

    # Compare long vs short and return max of the two
    factor = np.maximum(factor_long, factor_short)

    # want (1-2k)**(n/v) = 1/factor to set k for v timeframe
    # n/v is # of compound periods that pass
    return (1 - (1/factor)**(v/n))/2.0


def nvalue_at_risk(args: tp.Tuple) -> (float, float):
    (a, b, mu, sigma, k_n, v, g_inv, alpha, t) = args

    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t/a)**(1/a), parameterization=1)

    # var long
    cdf_x_ginv = pystable.cdf(x, [g_inv], 1)[0]
    q_long = pystable.q(x, [cdf_x_ginv - alpha], 1)[0]
    nvar_long = ((1-2*k_n)**(np.floor(t/v))) * (np.exp(q_long) - 1)

    # var short
    q_short = pystable.q(x, [alpha], 1)[0]
    nvar_short = ((1-2*k_n)**(np.floor(t/v))) * (1 - np.exp(q_short))

    return {
        'nvar_long': nvar_long,
        'nvar_short': nvar_short,
        'k_n': k_n,
        't': t
    }  # TODO:returning an arg seems weird


def nexpected_shortfall(args: tp.Tuple) -> (float, float, float, float):
    (a, b, mu, sigma, k_n, v, g_inv, alpha, t) = args

    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t/a)**(1/a), parameterization=1)
    oi_imb = ((1-2*k_n)**(np.floor(t/v)))

    def integrand(y): return pystable.pdf(x, [y], 1)[0] * np.exp(y)

    # expected shortfall long
    cdf_x_ginv = pystable.cdf(x, [g_inv], 1)[0]
    q_min_long = pystable.q(x, [cdf_x_ginv - alpha], 1)[0]
    integral_long, _ = integrate.quad(integrand, q_min_long, g_inv)
    nes_long = oi_imb * (integral_long/alpha - 1)

    # expected shortfall short
    q_max_short = pystable.q(x, [alpha], 1)[0]
    integral_short, _ = integrate.quad(integrand, -np.inf, q_max_short)
    nes_short = oi_imb * (1 - integral_short/alpha)

    return {
        'nes_long': nes_long,
        'nes_short': nes_short,
        'nes_long_uncond': nes_long * alpha,
        'nes_short_uncond': nes_short * alpha,
        'k_n': k_n,
        't': t
    }


def nexpected_value(args: tp.Tuple) -> (float, float):
    (a, b, mu, sigma, k_n, v, g_inv_long, cp, g_inv_short, t) = args
    x = pystable.create(alpha=a, beta=b, mu=mu*t,
                        sigma=sigma*(t/a)**(1/a), parameterization=1)
    oi_imb = ((1-2*k_n)**(np.floor(t/v)))

    def integrand(y): return pystable.pdf(x, [y], 1)[0] * np.exp(y)

    # expected value long
    cdf_x_ginv = pystable.cdf(x, [g_inv_long], 1)[0]
    integral_long, _ = integrate.quad(integrand, -np.inf, g_inv_long)
    nev_long = oi_imb * (integral_long - cdf_x_ginv + cp*(1-cdf_x_ginv))

    # expected value short
    cdf_x_ginv_one = pystable.cdf(x, [g_inv_short], 1)[0]
    integral_short, _ = integrate.quad(integrand, -np.inf, g_inv_short)
    nev_short = oi_imb * (2*cdf_x_ginv_one - 1 - integral_short)

    return {
        'nev_long': nev_long,
        'nev_short': nev_short,
        'k_n': k_n,
        't': t
    }


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
        "bucket": os.getenv('INFLUXDB_BUCKET', "ovl_funding_univ3"),
        "source": os.getenv('INFLUXDB_SOURCE', "ovl_metrics_univ3"),
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
    point_settings = PointSettings(**{"type": "metrics-daily"})
    point_settings.add_default_tag("influx-metrics", "ingest-data-frame")
    return point_settings


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


def get_curr_hour() -> (int):
    return datetime. datetime. utcnow().hour


def get_dst(q, query_api, cfg):
    bucket = cfg['source']
    org = cfg['org']
    qid = q['id']

    for tok in ['price0Cumulative', 'price1Cumulative']:
        print(f'Fetching params for {qid}, {tok} ...')
        query = f'''
            from(bucket:"{bucket}") |> range(start: -2d)
                |> filter(fn: (r) => r["_measurement"] == "mem")
                |> filter(fn: (r) => r["id"] == "{qid}")
                |> filter(fn: (r) => r["_type"] == "{tok}")
                |> last()
        '''
        df = query_api.query_data_frame(query=query, org=org)
        breakpoint()
        if type(df) == list:
            df = pd.concat(df, ignore_index=True)


def main():
    # - get distribution parameters from influx_metrics_univ3
    # - make into dst
    # - put the whole thing in a loop to calculate everyday
    config = get_config()
    quotes = get_quotes()
    client = create_client(config)
    query_api = client.query_api()
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    while True:
        curr_hour = get_curr_hour()
        print(f'Current hour is {curr_hour}')
        if int(curr_hour) != 0:
            sleep_time = 10
            print(f'Sleep for {sleep_time} mins')
            # time.sleep(sleep_time*60)
            # continue

        for q in quotes:
            print('id', q['id'])
            dst = get_dst(q, query_api, config)
            print(
                f'''
                fit params: alpha: {dst.contents.alpha}, beta: {dst.contents.beta},
                mu: {dst.contents.mu_1}, sigma: {dst.contents.sigma}
                '''
            )

            dst = rescale(dst, 1/T)
            print(
                f'''
                rescaled params (1/T = {1/T}):
                alpha: {dst.contents.alpha}, beta: {dst.contents.beta},
                mu: {dst.contents.mu_1}, sigma: {dst.contents.sigma}
                '''
            )

            # calc k (funding constant)
            g_inv = np.log(1+CP)
            g_inv_one = np.log(2)
            ks = []
            for n in NS:
                fundings = k(dst.contents.alpha, dst.contents.beta,
                            dst.contents.mu_1, dst.contents.sigma,
                            n, TC, g_inv, ALPHAS)
                ks.append(fundings)

            df_ks = pd.DataFrame(
                data=ks,
                columns=[f"alpha={alpha}" for alpha in ALPHAS],
                index=[f"n={n}" for n in NS]
            )
            print('ks:', df_ks)
            # df_ks.to_csv(f"csv/metrics/{FILENAME}-ks.csv")

            # For different k values at alpha = 0.05 level (diff n calibs),
            # plot VaR and ES at times into the future
            nvalue_at_risk_calls = []
            nexpected_value_calls = []
            for t in TS:
                for k_n in df_ks[f"alpha={ALPHA}"]:
                    nvalue_at_risk_calls.append(
                        (
                            dst.contents.alpha, dst.contents.beta,
                            dst.contents.mu_1, dst.contents.sigma,
                            k_n, TC, g_inv, ALPHA, t
                        )
                    )
                    nexpected_value_calls.append(
                        (
                            dst.contents.alpha, dst.contents.beta,
                            dst.contents.mu_1, dst.contents.sigma,
                            k_n, TC, g_inv, CP,
                            g_inv_one, t
                        )
                    )

            nexpected_shortfall_calls = nvalue_at_risk_calls

            nvalue_at_risk_values = []
            nexpected_shortfall_values = []
            nexpected_value_values = []

            start_time = time.time()

            with ProcessPoolExecutor() as executor:
                for item in executor.map(nvalue_at_risk, nvalue_at_risk_calls):
                    nvalue_at_risk_values.append(item)

            print('nvalue_at_risk_values: ', nvalue_at_risk_values)

            with ProcessPoolExecutor() as executor:
                for item in executor.map(
                        nexpected_shortfall,
                        nexpected_shortfall_calls
                        ):
                    nexpected_shortfall_values.append(item)

            print('nexpected_shortfall_values: ', nexpected_shortfall_values)

            with ProcessPoolExecutor() as executor:
                for item in executor.map(
                        nexpected_value,
                        nexpected_value_calls
                        ):
                    nexpected_value_values.append(item)

            print('nexpected_value_values: ', nexpected_value_values)

            end_time = time.time()

            print('time taken: ', end_time - start_time)


if __name__ == '__main__':
    main()
