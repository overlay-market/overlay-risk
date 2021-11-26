'''
Run using python from terminal.
Doesn't read from scripts directory (L13) when run from poetry shell.
'''

import pandas as pd
import pandas.testing as pd_testing
import typing as tp
import os
import unittest
from unittest import mock
import datetime
from scripts import influx_metrics_univ3 as imetrics


class TestInfluxMetrics(unittest.TestCase):

    def get_price_cumulatives_df(self, path) -> pd.DataFrame:
        '''
        Helper to return dataframe used to mock out `query_data_frame` in the
        `get_price_cumulatives` function in `scripts/influx_metrics.py`
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-price-cumulatives.csv')

        df = pd.read_csv(base, sep=',')
        df._start = pd.to_datetime(df._start)
        df._stop = pd.to_datetime(df._stop)
        df._time = pd.to_datetime(df._time)

        return df

    def get_pc_dfs(self, df: pd.DataFrame,
                   start_time: int, end_time: int) -> tp.List[pd.DataFrame]:
        '''
        Helper to format dataframe used to mock out `query_data_frame`
        '''
        df_filtered = df.filter(items=['_time', '_field', '_value'])

        df_p0c = df_filtered[df_filtered['_field'] == 'tick_cumulative']
        df_p0c = df_p0c.sort_values(by='_time', ignore_index=True)
        p = imetrics.get_params()
        points = p['points']
        start_time = start_time - points*24*60*60

        df_p0c = df_p0c[df_p0c['_time']
                        .apply(lambda x: x.timestamp()) >= start_time]
        df_p0c = df_p0c[df_p0c['_time']
                        .apply(lambda x: x.timestamp()) <= end_time]

        return df_p0c

    @mock.patch('scripts.influx_metrics_univ3.InfluxDBClient')
    def test_create_client(self, mock_idb_client):
        '''
        Assert that an `InfluxDBClient` is instantiated one time with config
        dict containing the `url` and `token` key-value pairs
        '''
        config = {
            'token': 'INFLUXDB_TOKEN',
            'org': 'INFLUXDB_ORG',
            'bucket': 'ovl_metrics_univ3',
            'source': 'ovl_univ3_1h',
            'url': 'INFLUXDB_URL',
        }
        self.assertEqual(mock_idb_client.call_count, 0)
        imetrics.create_client(config)
        self.assertEqual(mock_idb_client.call_count, 1)

    def test_get_config(self):
        """
        Assert `config` dict contains expected InfluxDB config parameter keys
        """
        expected = {'token', 'org', 'bucket', 'source', 'url'}
        actual = set(imetrics.get_config().keys())

        self.assertEqual(expected, actual)

    def test_get_params(self):
        """
        Assert `params` dict contains expected keys used in statistical
        estimates
        """
        expected_keys = {"points", "window", "period",
                         "tolerance", "alpha", "n", "data_start"}

        actual = imetrics.get_params()
        actual_keys = set(actual.keys())
        print(actual)
        print(type(actual['alpha']))

        self.assertEqual(expected_keys, actual_keys)
        self.assertIsInstance(actual['points'], int)
        self.assertIsInstance(actual['window'], int)
        self.assertIsInstance(actual['period'], int)
        self.assertIsInstance(actual['tolerance'], int)
        self.assertIsInstance(actual['alpha'], tp.List)
        self.assertIsInstance(actual['n'], tp.List)
        self.assertIsInstance(actual['data_start'], int)

        for i in actual['alpha']:
            self.assertIsInstance(i, float)

        for i in actual['n']:
            self.assertIsInstance(i, int)

    def test_get_quotes_path(self):
        '''
        Assert quote path is correct
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.abspath(os.path.join(base, os.pardir))
        qp = 'scripts/constants/univ3_quotes.json'
        expected = os.path.join(base, qp)

        actual = imetrics.get_quote_path()

        self.assertEqual(expected, actual)

    def test_get_quotes(self):
        expected_keys = {'id', 'block_deployed', 'time_deployed',
                         'pair', 'token0', 'token1', 'token0_name',
                         'token1_name', 'is_price0', 'amount_in'}

        actual = imetrics.get_quotes()

        self.assertIsInstance(actual, tp.List)

        for i in actual:
            actual_keys = set(i.keys())

            self.assertEqual(expected_keys, actual_keys)
            self.assertIsInstance(i['is_price0'], bool)

            self.assertIsInstance(i['id'], str)
            self.assertIsInstance(i['pair'], str)
            self.assertIsInstance(i['token0'], str)
            self.assertIsInstance(i['token1'], str)
            self.assertIsInstance(i['amount_in'], float)

    def test_get_price_fields(self):
        '''
        Assert price field is 'tick_cumulative'
        '''
        expected = 'tick_cumulative'
        actual = imetrics.get_price_fields()

        self.assertEqual(expected, actual)

    # TODO: run for all quotes in `quotes.json`
    @mock.patch('influxdb_client.client.query_api.QueryApi.query_data_frame')
    def test_get_price_cumulatives(self, mock_df):
        path = 'influx-metrics/uniswap_v3/'
        start_time = 1636526054
        end_time = 1636528398
        query_df = self.get_price_cumulatives_df(path)
        expected_pcs = self.get_pc_dfs(query_df, start_time, end_time)
        # mock_df.return_value = query_df

        config = {
            'token': 'INFLUXDB_TOKEN',
            'org': 'INFLUXDB_ORG',
            'bucket': 'ovl_metrics_univ3',
            'source': 'ovl_univ3_1h',
            'url': 'INFLUXDB_URL',
        }

        client = imetrics.create_client(config)
        query_api = client.query_api()
        params = imetrics.get_params()
        quotes = imetrics.get_quotes()
        quote = quotes[0]

        _, actual = imetrics.get_price_cumulatives(query_api, config,
                                                   quote, params,
                                                   datetime.datetime.utcfromtimestamp(start_time),
                                                   datetime.datetime.utcfromtimestamp(end_time))

        actual_pcs = actual[0]
        breakpoint()        

        pd_testing.assert_frame_equal(expected_pcs[0], actual_pcs[0])
        pd_testing.assert_frame_equal(expected_pcs[1], actual_pcs[1])

    # def test_compute_amount_out(self):
    #     pass

    # def test_get_twap(self):
    #     """
    #     get_twap(priceCumulatives, quote, params) should calculate rolling TWAP
    #     values for each (`_time`, `_value`) row in the `priceCumulatives`
    #     DataFrame. Rolling TWAP values should be calculated with a window size
    #     of `params['window']`.

    #     Should return a pandas DataFrame with columns [`timestamp`, `window`,
    #     `twap`] where for each row
    #       - `timestamp` is the last timestamp in the rolling window (close
    #         time)
    #       - `window` is the time elapsed within the window
    #       - `twap` is the TWAP value calculated for the window
    #     """
    #     path = 'influx-metrics/sushi/weth-wbtc'
    #     query_df = self.get_price_cumulatives_df(path)
    #     gpc_pcs = self.get_pc_dfs(query_df)

    #     params = imetrics.get_params()
    #     quotes = imetrics.get_quotes()
    #     quote = quotes[0]

    #     # Price Cumulative 0
    #     pcs_idx = 0
    #     expected = self.get_twap_df(path, pcs_idx)
    #     expected.timestamp = expected.timestamp.astype(object)
    #     expected.window = expected.window.astype(object)
    #     expected.twap = expected.twap.astype(float)
    #     expected.columns = ['timestamp', 'window', 'twap']

    #     actual = imetrics.get_twap(gpc_pcs[pcs_idx], quote, params)
    #     actual.columns = ['timestamp', 'window', 'twap']
    #     actual.twap = actual.twap.astype(float)
    #     pd_testing.assert_frame_equal(expected, actual)

    #     # Price Cumulative 1
    #     pcs_idx = 1
    #     expected = self.get_twap_df(path, pcs_idx)
    #     expected.twap = expected.twap.astype(float)
    #     expected.columns = ['timestamp', 'window', 'twap']

    #     actual = imetrics.get_twap(gpc_pcs[pcs_idx], quote, params)
    #     actual.columns = ['timestamp', 'window', 'twap']
    #     actual.twap = actual.twap.astype(float)
    #     pd_testing.assert_frame_equal(expected, actual)

    # def test_calc_vars(self):
    #     """
    #     calc_vars(mu, sig_sqrd, t, n, alphas) should calculate bracketed term

    #     [e**(mu * n * t + sqrt(sig_sqrd * n * t) * Psi^{-1}(1 - alpha))]

    #     in Value at Risk (VaR) expressions for each alpha value in
    #     numpy array `alphas`. `t` is the period size of how frequently we fetch
    #     updates and `n` is number of update periods into the future we wish
    #     to examine VaR values for.

    #     Should return a numpy array of calculated values for each `alpha`.

    #     SEE: https://oips.overlay.market/notes/note-4
    #     """
    #     pass

    # def test_get_stat(self):
    #     """
    #     get_stat(timestamp, sample, quote, params) should compute the maximum
    #     likelihood estimates (MLEs) for distributional params of `sample` data
    #     as well as bracketed VaR expressions for various confidence levels
    #     specified in `params['alpha']` list.

    #     MLE values having units of [time] should be in units of
    #     `params["period"]`.

    #     VaR expressions should be calculated for each value of `n` in
    #     `params['n']`, where `n` represents number of time periods into the
    #     future estimate is relevant for.

    #     Should return a pandas DataFrame with columns

    #     [`timestamp`, *mle_labels, *var_labels]

    #     where `timestamp` is last timestamp given as an input to get_stat fn
    #     """
    #     pass

    # def test_main(self):
    #     """
    #     main() should fetch price cumulative data over last
    #     get_params()['points'] number of days from config `source` of `org`,
    #     for quotes specified in `scripts/constants/quotes.json`.

    #     For each quote in JSON file, should compute rolling TWAP samples. Using
    #     rolling TWAP samples, should determine statistical estimates for
    #     distributional params (MLEs) of TWAP. With statistical estimates,
    #     should calculate VaR expressions and suggested funding constant `k`
    #     values.

    #     Should write computed statistical data (MLEs, VaRs) and funding
    #     constant suggested values to InfluxDB as a separate point in config
    #     `bucket` of config `org`
    #     """
    #     pass

    # if __name__ == '__main__':
    #     unittest.main()
