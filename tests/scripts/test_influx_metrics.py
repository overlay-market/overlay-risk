import pandas as pd
import pandas.testing as pd_testing
import os
import unittest
from unittest import mock
from scripts import influx_metrics as imetrics
import typing as tp


class TestInfluxMetrics(unittest.TestCase):

    def get_expected_price_fields(self) -> (str, str):
        '''
        Helper to return expected price fields
        '''
        return 'price0Cumulative', 'price1Cumulative'


    def get_query_df(self) -> pd.DataFrame:
        '''
        Helper to return dataframe used to mock out `query_data_frame`
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers/imetrics_query_df.csv')

        df = pd.read_csv(base, sep=',')
        df._start = pd.to_datetime(df._start)
        df._stop = pd.to_datetime(df._stop)
        df._time = pd.to_datetime(df._time)

        return df

    def get_pc_dfs(self, df: pd.DataFrame) -> tp.List[pd.DataFrame]:
        '''
        Helper to format dataframe used to mock out `query_data_frame`
        '''
        df_filtered = df.filter(items=['_time', '_field', '_value'])
        p0c_field, p1c_field = 'price0Cumulative', 'price1Cumulative'
        p0c_field, p1c_field = self.get_expected_price_fields()

        df_p0c = df_filtered[df_filtered['_field'] == p0c_field]
        df_p0c = df_p0c.sort_values(by='_time', ignore_index=True)

        df_p1c = df_filtered[df_filtered['_field'] == p1c_field]
        df_p1c = df_p1c.sort_values(by='_time', ignore_index=True)

        return [df_p0c, df_p1c]


    @mock.patch('scripts.influx_metrics.InfluxDBClient')
    def test_create_client(self, mock_idb_client):
        '''
        Assert that an `InfluxDBClient` is instantiated one time with config
        dict containing the `url` and `token` key-value pairs
        '''
        config = {
            'token': 'INFLUXDB_TOKEN',
            'org': 'INFLUXDB_ORG',
            'bucket': 'ovl_metrics_dev',
            'source': 'ovl_sushi',
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
        expected_keys = {"points", "window", "period", "alpha", "n"}

        actual = imetrics.get_params()
        actual_keys = set(actual.keys())
        print(actual)
        print(type(actual['alpha']))

        self.assertEqual(expected_keys, actual_keys)
        self.assertIsInstance(actual['points'], int)
        self.assertIsInstance(actual['window'], int)
        self.assertIsInstance(actual['period'], int)
        self.assertIsInstance(actual['alpha'], tp.List)
        self.assertIsInstance(actual['n'], tp.List)

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
        qp = 'scripts/constants/quotes.json'
        expected = os.path.join(base, qp)

        actual = imetrics.get_quote_path()

        self.assertEqual(expected, actual)

    def test_get_quotes(self):
        expected_keys = {'id', 'pair', 'token0', 'token1', 'token0_name',
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
        Assert price fields are `('price0Cumulative', 'price1Cumulative')`
        '''
        expected = self.get_expected_price_fields()
        actual = imetrics.get_price_fields()

        self.assertEqual(expected, actual)

    # RR TODO: run for all quotes in `quotes.json`
    @mock.patch('influxdb_client.client.query_api.QueryApi.query_data_frame')
    def test_get_price_cumulatives(self, mock_df):
        expected_timestamp = 1624136461.0
        query_df = self.get_query_df()
        expected_pcs = self.get_pc_dfs(query_df)
        mock_df.return_value = query_df

        config = {
            'token': 'INFLUXDB_TOKEN',
            'org': 'INFLUXDB_ORG',
            'bucket': 'ovl_metrics_dev',
            'source': 'ovl_sushi',
            'url': 'INFLUXDB_URL',
        }

        client = imetrics.create_client(config)
        query_api = client.query_api()
        params = imetrics.get_params()
        quotes = imetrics.get_quotes()
        quote = quotes[0]

        actual = imetrics.get_price_cumulatives(query_api, config, quote,
                                                params)
        actual_timestamp = actual[0]
        actual_pcs = actual[1]

        self.assertEqual(expected_timestamp, actual_timestamp)
        pd_testing.assert_frame_equal(expected_pcs[0], actual_pcs[0])
        pd_testing.assert_frame_equal(expected_pcs[1], actual_pcs[1])

    def test_compute_amount_out(self):
        """
        compute_amount_out(twap_112, amount_in) should convert
        FixedPoint.uq112x112 price average
        values of `twap_112` into integer values.

        `amount_in` is the unit value for the quote currency in the pair we are
        computing for. e.g. WETH in SushiSwap YFI/WETH uses `amount_in = 1e18`
        (18 decimals)

        SEE: e.g. https://github.com/overlay-market/overlay-v1-core/blob/master/contracts/OverlayV1MirinMarket.sol#L55 # noqa: E501
        """
        pass

    def test_get_twap(self):
        """
        get_twap(priceCumulatives, quote, params) should calculate rolling TWAP
        values for each (`_time`, `_value`) row in the `priceCumulatives`
        DataFrame. Rolling TWAP values should be calculated with a window size
        of `params['window']`.

        Should return a pandas DataFrame with columns [`timestamp`, `window`,
        `twap`] where for each row
          - `timestamp` is the last timestamp in the rolling window (close
            time)
          - `window` is the time elapsed within the window
          - `twap` is the TWAP value calculated for the window
        """
        pass

    def test_calc_vars(self):
        """
        calc_vars(mu, sig_sqrd, t, n, alphas) should calculate bracketed term

        [e**(mu * n * t + sqrt(sig_sqrd * n * t) * Psi^{-1}(1 - alpha))]

        in Value at Risk (VaR) expressions for each alpha value in
        numpy array `alphas`. `t` is the period size of how frequently we fetch
        updates and `n` is number of update periods into the future we wish
        to examine VaR values for.

        Should return a numpy array of calculated values for each `alpha`.

        SEE: https://oips.overlay.market/notes/note-4
        """
        pass

    def test_get_stat(self):
        """
        get_stat(timestamp, sample, quote, params) should compute the maximum
        likelihood estimates (MLEs) for distributional params of `sample` data
        as well as bracketed VaR expressions for various confidence levels
        specified in `params['alpha']` list.

        MLE values having units of [time] should be in units of
        `params["period"]`.

        VaR expressions should be calculated for each value of `n` in
        `params['n']`, where `n` represents number of time periods into the
        future estimate is relevant for.

        Should return a pandas DataFrame with columns

        [`timestamp`, *mle_labels, *var_labels]

        where `timestamp` is last timestamp given as an input to get_stat fn
        """
        pass

    def test_main(self):
        """
        main() should fetch price cumulative data over last
        get_params()['points'] number of days from config `source` of `org`,
        for quotes specified in `scripts/constants/quotes.json`.

        For each quote in JSON file, should compute rolling TWAP samples. Using
        rolling TWAP samples, should determine statistical estimates for
        distributional params (MLEs) of TWAP. With statistical estimates,
        should calculate VaR expressions and suggested funding constant `k`
        values.

        Should write computed statistical data (MLEs, VaRs) and funding
        constant suggested values to InfluxDB as a separate point in config
        `bucket` of config `org`
        """
        pass
