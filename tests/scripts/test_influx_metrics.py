import os
import unittest
from scripts import influx_metrics as imetrics
import typing as tp


class TestInfluxMetrics(unittest.TestCase):
    """
    get_config() should return a `config` dict containing InfluxDB
    configuration parameters
    """
    def test_get_config(self):
        expected = {'token', 'org', 'bucket', 'source', 'url'}
        actual = set(imetrics.get_config().keys())

        self.assertEqual(expected, actual)

    """
    create_client() should return an InfluxDBClient initialized with config
    `url` and `token` params
    """
    def test_create_client(self):
        pass

    """
    get_params() should return a `params` dict for parameters to use in
    statistical estimates

    Params dict should have keys
    { "points": int, "window": int, "period": int, "alpha": List[float],
    "n": List[int] }
    """
    def test_get_params(self):
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
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.abspath(os.path.join(base, os.pardir))
        qp = 'scripts/constants/quotes.json'
        expected = os.path.join(base, qp)

        actual = imetrics.get_quote_path()

        self.assertEqual(expected, actual)

    def test_get_quotes(self):
        expected_keys = {'id', 'pair', 'token0', 'token1', 'is_price0',
                         'amount_in'}
        # Sushiswap: OVL / WETH dict should NOT contain `is_price0` key
        expected_keys_no_is_price0 = {'id', 'pair', 'token0', 'token1',
                                      'amount_in'}
        actual = imetrics.get_quotes()

        self.assertIsInstance(actual, tp.List)

        for i in actual:
            actual_keys = set(i.keys())

            if i['id'] == 'Sushiswap: OVL / WETH':
                # Sushiswap: OVL / WETH dict should NOT contain `is_price0` key
                self.assertEqual(expected_keys_no_is_price0, actual_keys)
            else:
                self.assertEqual(expected_keys, actual_keys)
                self.assertIsInstance(i['is_price0'], bool)

            self.assertIsInstance(i['id'], str)
            self.assertIsInstance(i['pair'], str)
            self.assertIsInstance(i['token0'], str)
            self.assertIsInstance(i['token1'], str)
            self.assertIsInstance(i['amount_in'], float)

    def test_get_price_fields(self):
        expected = ('price0Cumulative', 'price1Cumulative')
        actual = imetrics.get_price_fields()

        self.assertEqual(expected, actual)

    def test_get_price_cumulatives(self):
        pass
        #  df = pd.DataFrame(columns=['_time', '_field', '_value'])
        #
        #  config = imetrics.get_config()
        #  params = imetrics.get_params()
        #  client = imetrics.create_client(config)
        #  query_api = client.query_api()
        #  quotes = imetrics.get_quotes()
        #  q = quotes[0]
        #  timestamp, pcs = imetrics.get_price_cumulatives(query_api, config,
        #                                                  q, params)
        #  print('DF', df.columns)
        #  print('AC', pcs[0].columns)
        #
        #  self.assertEqual(df.columns, pcs[0].columns)

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
