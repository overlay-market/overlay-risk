import unittest
from scripts import influx_metrics as imetrics


class TestInfluxMetrics(unittest.TestCase):
    def test_get_config(self):
        expected = {'token', 'org', 'bucket', 'source', 'url'}
        actual = set(imetrics.get_config().keys())

        self.assertEqual(expected, actual)

    def test_create_client(self):
        """
        create_client() should return an InfluxDBClient initialized with config `url` and `token` params
        """
        pass

    def test_get_params(self):
        """
        get_params() should return a `params` dict for parameters to use in statistical estimates

        Params dict should have keys
        { "points": int, "window": int, "period": int, "alpha": List[float], "n": List[int] }
        """
        pass

    def test_get_quotes(self):
        """
        get_quotes() should load from `scripts/constants/quotes.json` and return a List
        of quote dicts for quote data fetched from SushiSwap.

        Each quote dict should have keys
        { "id": str, "pair": str, "token0": str, "token1": "str", "is_price0": bool, "amount_in": float }
        """
        pass

    def test_get_price_cumulatives(self):
        """
        get_price_cumulatives(query_api, config, quote, params) should fetch priceCumulative values
        for the last `params['points']` number of days for id `quote['id']` from config bucket `source` in `org`.

        `query_api` is an InfluxDB client query_api instance.

        Should return tuple (timestamp: int, priceCumulatives0: pandas.DataFrame, priceCumulatives1: pandas.DataFrame)
        assembled from query where
          - timestamp: most recent timestamp of data in priceCumulative dataframes
          - priceCumulatives0: DataFrame with columns ['_time', '_value'] where '_value' is priceCumulative0 at unix timestamp '_time'
          - priceCumulatives1: DataFrame with columns ['_time', '_value'] where '_value' is priceCumulative1 at unix timestamp '_time'
        """
        pass

    def test_compute_amount_out(self):
        """
        compute_amount_out(twap_112, amount_in) should convert FixedPoint.uq112x112 price average
        values of `twap_112` into integer values.

        `amount_in` is the unit value for the quote currency in the pair we are
        computing for. e.g. WETH in SushiSwap YFI/WETH uses `amount_in = 1e18` (18 decimals)

        SEE: e.g. https://github.com/overlay-market/overlay-v1-core/blob/master/contracts/OverlayV1MirinMarket.sol#L55
        """
        pass

    def test_get_twap(self):
        """
        get_twap(priceCumulatives, quote, params) should calculate rolling TWAP values
        for each (`_time`, `_value`) row in the `priceCumulatives` DataFrame. Rolling TWAP values
        should be calculated with a window size of `params['window']`.

        Should return a pandas DataFrame with columns [`timestamp`, `window`, `twap`] where for each row
          - `timestamp` is the last timestamp in the rolling window (close time)
          - `window` is the time elapsed within the window
          - `twap` is the TWAP value calculated for the window
        """
        pass

    def test_calc_vars(self):
        """
        calc_vars(mu, sig_sqrd, t, n, alphas) should calculate bracketed term

        [e**(mu * n * t + sqrt(sig_sqrd * n * t) * Psi^{-1}(1 - alpha))]

        in Value at Risk (VaR) expressions for each alpha value in
        numpy array `alphas`. `t` is the period size of how frequently we fetch updates
        and `n` is number of update periods into the future we wish to examine
        VaR values for.

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

        MLE values having units of [time] should be in units of `params["period"]`.

        VaR expressions should be calculated for each value of `n` in `params['n']`,
        where `n` represents number of time periods into the future estimate
        is relevant for.

        Should return a pandas DataFrame with columns

        [`timestamp`, *mle_labels, *var_labels]

        where `timestamp` is last timestamp given as an input to get_stat fn
        """
        pass

    def test_main():
        """
        main() should fetch price cumulative data over last get_params()['points'] number of days
        from config `source` of `org`, for quotes specified in `scripts/constants/quotes.json`.

        For each quote in JSON file, should compute rolling TWAP samples. Using
        rolling TWAP samples, should determine statistical estimates for distributional params (MLEs) of TWAP.
        With statistical estimates, should calculate VaR expressions and suggested funding constant `k`
        values.

        Should write computed statistical data (MLEs, VaRs) and funding constant suggested values
        to InfluxDB as a separate point in config `bucket` of config `org`
        """
        pass
