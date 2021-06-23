import unittest
from scripts import influx_sushi as isushi


class TestInfluxSushi(unittest.TestCase):
    def test_get_config(self):
        expected = {'token', 'org', 'bucket', 'url'}
        actual = set(isushi.get_config().keys())

        self.assertEqual(expected, actual)

    def test_create_client(self):
        """
        create_client() should return an InfluxDBClient initialized with config
        `url` and `token` params
        """
        pass

    def test_get_quotes(self):
        """
        get_quotes() should load from `scripts/constants/quotes.json` and
        return a List of quote dicts to use in fetching from SushiSwap.

        Each quote dict should have keys
        { "id": str, "pair": str, "token0": str, "token1": "str",
        "is_price0": bool, "amount_in": float }
        """
        pass

    def test_get_prices(self):
        """
        get_prices(q) should fetch on-chain data (timestamp, priceCumulative0,
        priceCumulative1) from SushiSwap V2 `pool` address for a quote object
        `q`

        Should return a pandas DataFrame with columns ['timestamp',
        'priceCumulative0', 'priceCumulative1']
        """
        pass

    def test_main(self):
        """
        main() should fetch current (timestamp, priceCumulative0,
        priceCumulative1) values from SushiSwap for quotes specified in
        `scripts/constants/quotes.json`.

        For each quote in JSON file, should write fetched priceCumulative data
        to InfluxDB as a separate point
        in config `bucket` of config `org`
        """
        pass
