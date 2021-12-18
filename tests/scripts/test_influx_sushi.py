import unittest
from unittest import mock
from scripts import influx_sushi as isushi
import typing as tp
import os


class TestInfluxSushi(unittest.TestCase):
    def test_get_config(self):
        expected = {'token', 'org', 'bucket', 'url'}
        actual = set(isushi.get_config().keys())

        self.assertEqual(expected, actual)

    @mock.patch('scripts.influx_sushi.InfluxDBClient')
    def test_create_client(self, mock_idb_client):
        '''
        Assert that an `InfluxDBClient` is instantiated one time with config
        dict containing the `url` and `token` key-value pairs
        '''
        config = {
            'token': 'INFLUXDB_TOKEN',
            'org': 'INFLUXDB_ORG',
            'bucket': 'ovl_sushi',
            'url': 'INFLUXDB_URL',
        }
        self.assertEqual(mock_idb_client.call_count, 0)
        isushi.create_client(config)
        self.assertEqual(mock_idb_client.call_count, 1)

    def test_get_quotes_path(self):
        '''
        Assert quote path is correct
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.abspath(os.path.join(base, os.pardir))
        qp = 'scripts/constants/quotes.json'
        expected = os.path.join(base, qp)

        actual = isushi.get_quote_path()

        self.assertEqual(expected, actual)

    def test_get_quotes(self):
        """
        get_quotes() should load from `scripts/constants/quotes.json` and
        return a List of quote dicts to use in fetching from SushiSwap.

        Each quote dict should have keys
        { "id": str, "pair": str, "token0": str, "token1": "str",
        "is_price0": bool, "amount_in": float }
        """
        expected_keys = {'id', 'pair', 'token0', 'token1', 'token0_name',
                         'token1_name', 'is_price0', 'amount_in'}

        actual = isushi.get_quotes()

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
