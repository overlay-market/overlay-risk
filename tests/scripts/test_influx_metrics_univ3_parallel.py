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
from scripts import influx_metrics_univ3_parallel as imetrics


class TestInfluxMetrics(unittest.TestCase):

    @mock.patch('scripts.influx_metrics_univ3_parallel.InfluxDBClient')
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
        mock_idb_client.asset_called_with(config)

    def test_get_config(self):
        """
        Assert `config` dict contains expected InfluxDB config parameter keys
        """
        expected = {'token', 'org', 'bucket', 'source', 'url'}
        actual = set(imetrics.get_config().keys())

        self.assertEqual(expected, actual)

    def test_get_price_fields(self):
        '''
        '''
        actual = imetrics.get_price_fields()

        # testing if the return value is string
        self.assertIsInstance(actual, str)

    def test_get_twap(self):
        """
        """
        path = 'tests/helpers/influx-metrics/uniswap_v3/'
        input_df = pd.read_csv(path + 'get_twap_input_df.csv', index_col=0)
        input_df._time = pd.to_datetime(input_df._time)

        params = imetrics.get_params()
        quotes = imetrics.get_quotes()
        quote = quotes[0]

        actual_df = imetrics.get_twap(input_df, quote, params)

        # Is it pd.DataFrame
        self.assertIsInstance(actual_df, pd.DataFrame)

        # twap not contains any null values
        self.assertEqual(actual_df['twap'].isnull().values.any(), False)
