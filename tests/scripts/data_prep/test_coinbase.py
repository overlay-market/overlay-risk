from scripts.risk_pipeline.data_prep.get import coinbase as cb
import os
import unittest
from unittest import mock
import pandas as pd


class TestCoinbase(unittest.TestCase):
    def test_get_file_name(self):
        check = cb.get_file_name('ETH-USD', 300,
                                 '2020-01-01-00-00', '2020-01-10-00-00')
        assert check == 'ETH-USD_2020-01-01-00-00_2020-01-10-00-00_300secs.csv'

    @mock.patch('scripts.risk_pipeline.data_prep.get.coinbase.helpers.file_exists')  # noqa
    @mock.patch('scripts.risk_pipeline.data_prep.get.coinbase.get_file_name')
    @mock.patch('scripts.risk_pipeline.data_prep.get.coinbase.pd.read_csv')
    def test_get_data_file_exists(self, mock_read,
                                  mock_file_name, mock_file_exists):
        sample_name = 'sample.csv'
        mock_file_exists.return_value = True
        mock_file_name.return_value = 'sample.csv'
        mock_read.return_value = pd.DataFrame()
        pair = 'LTC-USD'
        t = 300
        start = '2022-01-01-00-00'
        end = '2022-01-02-00-00'
        df, full_path = cb.get_data(pair, t, start, end)
        self.assertIsInstance(df, pd.DataFrame)
        assert full_path == os.getcwd()\
            + '/scripts/risk_pipeline/outputs/data/'\
            + sample_name
