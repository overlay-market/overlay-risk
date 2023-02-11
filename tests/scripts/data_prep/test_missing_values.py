import scripts.risk_pipeline.data_prep.treat.analyse.missing_values as mv
import unittest
import pandas as pd
import pandas.testing as pdt


class TestMissingValues(unittest.TestCase):
    def col_to_datetime(self, df, col):
        df[col] = pd.to_datetime(df[col])
        return df

    def get_dfs_missing_candlesticks(self):
        input_df = pd.read_csv(
            'tests/helpers/data_prep/'
            'ETH-BTC_2022-06-01-00-00_2022-10-01-00-00_300secs.csv')
        expect_null_report = pd.read_csv(
            'tests/helpers/data_prep/'
            'ETH-BTC_2022-06-01-00-00_2022-10-01-00-00'
            '_300secs_missing_values.csv', index_col=0)
        expect_df = pd.read_csv(
            'tests/helpers/data_prep/'
            'ETH-BTC_2022-06-01-00-00_2022-10-01-00-00'
            '_300secs_df_m.csv', index_col=0)
        expect_df = self.col_to_datetime(expect_df, 'time')
        expect_null_report = self.col_to_datetime(expect_null_report,
                                                  'start_time')
        expect_null_report = self.col_to_datetime(expect_null_report,
                                                  'end_time')
        return input_df, expect_null_report, expect_df

    def test_missing_candlesticks(self):
        input_df, expect_null_report,\
            expect_df = self.get_dfs_missing_candlesticks()
        actual_df, actual_null_report = mv.missing_candlesticks(
            input_df, 300,
            'time', 'close',
        )
        pdt.assert_frame_equal(actual_df, expect_df)
        pdt.assert_frame_equal(expect_null_report, actual_null_report)
