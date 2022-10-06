import pandas as pd
import os
import numpy as np
import numpy.testing as np_testing
from scripts import csv_pricedrift as cpdrift
import unittest


class TestCsvImpact(unittest.TestCase):

    def get_csv(self, name) -> pd.DataFrame:
        '''
        Helper to return dataframes from data directory
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, 'csv-pricedrift')
        base = os.path.join(base, name)

        df = pd.read_csv(base, sep=',')
        return df

    def get_stable_params(self, path) -> pd.DataFrame:
        '''
        Helper to return dataframe used to init stable params
        in all functions
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-stable-params.csv')

        df = pd.read_csv(base, sep=',')
        return df

    def get_alphas(self, path) -> np.ndarray:
        '''
        Helper to return dataframe used for uncertainties
        in all functions
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-alphas.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_v(self, path) -> float:
        '''
        Helper to return n to use in all calculations
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-vs.csv')

        df = pd.read_csv(base, sep=',')
        return float(df['v'])

    def get_cp(self, path) -> float:
        '''
        Helper to return cp to use in all calculations
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-cps.csv')

        df = pd.read_csv(base, sep=',')
        return float(df['cp'])

    def get_mu_max_longs(self, path) -> np.ndarray:
        '''
        Helper to return expected mu-max longs for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-mu-max-longs.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_mu_max_shorts(self, path) -> np.ndarray:
        '''
        Helper to return expected mu-max shorts for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-mu-max-shorts.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_mu_maxs(self, path) -> np.ndarray:
        '''
        Helper to return expected mu-maxs for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)

        # long
        base_l = os.path.join(base, 'get-mu-max-longs.csv')
        df_l = pd.read_csv(base_l, sep=',')

        # short
        base_s = os.path.join(base, 'get-mu-max-shorts.csv')
        df_s = pd.read_csv(base_s, sep=',')

        return np.maximum(df_l.values[0], df_s.values[0])

    def test_mu_max_long(self):
        df_params = self.get_csv('get-stable-params.csv')
        nd_alphas = self.get_csv('get-alphas.csv').values[0]
        v = self.get_csv('get-vs.csv')
        expect_mu_max_ls = self.get_csv('get-mu-max-longs.csv').values[0]
        actual_mu_max_ls = cpdrift.mu_max_long(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0], v=v.iloc[0, 0],
            alphas=nd_alphas)
        np_testing.assert_allclose(expect_mu_max_ls, actual_mu_max_ls)

    def test_mu_max_short(self):
        df_params = self.get_csv('get-stable-params.csv')
        nd_alphas = self.get_csv('get-alphas.csv').values[0]
        v = self.get_csv('get-vs.csv')
        expect_mu_max_ss = self.get_csv('get-mu-max-shorts.csv').values[0]
        actual_mu_max_ss = cpdrift.mu_max_short(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0], v=v.iloc[0, 0],
            alphas=nd_alphas)
        np_testing.assert_allclose(expect_mu_max_ss, actual_mu_max_ss)

    def test_mu_max(self):
        df_params = self.get_csv('get-stable-params.csv')
        nd_alphas = self.get_csv('get-alphas.csv').values[0]
        v = self.get_csv('get-vs.csv')
        expect_mu_max_ls = self.get_csv('get-mu-max-longs.csv').values[0]
        expect_mu_max_ss = self.get_csv('get-mu-max-shorts.csv').values[0]
        expect_mu_max = np.maximum(expect_mu_max_ls, expect_mu_max_ss)
        actual_mu_max = cpdrift.mu_max(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0], v=v.iloc[0, 0],
            alphas=nd_alphas)
        np_testing.assert_allclose(expect_mu_max, actual_mu_max)
