import pandas as pd
import os
import numpy as np
import numpy.testing as np_testing
from scripts import csv_funding as cfunding
import unittest


class TestCsvFunding(unittest.TestCase):

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

    def get_ks(self, path) -> np.ndarray:
        '''
        Helper to return expected ks for stable params and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-ks.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_n(self, path) -> float:
        '''
        Helper to return n to use in all calculations
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-ns.csv')

        df = pd.read_csv(base, sep=',')
        return float(df['n'])

    def test_k(self):
        path = 'csv-funding'
        df_params = self.get_stable_params(path)
        df_alphas = self.get_alphas(path)
        n = self.get_n(path)

        expect_ks = self.get_ks(path)
        actual_ks = cfunding.k(a=df_params['alpha'], b=df_params['beta'],
                               mu=df_params['mu'], sig=df_params['sigma'],
                               n=n, alphas=df_alphas)
        np_testing.assert_allclose(expect_ks, actual_ks)

    def test_nvalue_at_risk(self):
        pass

    def test_nexpected_shortfall(self):
        pass

    def test_nexpected_value(self):
        pass
