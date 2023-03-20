import pandas as pd
import pystable
import os
import numpy as np
import numpy.testing as np_testing
from scripts.risk_pipeline.risk.parameters import csv_pricedrift as cpdrift
import unittest


class TestCsvPricedrift(unittest.TestCase):

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

    def test_gaussian(self):
        expect = pystable.create(alpha=2.0, beta=0.0, mu=0.0, sigma=1.0,
                                 parameterization=1)
        actual = cpdrift.gaussian()
        self.assertIsInstance(actual, type(expect))

        expect_params = (expect.contents.alpha, expect.contents.beta,
                         expect.contents.mu_1, expect.contents.sigma)
        actual_params = (actual.contents.alpha, actual.contents.beta,
                         actual.contents.mu_1, actual.contents.sigma)
        self.assertEqual(expect_params, actual_params)

    def test_rescale(self):
        df_params = self.get_csv('get-stable-params.csv')
        v = self.get_csv('get-vs.csv').iloc[0, 0]

        dist = pystable.create(
            alpha=df_params['alpha'][0], beta=df_params['beta'][0],
            mu=df_params['mu'][0], sigma=df_params['sigma'][0],
            parameterization=1)

        expect = pystable.create(
            alpha=df_params['alpha'][0], beta=df_params['beta'][0],
            mu=df_params['mu'][0]*v,
            sigma=df_params['sigma'][0]*v**(1/df_params['alpha'][0]),
            parameterization=1)
        actual = cpdrift.rescale(dist, v)
        self.assertIsInstance(actual, type(expect))

        expect_params = (expect.contents.alpha, expect.contents.beta,
                         expect.contents.mu_1, expect.contents.sigma)
        actual_params = (actual.contents.alpha, actual.contents.beta,
                         actual.contents.mu_1, actual.contents.sigma)
        self.assertEqual(expect_params, actual_params)

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
