import pandas as pd
import pystable
import os
import numpy as np
import numpy.testing as np_testing
from scripts.risk_pipeline.risk.parameters import csv_liquidations as cliq
import unittest


class TestCsvLiquidations(unittest.TestCase):

    def get_csv(self, name) -> pd.DataFrame:
        '''
        Helper to return dataframes from data directory
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, 'csv-liquidations')
        base = os.path.join(base, name)

        df = pd.read_csv(base, sep=',')
        return df

    def test_gaussian(self):
        expect = pystable.create(alpha=2.0, beta=0.0, mu=0.0, sigma=1.0,
                                 parameterization=1)
        actual = cliq.gaussian()
        self.assertIsInstance(actual, type(expect))

        expect_params = (expect.contents.alpha, expect.contents.beta,
                         expect.contents.mu_1, expect.contents.sigma)
        actual_params = (actual.contents.alpha, actual.contents.beta,
                         actual.contents.mu_1, actual.contents.sigma)
        self.assertEqual(expect_params, actual_params)

    def test_rescale(self):
        df_params = self.get_csv('get-stable-params.csv')
        v = self.get_csv('get-ts.csv').iloc[0, 0]

        dist = pystable.create(
            alpha=df_params['alpha'][0], beta=df_params['beta'][0],
            mu=df_params['mu'][0], sigma=df_params['sigma'][0],
            parameterization=1)

        expect = pystable.create(
            alpha=df_params['alpha'][0], beta=df_params['beta'][0],
            mu=df_params['mu'][0]*v,
            sigma=df_params['sigma'][0]*v**(1/df_params['alpha'][0]),
            parameterization=1)
        actual = cliq.rescale(dist, v)
        self.assertIsInstance(actual, type(expect))

        expect_params = (expect.contents.alpha, expect.contents.beta,
                         expect.contents.mu_1, expect.contents.sigma)
        actual_params = (actual.contents.alpha, actual.contents.beta,
                         actual.contents.mu_1, actual.contents.sigma)
        self.assertEqual(expect_params, actual_params)

    def test_mm_long(self):
        df_params = self.get_csv('get-stable-params.csv')
        nd_alphas = self.get_csv('get-alphas.csv').values[0]
        t = self.get_csv('get-ts.csv').iloc[0, 0]
        expect_mm_ls = self.get_csv('get-mm-longs.csv').values[0]
        actual_mm_ls = cliq.mm_long(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0], t=t,
            alphas=nd_alphas)
        np_testing.assert_allclose(expect_mm_ls, actual_mm_ls)

    def test_mm_short(self):
        df_params = self.get_csv('get-stable-params.csv')
        nd_alphas = self.get_csv('get-alphas.csv').values[0]
        t = self.get_csv('get-ts.csv').iloc[0, 0]
        expect_mm_ss = self.get_csv('get-mm-shorts.csv').values[0]
        actual_mm_ss = cliq.mm_short(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0], t=t,
            alphas=nd_alphas)
        np_testing.assert_allclose(expect_mm_ss, actual_mm_ss)

    def test_mm(self):
        df_params = self.get_csv('get-stable-params.csv')
        nd_alphas = self.get_csv('get-alphas.csv').values[0]
        t = self.get_csv('get-ts.csv').iloc[0, 0]
        expect_mm_ls = self.get_csv('get-mm-longs.csv').values[0]
        expect_mm_ss = self.get_csv('get-mm-shorts.csv').values[0]
        expect_mm = np.maximum(expect_mm_ls, expect_mm_ss)
        actual_mm = cliq.mm(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0], t=t,
            alphas=nd_alphas)
        np_testing.assert_allclose(expect_mm, actual_mm)

    def test_rho_long(self):
        df_params = self.get_csv('get-stable-params.csv')
        t = self.get_csv('get-ts.csv').iloc[0, 0]
        mm = self.get_csv('get-mm-longs.csv').iloc[0, 0]
        expect_rho = self.get_csv('get-rho-long.csv').loc[0, 'rho']
        actual_rho = cliq.rho(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0],
            g_inv_short=np.log(2), t=t, alpha=0.05,
            is_long=True, mm=mm)
        np_testing.assert_allclose(expect_rho, actual_rho)

    def test_rho_short(self):
        df_params = self.get_csv('get-stable-params.csv')
        t = self.get_csv('get-ts.csv').iloc[0, 0]
        mm = self.get_csv('get-mm-shorts.csv').iloc[0, 0]
        expect_rho = self.get_csv('get-rho-short.csv').loc[0, 'rho']
        actual_rho = cliq.rho(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0],
            g_inv_short=np.log(2), t=t, alpha=0.05,
            is_long=True, mm=mm)
        np_testing.assert_allclose(expect_rho, actual_rho)

    def test_beta(self):
        df_params = self.get_csv('get-stable-params.csv')
        t = self.get_csv('get-ts.csv').iloc[0, 0]
        mm = self.get_csv('get-mm-longs.csv').iloc[0, 0]
        expect_beta = self.get_csv('get-beta.csv').loc[0, 'beta']
        actual_beta = cliq.beta(
            a=df_params['alpha'][0], b=df_params['beta'][0],
            mu=df_params['mu'][0], sig=df_params['sigma'][0],
            t=t, alpha=0.05, mm=mm)
        np_testing.assert_allclose(expect_beta, actual_beta)
