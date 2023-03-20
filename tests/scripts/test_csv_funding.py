import pandas as pd
import os
import pystable
import numpy as np
import numpy.testing as np_testing
from scripts.risk_pipeline.risk.parameters import csv_funding as cfunding
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

    def get_t(self, path) -> float:
        '''
        Helper to return t to use in all calculations
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-ts.csv')

        df = pd.read_csv(base, sep=',')
        return float(df['t'])

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

    def get_value_at_risks(self, path) -> np.ndarray:
        '''
        Helper to return expected VaRs for stable params and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-vars.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_expected_shortfalls(self, path) -> np.ndarray:
        '''
        Helper to return expected ESs for stable params and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-ess.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_expected_values(self, path) -> np.ndarray:
        '''
        Helper to return expected EVs for stable params and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-evs.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def test_gaussian(self):
        expect = pystable.create(alpha=2.0, beta=0.0, mu=0.0, sigma=1.0,
                                 parameterization=1)
        actual = cfunding.gaussian()
        self.assertIsInstance(actual, type(expect))

        expect_params = (expect.contents.alpha, expect.contents.beta,
                         expect.contents.mu_1, expect.contents.sigma)
        actual_params = (actual.contents.alpha, actual.contents.beta,
                         actual.contents.mu_1, actual.contents.sigma)
        self.assertEqual(expect_params, actual_params)

    def test_rescale(self):
        path = 'csv-funding'
        df_params = self.get_stable_params(path)
        t = self.get_t(path)

        dist = pystable.create(
            alpha=df_params['alpha'], beta=df_params['beta'],
            mu=df_params['mu'], sigma=df_params['sigma'], parameterization=1)

        expect = pystable.create(
            alpha=df_params['alpha'], beta=df_params['beta'],
            mu=df_params['mu']*t,
            sigma=df_params['sigma']*t**(1/df_params['alpha']),
            parameterization=1)
        actual = cfunding.rescale(dist, t)
        self.assertIsInstance(actual, type(expect))

        expect_params = (expect.contents.alpha, expect.contents.beta,
                         expect.contents.mu_1, expect.contents.sigma)
        actual_params = (actual.contents.alpha, actual.contents.beta,
                         actual.contents.mu_1, actual.contents.sigma)
        self.assertEqual(expect_params, actual_params)

    def test_k(self):
        path = 'csv-funding'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)
        n = self.get_n(path)

        expect_ks = self.get_ks(path)
        actual_ks = cfunding.k(a=df_params['alpha'], b=df_params['beta'],
                               mu=df_params['mu'], sig=df_params['sigma'],
                               n=n, alphas=nd_alphas)
        np_testing.assert_allclose(expect_ks, actual_ks)

    def test_nvalue_at_risk(self):
        path = 'csv-funding'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)

        alpha = nd_alphas[0]
        k = self.get_ks(path)[0]
        t = self.get_t(path)

        expect_vars = self.get_value_at_risks(path)
        actual_vars = np.array(cfunding.nvalue_at_risk(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sigma=df_params['sigma'],
            k_n=k, alpha=alpha, t=t))
        np_testing.assert_allclose(expect_vars, actual_vars)

    def test_nexpected_shortfall(self):
        path = 'csv-funding'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)

        alpha = nd_alphas[0]
        k = self.get_ks(path)[0]
        t = self.get_t(path)
        cp = self.get_cp(path)

        g_inv = np.log(1+cp)

        expect_ess = self.get_expected_shortfalls(path)
        actual_ess = np.array(cfunding.nexpected_shortfall(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sigma=df_params['sigma'],
            k_n=k, g_inv=g_inv, cp=cp, alpha=alpha,
            t=t))
        np_testing.assert_allclose(expect_ess, actual_ess)

    def test_nexpected_value(self):
        path = 'csv-funding'
        df_params = self.get_stable_params(path)

        k = self.get_ks(path)[0]
        t = self.get_t(path)
        cp = self.get_cp(path)

        g_inv_long = np.log(1+cp)
        g_inv_short = np.log(2)

        expect_evs = self.get_expected_values(path)
        actual_evs = np.array(cfunding.nexpected_value(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sigma=df_params['sigma'],
            k_n=k, g_inv_long=g_inv_long, cp=cp,
            g_inv_short=g_inv_short, t=t))
        np_testing.assert_allclose(expect_evs, actual_evs)
