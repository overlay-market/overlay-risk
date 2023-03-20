import pandas as pd
import os
import pystable
import numpy as np
import numpy.testing as np_testing
from scripts.risk_pipeline.risk.parameters import csv_impact as cimpact
import unittest


class TestCsvImpact(unittest.TestCase):

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

    def get_q0s(self, path) -> np.ndarray:
        '''
        Helper to return q0s to use in all calculations
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-q0s.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_delta_longs(self, path) -> np.ndarray:
        '''
        Helper to return expected delta longs for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-delta-longs.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_delta_shorts(self, path) -> np.ndarray:
        '''
        Helper to return expected delta shorts for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-delta-shorts.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_deltas(self, path) -> np.ndarray:
        '''
        Helper to return expected deltas for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)

        # long
        base_l = os.path.join(base, 'get-delta-longs.csv')
        df_l = pd.read_csv(base_l, sep=',')

        # short
        base_s = os.path.join(base, 'get-delta-shorts.csv')
        df_s = pd.read_csv(base_s, sep=',')

        return np.maximum(df_l.values[0], df_s.values[0])

    def get_lmbda_longs(self, path) -> np.ndarray:
        '''
        Helper to return expected lmbda longs for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-lmbda-longs.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_lmbda_shorts(self, path) -> np.ndarray:
        '''
        Helper to return expected lmbda shorts for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)
        base = os.path.join(base, 'get-lmbda-shorts.csv')

        df = pd.read_csv(base, sep=',')
        return df.values[0]

    def get_lmbdas(self, path) -> np.ndarray:
        '''
        Helper to return expected lmbdas for stable params
        and uncertainty files
        '''
        base = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(base, os.pardir))
        base = os.path.join(base, 'helpers')
        base = os.path.join(base, path)

        # long
        base_l = os.path.join(base, 'get-lmbda-longs.csv')
        df_l = pd.read_csv(base_l, sep=',')

        # short
        base_s = os.path.join(base, 'get-lmbda-shorts.csv')
        df_s = pd.read_csv(base_s, sep=',')

        return np.maximum(df_l.values[0], df_s.values[0])

    def test_gaussian(self):
        expect = pystable.create(alpha=2.0, beta=0.0, mu=0.0, sigma=1.0,
                                 parameterization=1)
        actual = cimpact.gaussian()
        self.assertIsInstance(actual, type(expect))

        expect_params = (expect.contents.alpha, expect.contents.beta,
                         expect.contents.mu_1, expect.contents.sigma)
        actual_params = (actual.contents.alpha, actual.contents.beta,
                         actual.contents.mu_1, actual.contents.sigma)
        self.assertEqual(expect_params, actual_params)

    def test_rescale(self):
        path = 'csv-impact'
        df_params = self.get_stable_params(path)
        v = self.get_v(path)

        dist = pystable.create(
            alpha=df_params['alpha'], beta=df_params['beta'],
            mu=df_params['mu'], sigma=df_params['sigma'], parameterization=1)

        expect = pystable.create(
            alpha=df_params['alpha'], beta=df_params['beta'],
            mu=df_params['mu']*v,
            sigma=df_params['sigma']*v**(1/df_params['alpha']),
            parameterization=1)
        actual = cimpact.rescale(dist, v)
        self.assertIsInstance(actual, type(expect))

        expect_params = (expect.contents.alpha, expect.contents.beta,
                         expect.contents.mu_1, expect.contents.sigma)
        actual_params = (actual.contents.alpha, actual.contents.beta,
                         actual.contents.mu_1, actual.contents.sigma)
        self.assertEqual(expect_params, actual_params)

    def test_delta_long(self):
        path = 'csv-impact'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)
        v = self.get_v(path)

        expect_delta_ls = self.get_delta_longs(path)
        actual_delta_ls = cimpact.delta_long(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sig=df_params['sigma'], v=v,
            alphas=nd_alphas)

        np_testing.assert_allclose(expect_delta_ls, actual_delta_ls)

    def test_delta_short(self):
        path = 'csv-impact'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)
        v = self.get_v(path)

        expect_delta_ss = self.get_delta_shorts(path)
        actual_delta_ss = cimpact.delta_short(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sig=df_params['sigma'], v=v,
            alphas=nd_alphas)

        np_testing.assert_allclose(expect_delta_ss, actual_delta_ss)

    def test_delta(self):
        path = 'csv-impact'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)
        v = self.get_v(path)

        expect_deltas = self.get_deltas(path)
        actual_deltas = cimpact.delta(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sig=df_params['sigma'], v=v,
            alphas=nd_alphas)

        np_testing.assert_allclose(expect_deltas, actual_deltas)

    def test_lmbda_long(self):
        path = 'csv-impact'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)

        alpha = nd_alphas[0]
        v = self.get_v(path)
        cp = self.get_cp(path)
        q0s = self.get_q0s(path)

        g_inv = np.log(1+cp)

        expect_lmbda_ls = self.get_lmbda_longs(path)
        actual_lmbda_ls = cimpact.lmbda_long(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sig=df_params['sigma'],
            g_inv=g_inv, v=v, alpha=alpha, q0s=q0s)

        np_testing.assert_allclose(expect_lmbda_ls, actual_lmbda_ls)

    def test_lmbda_short(self):
        path = 'csv-impact'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)

        alpha = nd_alphas[0]
        v = self.get_v(path)
        q0s = self.get_q0s(path)

        expect_lmbda_ss = self.get_lmbda_shorts(path)
        actual_lmbda_ss = cimpact.lmbda_short(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sig=df_params['sigma'],
            v=v, alpha=alpha, q0s=q0s)

        np_testing.assert_allclose(expect_lmbda_ss, actual_lmbda_ss)

    def test_lmbda(self):
        path = 'csv-impact'
        df_params = self.get_stable_params(path)
        nd_alphas = self.get_alphas(path)

        alpha = nd_alphas[0]
        v = self.get_v(path)
        cp = self.get_cp(path)
        q0s = self.get_q0s(path)

        g_inv = np.log(1+cp)

        expect_lmbdas = self.get_lmbdas(path)
        actual_lmbdas = cimpact.lmbda(
            a=df_params['alpha'], b=df_params['beta'],
            mu=df_params['mu'], sig=df_params['sigma'],
            g_inv=g_inv, v=v, alpha=alpha, q0s=q0s)

        np_testing.assert_allclose(expect_lmbdas, actual_lmbdas)
