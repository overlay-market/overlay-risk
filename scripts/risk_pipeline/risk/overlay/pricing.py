import numpy as np


def bid(min_twap, delta, lmb, vol):
    return min_twap * np.exp(-delta - (lmb*vol))


def ask(min_twap, delta, lmb, vol):
    return min_twap * np.exp(delta + (lmb*vol))
