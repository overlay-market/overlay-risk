import numpy as np


def bid(min_twap, delta, lmb, vol):
    return min_twap * np.exp(-delta - (lmb*vol))


def ask(min_twap, delta, lmb, vol):
    return min_twap * np.exp(delta + (lmb*vol))


def make_numeric(df, pre, col):
    df[col] = df[col].apply(lambda x: float(x.replace(pre, '')))
    return df


def bid_ask_perc_change(df, lmbd, vol, dlt=0, twap=100):
    df['bid'] = df.apply(
        lambda x: bid(twap, dlt, x[lmbd], x[vol]), axis=1
    )
    df['ask'] = df.apply(
        lambda x: ask(twap, dlt, x[lmbd], x[vol]), axis=1
    )

    df['bid_perc'] = (abs(df.bid-twap)/twap) * 100
    df['ask_perc'] = (abs(df.ask-twap)/twap) * 100
    return df
