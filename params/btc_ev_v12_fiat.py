#!/usr/bin/env python
# coding: utf-8

# Risk Parameters of BTC EV Fiat Market

import numpy as np
import pandas as pd
import plotly.express as px
pd.options.plotting.backend = "plotly"

import funding
import impact
import liquidations as liq
import pricedrift as drift
import pystable
from tqdm import tqdm


# Set data-specific parameters:

file_name = "/Users/fredericoteixeira/Projects/overlay/data/btc_ev_v12_fiat.csv"
periodicity = 1./funding.NS[0]  # converts 1 day in seconds
cap = 10  # cap on pay off, set by governance


# ## Understanding the Data

# Load and plot the data:

df = pd.read_csv(file_name).set_index("date").drop("created_at", axis=1)
df.plot()


# Compute `k`, the funding related risk metric:

ks, dst = funding.generic_get_ks(df["btc_ev_v12_fiat_index"].to_numpy(), periodicity)

df_ks = pd.DataFrame(
    data=ks,
    columns=[alpha for alpha in funding.ALPHAS],
    index=[n/funding.NS[0] for n in funding.NS]
)
df_ks.columns.name = "alpha"
df_ks.index.name = "days"

df_ks.plot()


# Draw the histogram of log returns of the index (this will be compared against the probability distribution later)

log_diff = np.log(df["btc_ev_v12_fiat_index"].div(df["btc_ev_v12_fiat_index"].shift(1)).dropna())
fig = px.histogram(log_diff, nbins=100)
fig.show()


# Plot the histogram of the fitted Stable distribution:

draws = pystable.rnd(dst, n=len(log_diff))
fig = px.histogram(np.sort(np.clip(draws, -0.2, 0.2)), nbins=100)
fig.show()

# ## Funding Rate parameters
# 
# What if we had started trading this index `3` years ago, and using all data available to compute the index? 
# 
# We look back the whole history of data for each day and compute the risk parameters.
# 
# For simplicity, we just collect results for $1$ day prediction.

rolling_window = 2 * 365  # 1 years
prediction = 0  # 1 day prediction is the 0-th row of the df_ks dataframe

df_ks_hist = pd.DataFrame(
    0.,
    columns=df_ks.columns,
    index=df.index[rolling_window:]
)
df_ks_hist.columns.name = df_ks.columns.name
df_ks_hist.index.name = "date"

df_dst_hist = pd.DataFrame(
    0.,
    columns=["alpha", "beta", "mu", "sigma"],
    index=df.index[rolling_window:]
)
df_dst_hist.columns.name = "parameters"
df_dst_hist.index.name = "date"

bar = tqdm(total=df_ks_hist.index.size)

for cur_idx in range(0, df_ks_hist.index.size):

    today = df_dst_hist.index[cur_idx]
    past_date = df.index[cur_idx]

    index_price = df.loc[past_date:today,"btc_ev_v12_fiat_index"].to_numpy()

    tmp_ks, tmp_dst = funding.generic_get_ks(p=index_price, periodicity=periodicity, verbose=False)
    
    df_ks_hist.loc[today, :] = np.array(tmp_ks[prediction])

    df_dst_hist.loc[today, :] = np.array(
        [dst.contents.alpha, dst.contents.beta, dst.contents.mu_1, dst.contents.sigma]
    )

    bar.update(1)

bar.close()

df_ks_hist["index"] = df.loc[df_ks_hist.index, "btc_ev_v12_fiat_index"].to_numpy()

fig = df_ks_hist.drop("index", axis=1).plot(title="Funding rate parameters")
fig.show()

fig = df_ks_hist.div(df_ks_hist.iloc[0,:]).plot(title="Normalized funding rate and index parameters")
fig.show()

def plot_df_ks_hist(alpha):
    return df_ks_hist.drop("index", axis=1).loc[:,alpha].plot(title=f"Funding rate with alpha = {alpha}")
