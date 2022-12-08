from brownie import Contract
import pandas as pd
import numpy as np


def split_args(args):
    l_arg = args.split(',')
    pool_addr = l_arg[0].strip()
    lb = int(l_arg[1].strip())
    ub = int(l_arg[2].strip())
    return pool_addr, lb, ub


def load_contract(address):
    try:
        return Contract(address)
    except ValueError:
        return Contract.from_explorer(address)


def get_block_ranges(lb, ub):
    step = 100_000
    rng = list(range(lb, ub, step))
    if rng[-1] < ub:
        rng.append(ub)
    return rng


def get_args_df(event_list):
    args = []
    for i in range(len(event_list)):
        args.append(event_list[i].args)
    return pd.DataFrame(args)


def get_event_df(event_list, cols):
    args_df = get_args_df(event_list)
    event_df = pd.DataFrame(event_list, columns=cols)
    return event_df.join(args_df)


def dynamic_window(
        df: pd.DataFrame,
        max_rows: int,
        window: int
        ) -> pd.DataFrame:
    '''
    Computes the window size in terms of rows such that there is as much data
    as there are seconds specified in the `window` variable.
    '''

    for i in range(1, int(max_rows+1)):
        df.loc[:, 'lag_time'] = df.loc[:, '_time'].shift(i)
        df.loc[:, i] =\
            (pd.to_datetime(df.loc[:, '_time'])
             - pd.to_datetime(df.loc[:, 'lag_time']))\
            .dt.total_seconds()
        df.loc[:, i] = abs(df.loc[:, i] - (window * 60))

        df.drop(['lag_time'], axis=1, inplace=True)

    min_df = df[[i for i in range(1, int(max_rows+1))]]\
        .idxmin(axis="columns")

    df.dropna(inplace=True)
    df = df.join(pd.DataFrame(min_df, columns=['dynamic_window']))
    df['dynamic_window'] = df['dynamic_window'].astype(int)
    return df


def delta_window(
        row: pd.Series,
        values: pd.Series,
        lookback: pd.Series
        ) -> pd.Series:
    '''
    Computes difference based on window sizes specified in `lookback`
    '''

    loc = values.index.get_loc(row.name)
    lb = lookback.loc[row.name]
    return values.iloc[loc] - values.iloc[loc-lb]


def main(args):
    # Get args
    pool_addr, lb, ub = split_args(args)

    # Load contracts
    pool = load_contract(pool_addr)

    # Get events in batches (single pull might result in timeout error)
    pool_events_l = []
    rng = get_block_ranges(lb, ub)
    for i in range(len(rng)):
        print(f'{np.round((i/len(rng)) * 100, 2)}% complete', end="\r")
        if i == len(rng)-1:
            break
        pool_events_l.append(
            pool.events.get_sequence(
                from_block=rng[i]+1,
                to_block=rng[i+1])
        )

    # Make pandas df out of events dict
    sync_l = []
    for i in range(len(pool_events_l)):
        sync_l.append(get_event_df(pool_events_l[i].Sync,
                                   ['logIndex',
                                    'transactionHash', 'blockNumber']))
    sync_df = pd.concat(sync_l)

    # Get timestamps wrt block number.
    # Using raw data pulled from bigquery for ethereum (google).
    # Because using web3py API calls was too slow even with multithreading.
    block_df = pd.read_csv(
        'csv/block_to_timestamp-20221123-141917-1669213222210.csv')
    block_df.timestamp = block_df.timestamp.str.replace(' UTC', '')
    block_df.timestamp = pd.to_datetime(block_df.timestamp,
                                        format='%Y-%m-%d %H:%M:%S')
    block_df.columns = ['timestamp', 'blockNumber']

    # Get block timestamps
    df = sync_df.merge(block_df, on='blockNumber', how='inner')
    df['close'] = df.reserve1/df.reserve0

    # There are multiple swaps within a block and all affect price.
    # Keep only the last swap. That price should only be associated with
    # the timestamp of that block
    last_swap = df[['blockNumber', 'logIndex']].groupby(['blockNumber']).max()
    df = df.merge(last_swap, on=['blockNumber', 'logIndex'], how='inner')

    # Save data
    df.to_csv('csv/NFTX-WETH-SPOT-20210115-20210630.csv')

    # Get 10m TWAP
    close_df = df[['close', 'timestamp']]
    close_df.set_index('timestamp', inplace=True)
    close_df = close_df.rolling('600s', min_periods=1).mean()
    close_df.reset_index(inplace=True)
    close_df.columns = ['timestamp', 'twap']
    df = df.merge(close_df, on='timestamp', how='inner')
    df.to_csv('csv/NFTX-WETH-10mTWAP-20210115-20210630.csv')
