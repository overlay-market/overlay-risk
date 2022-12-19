from brownie import Contract
import pandas as pd
import numpy as np
import json


POOL_NAME = 'ETH-USDC'


def split_args(args):
    l_arg = args.split(',')
    pool_addr = l_arg[0].strip()
    lb = int(l_arg[1].strip())
    ub = int(l_arg[2].strip())
    return pool_addr, lb, ub


def json_load(name):
    f = open(f'scripts/constants/{name}.json')
    return json.load(f)


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


def main(args):
    # Get args
    pool_addr, lb, ub = split_args(args)

    # Load contracts and token decimals info
    abi = json_load('univ3.abi')
    pool = Contract.from_abi('pool', pool_addr, abi)
    decimals_0 = load_contract(pool.token0()).decimals()
    decimals_1 = load_contract(pool.token1()).decimals()

    # Get events in batches (single pull might result in timeout error)
    pool_events_l = []
    rng = get_block_ranges(lb, ub)
    for i in range(len(rng)):
        print(f'{np.round((i/len(rng)) * 100, 2)}% complete', end="\r")
        if i == len(rng)-1:
            break
        pool_events_l.append(
            pool.events.get_sequence(
                event_type='Swap',
                from_block=rng[i]+1,
                to_block=rng[i+1])
        )

    # Make pandas df out of events dict
    swap_l = []
    for i in range(len(pool_events_l)):
        swap_l.append(get_event_df(pool_events_l[0],
                                   ['logIndex',
                                    'transactionHash', 'blockNumber']))
    swap_df = pd.concat(swap_l)

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
    df = swap_df.merge(block_df, on='blockNumber', how='inner')
    df['close'] =\
        ((df.sqrtPriceX96**2) * (10**(decimals_0-decimals_1))) / (2**(96*2))

    # There are multiple swaps within a block and all affect price.
    # Keep only the last swap. That price should only be associated with
    # the timestamp of that block
    last_swap = df[['blockNumber', 'logIndex']].groupby(['blockNumber']).max()
    df = df.merge(last_swap, on=['blockNumber', 'logIndex'], how='inner')

    # Save data
    df = df[['close', 'timestamp']]
    df.to_csv(f'csv/{POOL_NAME}-SPOT-check.csv')

    # Get 10m TWAP at 10m periodicity
    freq_df = df.set_index('timestamp')
    freq_df = freq_df.asfreq('10min').reset_index()
    freq_df['flag'] = 1
    df['flag'] = 0
    combine_df = df.append(freq_df)
    combine_df = combine_df.groupby('timestamp').max()
    close_df = combine_df.bfill()['close']
    close_df = close_df.rolling('600s', min_periods=1).mean()
    close_df = pd.DataFrame(close_df).join(combine_df['flag'])
    close_df.reset_index(inplace=True)
    close_df = close_df[close_df.flag == 1]
    close_df.drop(['flag'], axis=1, inplace=True)
    close_df.columns = ['timestamp', 'twap']
    close_df.to_csv(f'csv/{POOL_NAME}-10mTWAP-check.csv')