from brownie import Contract
import pandas as pd
import numpy as np
from brownie import web3
from concurrent.futures import ThreadPoolExecutor


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


def get_block_timestamp(b):
    return web3.eth.get_block(b).timestamp


def main(args):
    # Get args
    pool_addr, lb, ub = split_args(args)

    # Load contracts
    pool = load_contract(pool_addr)

    # Get events and build pandas df
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

    sync_l = []
    for i in range(len(pool_events_l)):
        sync_l.append(get_event_df(pool_events_l[i].Sync,
                                   ['logIndex',
                                    'transactionHash', 'blockNumber']))
    sync_df = pd.concat(sync_l)
    time_l = list(sync_df['blockNumber'])
    time_f = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for item in executor.map(get_block_timestamp, time_l):
            print(item)
            time_f.append(item)

