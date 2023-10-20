from brownie import Contract, chain, web3
import matplotlib.pyplot as plt
import concurrent.futures
import itertools
import pandas as pd
import numpy as np
import json


FEED_NAME = 'CSGO'
UB = chain.height
# LB = UB - 5 * (1/0.25) * 86400  # 30 days (~0.25 sec/block on arb)
LB = 49065370


def json_load(name):
    f = open(f'scripts/constants/{name}.json')
    return json.load(f)


def get_abi_from_file():
    with open('scripts/constants/abi.json') as f:
        abi = json.load(f)
    return abi


def load_contract(address):
    cont = Contract.from_abi('Feed', address, abi=get_abi_from_file())
    return cont


def get_block_ranges(lb, ub):
    step = 250 * 60
    rng = list(range(lb, ub, step))
    if rng[-1] < ub:
        rng.append(ub)
    return rng


def get_args_df(event_list):
    args = []
    for i in range(len(event_list)):
        args.append(event_list[i].args)
    return pd.DataFrame(args)


def get_block_timestamp(b):
    return web3.eth.get_block(b).timestamp


def get_timestamps(block_numbers):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        timestamps = list(executor.map(get_block_timestamp, block_numbers))
    return timestamps


def get_event_df(event_list):
    args_df = get_args_df(event_list)
    event_df = pd.DataFrame(
        event_list,
        columns=['address', 'event', 'logIndex',
                 'transactionHash', 'blockNumber']
    )
    return event_df.join(args_df)


def main(pool_addr, lb=LB, ub=UB):
    lb = int(lb)
    ub = int(ub)
    # Load feed contract
    feed = load_contract(pool_addr)

    # Get events in batches (single pull might result in timeout error)
    pool_events_l = []
    rng = get_block_ranges(lb, ub)
    for i in range(len(rng)):
        print(f'{np.round((i/len(rng)) * 100, 2)}% complete', end="\r")
        if i == len(rng)-1:
            break
        pool_events_l.append(
            feed.events.get_sequence(
                event_type='SubmissionReceived',
                from_block=rng[i]+1,
                to_block=rng[i+1])
        )
    # Flatter in pool_events_l
    pool_events = list(itertools.chain.from_iterable(pool_events_l))
    df = get_event_df(pool_events)

    # Add timestamps to every nth block
    step = 1
    block_numbers_to_fetch = [b if i % step == 0 else None for i, b in enumerate(df['blockNumber'])]
    block_numbers_to_fetch = [b for b in block_numbers_to_fetch if b is not None]
    timestamps = get_timestamps(block_numbers_to_fetch)

    timestamp_dict = dict(zip(block_numbers_to_fetch, timestamps))
    df['timestamp'] = [timestamp_dict.get(b, np.nan) for b in df['blockNumber']]

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Remove NaNs
    df2 = df.dropna(subset=['timestamp'])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot_date(df2['timestamp'], df2['submission'], linestyle='solid')
    plt.title('Answer over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Answer')
    plt.grid(True)
    plt.show()
