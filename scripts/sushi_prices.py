from brownie import Contract
from itertools import chain as ichain
import pandas as pd
import numpy as np
import time


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
    rng = list(range(lb, ub, 100_000))
    if rng[-1] < ub:
        rng.append(ub)
    return rng


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
