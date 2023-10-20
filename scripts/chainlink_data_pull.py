from brownie import Contract, chain, web3
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


def main(pool_addr, lb=LB, ub=UB):
    lb = int(lb)
    ub = int(ub)
    # Load feed contract
    feed = load_contract(pool_addr)

    pool_events_l = []
    feed.events.get_sequence(
                event_type='SubmissionReceived',
                from_block=LB,
                to_block=UB
        )
    # Flatten in pool_events_l
    pool_events = list(itertools.chain.from_iterable(pool_events_l))