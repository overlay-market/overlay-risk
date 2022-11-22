from brownie import Contract
from itertools import chain as ichain
import pandas as pd
import numpy as np
import time


def load_contract(address):
    try:
        return Contract(address)
    except ValueError:
        return Contract.from_explorer(address)


def main(pool_addr):
    # Load contracts
    pool = load_contract(pool_addr)

    # Get events and build pandas df
    pool_events = pool.events.get_sequence(from_block=15691468)
