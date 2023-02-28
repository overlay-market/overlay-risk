# To run: brownie run check_chainlink_decimals --network arbitrum-mainnet

from brownie import Contract, chain, web3

link_feed = '0x52099d4523531f678dfc568a7b1e5038aadce1d6'
overlay_feed = '0xFA261B7c48A7426E00385F2435963005A7675df3'


def load_contract(address):
    try:
        return Contract(address)
    except ValueError:
        return Contract.from_explorer(address)


def get_block_timestamp(b):
    return web3.eth.get_block(b).timestamp


def get_latest_overlay_feed(ofeed, blk):
    _, _, _, micro_twap, _, _, _, _ = ofeed.latest(block_identifier=blk)
    return micro_twap


def get_latest_link_feed(lfeed, blk):
    round_id, link_spot, _, link_update_time, _ = \
        lfeed.latestRoundData(block_identifier=blk)
    return round_id, link_spot, link_update_time


def get_rid_link_feed(lfeed, rid):
    round_id, link_spot, _, link_update_time, _ = lfeed.getRoundData(rid)
    return round_id, link_spot, link_update_time


def main():
    ofeed = load_contract(overlay_feed)
    lfeed = load_contract(link_feed)

    blk_ht = chain.height - 1

    micro_twap = get_latest_overlay_feed(ofeed, blk_ht)
    rid, link_spot, link_update_time = get_latest_link_feed(lfeed, blk_ht)

    print(f'overlay_twap at block {blk_ht} is {micro_twap}')
    print(f'link_spot at block {blk_ht} is {link_spot},'
          f'updated at {link_update_time}')

    link_pi = []
    link_ti = []

    blk_ht_time = get_block_timestamp(blk_ht)
    micro_wind = 600

    while micro_wind > 0:

        _, link_spot, link_update_time = get_rid_link_feed(lfeed, rid)
        link_pi.append(link_spot)

        if blk_ht_time - link_update_time < micro_wind:
            dt = blk_ht_time - link_update_time
        else:
            dt = micro_wind

        micro_wind = micro_wind - dt

        link_ti.append(dt)
        rid = rid - 1

    print('Chainlink oracle Ti:', link_ti)
    print('Chainlink oracle Pi:', link_pi)

    result = 0
    for i in range(len(link_ti)):
        result += link_ti[i] * link_pi[i]

    twap = result/sum(link_ti)
    print('Chainlink TWAP = summation(Pi*Ti)/summation(Ti) =', twap)

    print('Chainlink TWAP after dividing by 8 decimals =', twap/1e8)
    print('Overlay TWAP after dividing by 18 decimals =', micro_twap/1e18)
