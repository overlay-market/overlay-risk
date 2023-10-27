from brownie import accounts, Contract
import numpy as np
import json
from pytest import approx

OVL_ADDR = '0x738C250ef3e490acC17A8552A0BF11BabB29613c'
STATE_ADDR = '0xA9d9981974f0f6FB192275489053299B1b2502F6'
MARKET_ADDR = '0x436a0f8F1967CD7db6bEF6E9774e46b1d47f685D'
FEED_ADDR = '0xeE878326772Ca2886D363Ec08b459cdCE46B497A'
PARAMS = {
    "k": 115740740740,
    "lambda": 750000000000000000,
    "delta": 2475000000000000,
    "capPayoff": 5000000000000000000,
    "capNotional": 20000000000000000000000,
    "capLeverage": 10000000000000000000,
    "circuitBreakerWindow": 2592000,
    "circuitBreakerMintTarget": 1666666666666666666666,
    "maintenanceMarginFraction": 40000000000000000,
    "maintenanceMarginBurnRate": 50000000000000000,
    "liquidationFeeRate": 50000000000000000,
    "tradingFeeRate": 750000000000000,
    "minCollateral": 100000000000000,
    "priceDriftUpperLimit": 87000000000000,
    "averageBlockTime": 0
}


def load_contract(address, abi=None):
    if abi:
        with open(f'scripts/constants/{abi}.json') as f:
            abi = json.load(f)
        return Contract.from_abi('Contract', address, abi)
    else:
        try:
            return Contract(address)
        except ValueError:
            return Contract.from_explorer(address)


def get_abi(abi_name):
    with open(f'constants/{abi_name}.json') as f:
        return json.load(f)['abi']


def get_prices(market, state):
    return state.prices(market)


def get_latest_from_feed(feed):
    return feed.latest()


def get_ois(market, state):
    return state.ois(market)


def approve_all_ovl_to_market(ovl, acc, market):
    bal = ovl.balanceOf(acc)
    ovl.approve(market, bal, {'from': acc})


def build(market, is_long, acc):
    price_limit = 2**256-1 if is_long else 0
    tx = market.build(1e17, 1e18, is_long, price_limit, {'from': acc})
    pid = tx.events['Build']['positionId']
    return pid


def unwind(market, is_long, pid, acc):
    price_limit = 0 if is_long else 2**256-1
    market.unwind(pid, 1e18, price_limit, {'from': acc})


def test_static_spread(prices, latest_feed):
    (bid, ask, _) = prices
    (_, _, _, price_micro, price_macro, _, _, _) = latest_feed
    expect_bid = int(
        min(price_micro, price_macro) * np.exp(-PARAMS['delta']/1e18)
    )
    expect_ask = int(
        max(price_micro, price_macro) * np.exp(-PARAMS['delta']/1e18)
    )
    assert expect_bid == approx(bid, rel=1e4)
    assert expect_ask == approx(ask, rel=1e4)


def test_params_equal(market):
    assert market.params(0) == PARAMS['k']
    assert market.params(1) == PARAMS['lambda']
    assert market.params(2) == PARAMS['delta']
    assert market.params(3) == PARAMS['capPayoff']
    assert market.params(4) == PARAMS['capNotional']
    assert market.params(5) == PARAMS['capLeverage']
    assert market.params(6) == PARAMS['circuitBreakerWindow']
    assert market.params(7) == PARAMS['circuitBreakerMintTarget']
    assert market.params(8) == PARAMS['maintenanceMarginFraction']
    assert market.params(9) == PARAMS['maintenanceMarginBurnRate']
    assert market.params(10) == PARAMS['liquidationFeeRate']
    assert market.params(11) == PARAMS['tradingFeeRate']
    assert market.params(12) == PARAMS['minCollateral']
    assert market.params(13) == PARAMS['priceDriftUpperLimit']


def test_impact(market, state, prices):
    lmbd = PARAMS['lambda']
    (bid, ask, mid) = prices
    oi = (10e18/mid) * 1e18  # Test with position of notional size 10 OVL
    frac_oi = state.fractionOfCapOi(market, oi)
    vol_ask = state.volumeAsk(market, frac_oi)
    actual_ask_w_impact = state.ask(market, frac_oi)
    expected_ask_w_impact = int(ask * np.exp((lmbd/1e18) * (vol_ask/1e18)))
    assert expected_ask_w_impact == approx(actual_ask_w_impact, rel=1e4)

    vol_bid = state.volumeBid(market, frac_oi)
    actual_bid_w_impact = state.bid(market, frac_oi)
    expected_bid_w_impact = int(bid * np.exp((lmbd/1e18) * (vol_bid/1e18)))
    assert expected_bid_w_impact == approx(actual_bid_w_impact, rel=1e4)


def test_funding_rate(market, state):
    k = PARAMS['k']
    long_oi, short_oi = market.oiLong(), market.oiShort()
    imb_oi = abs(long_oi - short_oi) * (-1 if short_oi > long_oi else 1)
    total_oi = long_oi + short_oi
    actual_funding_rate = state.fundingRate(market)

    if total_oi == 0:
        assert actual_funding_rate == 0
    else:
        expected_funding_rate = 2 * k * imb_oi/total_oi
        assert expected_funding_rate == actual_funding_rate


def main(acc):
    acc = accounts.load(acc)
    market = load_contract(MARKET_ADDR, 'market')
    feed = load_contract(FEED_ADDR, 'feed')
    state = load_contract(STATE_ADDR, 'state')
    ovl = load_contract(OVL_ADDR, 'ovl')

    prices = get_prices(market, state)
    latest_feed = get_latest_from_feed(feed)

    print(f'Amount OVL held by testing account: {ovl.balanceOf(acc)/1e18}')
    print(f'Amount ETH held by testing account: {acc.balance()/1e18}')

    # Check if parameters input was correct while deploying contract
    test_params_equal(market)
    print('On-chain risk parameters as expected')

    # Check bid-ask static spread (delta)
    test_static_spread(prices, latest_feed)
    print('Static spread as expected')

    # Check impact (lambda)
    test_impact(market, state, prices)
    print('Market impact as expected')

    # Check funding rate
    test_funding_rate(market, state)
    print('Funding rate as expected')

    # Build position and check funding rate
    approve_all_ovl_to_market(ovl, acc, market)
    is_long = True
    pid = build(market, is_long, acc)
    test_funding_rate(market, state)
    print('Funding rate after building position as expected')

    # Unwind position and check funding rate
    unwind(market, is_long, pid, acc)
    test_funding_rate(market, state)
    print('Funding rate after unwinding position as expected')
