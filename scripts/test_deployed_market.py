from brownie import accounts, Contract

STATE_ADDR = '0xC3cB99652111e7828f38544E3e94c714D8F9a51a'
MARKET_ADDR = '0xa811698d855153cc7472D1FB356149a94bD618e7'
OVL_ADDR = '0x4305C4Bc521B052F17d389c2Fe9d37caBeB70d54'
PARAMS = {
    'k': 214443204786,
    'lambda': 725673770858147328,
    'delta': 3813581704489073,
    'capPayoff': 10000000000000000000,
    'capNotional': 800000000000000000000000,
    'capLeverage': 5000000000000000000,
    'circuitBreakerWindow': 2592000,
    'circuitBreakerMintTarget': 66670000000000000000000,
    'maintenanceMarginFraction': 72427569355569376,
    'maintenanceMarginBurnRate': 53946782638906936,
    'liquidationFeeRate': 50000000000000000,
    'tradingFeeRate': 750000000000000,
    'minCollateral': 100000000000000,
    'priceDriftUpperLimit': 100000000000000
}


def load_contract(address):
    try:
        return Contract(address)
    except ValueError:
        return Contract.from_explorer(address)


def main(acc):
    acc = accounts.load(acc)
    market = load_contract(MARKET_ADDR)
    state = load_contract(STATE_ADDR)
    ovl = load_contract(OVL_ADDR)

    print(f'Amount OVL held by testing account: {ovl.balanceOf(acc)/1e18}')
    