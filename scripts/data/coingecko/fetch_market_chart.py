from pycoingecko import CoinGeckoAPI
import numpy as np


cg = CoinGeckoAPI()
FILE = 'coingecko_ethusd_01012020_08232021.csv'
ID = "ethereum"
VS = "usd"
SINCE = 1577854800  # Jan 1, 2020
UNTIL = 1635206696  # Oct 25, 2021
LIMIT = 86400*60

# NOTE from CoinGecko API docs:
# Minutely data will be used for duration within 1 day,
# Hourly data will be used for duration between 1 day and 90 days,
# Daily data will be used for duration above 90 days.


def main():
    since = SINCE
    until = UNTIL
    limit = LIMIT
    print('until', until)
    print('limit', limit)

    if limit <= 86400:
        candles = '1m'
    elif limit > 86400 and limit < 90*86400:
        candles = '1h'
    else:
        candles = '1d'

    print('candles', candles)
    with open(FILE, 'wb') as f:
        while since < until:
            print('since', since)
            resp = cg.get_coin_market_chart_range_by_id(
                id=ID,
                vs_currency=VS,
                from_timestamp=since,
                to_timestamp=since+limit,
            )
            prices = resp['prices']
            if len(prices):
                print('prices[len(prices) - 1]', prices[len(prices) - 1])
                since = prices[len(prices) - 1][0]/1000.0
                print('new since', since)
            else:
                break

            print('number of candles', len(prices))
            a = np.array(prices)
            np.savetxt(f, a, delimiter=',')


if __name__ == '__main__':
    main()
