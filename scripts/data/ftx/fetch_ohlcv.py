import asyncio
import ccxt
import numpy as np

exchange = ccxt.ftx()
FILE = 'ethusd_01012020_08232021.csv'

# Python


async def main():
    if exchange.has['fetchOHLCV']:
        # since = exchange.milliseconds () - 86400000  # -1 day from now
        # alternatively, fetch from a certain starting datetime
        since = exchange.parse8601('2020-01-01T00:00:00Z')
        # alternatively, fetch until current time
        until = exchange.milliseconds()
        # until = exchange.parse8601('2020-08-01T00:00:00Z')
        with open(FILE, 'wb') as f:
            while since < until:
                print('since', since)
                symbol = "ETH/USD"  # change for your symbol
                limit = 1440  # change for your limit
                ohlcvs = exchange.fetch_ohlcv(
                    symbol=symbol, since=since, limit=limit)
                if len(ohlcvs):
                    print('ohlcvs[len(ohlcvs) - 1]', ohlcvs[len(ohlcvs) - 1])
                    since = ohlcvs[len(ohlcvs) - 1][0]
                    # all_ohlcvs += ohlcvs
                else:
                    break

                print('number of candles', len(ohlcvs))
                a = np.array(ohlcvs)
                np.savetxt(f, a, delimiter=',')

asyncio.run(main())
