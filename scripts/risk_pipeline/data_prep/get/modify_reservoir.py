import pandas as pd


def main():
    df = pd.read_csv(
        'scripts/risk_pipeline/outputs/data/BAYC_1654021800_1678386600.csv'
    )
    df_eth = df[(df.currency == 'Ether') | (df.currency == 'Wrapped Ether')]
    df_eth['chain_id'] = 1
    df_eth['token_id'] = 10001
    df_eth.time = pd.to_datetime(df_eth.time)
    df_eth['unix_timestamp'] = \
        (df_eth['time'].astype('int')/10**9).astype('int')
    df_eth['ds'] = df_eth.time.dt.date
    df_eth = df_eth[['chain_id', 'collection', 'token_id', 'ds',
                     'block_number', 'unix_timestamp', 'price']]
    df_eth.columns = ['chain_id', 'contract_address', 'token_id', 'ds',
                      'block_number', 'unix_timestamp', 'price_eth']
    df_nft = pd.read_csv(
        '/Users/anantdeepparihar/Overlay/coinbase-nft-floor-price'
        '/nft_trades.csv')
    df_fin = df_nft.append(df_eth)
    df_fin.reset_index(inplace=True)
    df_fin.drop('index', axis=1, inplace=True)
    df_fin.to_csv('/Users/anantdeepparihar/Overlay/coinbase-nft-floor-price/'
                  'nft_trades_v2.csv')


if __name__ == '__main__':
    main()
