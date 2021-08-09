import time
import pandas as pd
import argparse
import os
import typing as tp
import logging

from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


def get_config_source() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN_SOURCE'),
        "org": os.getenv('INFLUXDB_ORG_SOURCE'),
        "url": os.getenv("INFLUXDB_URL_SOURCE"),
    }


def get_config_dest() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_source_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'],
                          debug=False)


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-sushi", "ingest-data-frame")
    return point_settings


def get_params() -> tp.Dict:
    points = 30  # Default to 30d in past for influx start range
    source = "ovl_sushi"  # Default to ovl_sushi InfluxDB bucket
    dest = "ovl_sushi"  # Default to ovl_sushi InfluxDB bucket
    symb_flag = 1

    parser = argparse.ArgumentParser(
        description='Fetch from source bucket and transfer to destination'
    )
    parser.add_argument(
        '--source', type=str,
        help='name of the influx bucket to query'
    )
    parser.add_argument(
        '--points', type=int,
        help='number of days in the past to use as start range of influx query'
    )
    parser.add_argument(
        '--destination', type=str,
        help='name of the destination influx bucket'
    )
    parser.add_argument(
        '--fix_symbols', type=int,
        help='flag to indicate if symbols should be flipped to correct order'
    )

    args = parser.parse_args()
    if args.points:
        points = args.points
    if args.source:
        source = args.source
    if args.destination:
        dest = args.destination
    if args.fix_symbols:
        symb_flag = args.fix_symbols

    return {
        "points": points,
        "source": source,
        "dest": dest,
        "symb_flag": symb_flag
    }


def get_data(query_api, cfg: tp.Dict, p: tp.Dict) -> pd.DataFrame:
    points = p['points']
    bucket = p['source']
    org = cfg['org']

    print(f'Fetching data from {bucket} for the last {points} days ...')
    query = f'''
        from(bucket:"{bucket}") |> range(start: -{points}d)
    '''
    df = query_api.query_data_frame(query=query, org=org)
    if type(df) == list:
        df = pd.concat(df, ignore_index=True)
    return df


def pivot_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(labels=['result', 'table', '_start', '_stop',
            '_measurement', 'influx-sushi', 'type'], inplace=True, axis=1)

    df = df.pivot(
        index=['id', '_time', 'token0_name', 'token1_name'],
        columns=['_field'], values='_value'
        ).reset_index()

    df = df.rename_axis(None, axis=1).reset_index(drop=True)
    df = df.set_index("_time")

    return df


def symbol_fix(df: pd.DataFrame, symb_flag: int) -> pd.DataFrame:
    if symb_flag == 1:
        df.loc[df.id == 'Sushiswap: WETH / WBTC', 'token0_name'] = 'WBTC'
        df.loc[df.id == 'Sushiswap: WETH / WBTC', 'token1_name'] = 'WETH'

        df.loc[df.id == 'Sushiswap: WETH / USDC', 'token0_name'] = 'USDC'
        df.loc[df.id == 'Sushiswap: WETH / USDC', 'token1_name'] = 'WETH'

        df.loc[df.id == 'Sushiswap: WETH / DAI', 'token0_name'] = 'DAI'
        df.loc[df.id == 'Sushiswap: WETH / DAI', 'token1_name'] = 'WETH'

        df.loc[df.id == 'Sushiswap: CRV / WETH', 'token0_name'] = 'WETH'
        df.loc[df.id == 'Sushiswap: CRV / WETH', 'token1_name'] = 'CRV'

        df.loc[df.id == 'Sushiswap: ALCX / WETH', 'token0_name'] = 'WETH'
        df.loc[df.id == 'Sushiswap: ALCX / WETH', 'token1_name'] = 'ALCX'
    return df


def main():
    """
    Fetch from an influx bucket and write data to destination influx bucket
    """

    config = get_config_source()
    config_dest = get_config_dest()
    params = get_params()
    source_client = create_source_client(config)
    query_api = source_client.query_api()

    try:
        df = get_data(query_api, config, params)
    except Exception as e:
        print("Failed to retrieve data from influx")
        logging.exception(e)

    df = pivot_data(df)
    df = symbol_fix(df, params['symb_flag'])

    stp = 2000
    for i in range(0, df.shape[0], stp):
        with InfluxDBClient(
                url=config_dest['url'],
                token=config_dest['token'],
                org=config_dest['org']
                ) as client:

            f"""
            Ingest DataFrame with default tags
            Rows: {i} to {i+stp-1}
            """
            write_api = client.write_api(
                write_options=SYNCHRONOUS, point_settings=get_point_settings())

            j = 1
            while j == 1:
                try:
                    write_api.write(bucket=params['dest'],
                                    record=df[i:i+stp-1],
                                    data_frame_measurement_name="mem",
                                    data_frame_tag_columns=[
                                        'id', 'token0_name', 'token1_name'
                                        ]
                                    )
                    j = 0
                except Exception as e:
                    err_cls = e.__class__
                    err_msg = str(e)
                    msg = f'''
                    Error type = {err_cls}
                    Error message = {err_msg}
                    Wait 5 secs
                    '''
                    print(msg)
                    continue

        print("Wait 15 secs to push data")
        time.sleep(15)
        client.close()
        print("Wait 5 secs to not exceed max write limit")
        time.sleep(5)


if __name__ == '__main__':
    main()
