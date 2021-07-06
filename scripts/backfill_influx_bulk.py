import pandas as pd
import sys
import numpy as np
import os
import json
import typing as tp
import logging

from datetime import datetime
from scipy.stats import norm

from influxdb_client import InfluxDBClient, Point, WritePrecision, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings

from collections import OrderedDict
from csv import DictReader

import rx
from rx import operators as ops

import time

# init parameters
bucket = "ovl_sushi"

# pull from mikey's bucket
file = open('mikey_influx.txt', 'r')
creds = json.load(file)
org = creds['org']
url = creds['url']
token = creds['token']
file.close()

client = InfluxDBClient(url=url, token=token, debug=False)
query_api = client.query_api()

query = f'''
    from(bucket:"{bucket}") |> range(start: -30d)
        '''

df = query_api.query_data_frame(query=query, org=org)
df.shape
# df = df[df['_time'] > "2021-06-22T18:50:00Z"]



df.drop(labels = ['result', 'table', '_start', '_stop', '_measurement', 'influx-sushi', 'type'], inplace = True, axis = 1)
df = df.pivot(index = ['id', '_time', 'token0_name', 'token1_name'], columns = ['_field'], values = '_value').reset_index()
df = df.rename_axis(None, axis=1).reset_index(drop=True)

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

df = df.set_index("_time")


file = open('deep_influx.txt', 'r')
creds = json.load(file)
org = creds['org']
url = creds['url']
token = creds['token']
file.close()



stp = 2000
for i in range(0, df.shape[0], stp):
    with InfluxDBClient(url=url, token=token, org=org) as client:
        f"""
        Ingest DataFrame with default tags
        Rows: {i} to {i+stp-1}
        """
        point_settings = PointSettings(**{"type": "metrics-hourly"})
        point_settings.add_default_tag("influx-sushi", "ingest-data-frame")

        write_api = client.write_api(write_options=SYNCHRONOUS, point_settings=point_settings)

        j = 1
        while j == 1:
            try:
                write_api.write(bucket=bucket, record=df[i:i+stp-1], data_frame_measurement_name="mem",\
                    data_frame_tag_columns=['id', 'token0_name', 'token1_name'])
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