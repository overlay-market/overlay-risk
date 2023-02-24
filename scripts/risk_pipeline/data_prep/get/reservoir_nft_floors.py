import requests
import pandas as pd
import argparse
import time
import os
import sys
sys.path.insert(0, os.getcwd()+'/scripts/risk_pipeline')
from helpers import helpers  # noqa: E402

URL = "https://api.reservoir.tools/events/collections/floor-ask/v1"


def epoch_time(ts_string):
    time_struct = time.strptime(ts_string, "%Y-%m-%d-%H-%M")

    # Convert the time struct to Unix time and return
    return time.mktime(time_struct)


def get_params():
    """
    Get parameters from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--collection_addr', type=str,
        help=('Contract address of the collection.'
              'Ex, for BAYC: 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D')
    )
    parser.add_argument(
        '--collection_name', type=str,
        help='Name of the collection. Ex: BAYC'
    )
    parser.add_argument(
        '--start_time', type=str,
        help='Start time for data pull. Ex: 2016-06-01-00-00'
    )
    parser.add_argument(
        '--end_time', type=str,
        help='End time for data pull. Ex: 2016-06-01-00-00'
    )

    args = parser.parse_args()
    start_time = epoch_time(args.start_time)
    end_time = epoch_time(args.end_time)

    return args.collection_addr, args.collection_name, start_time, end_time


def get_file_name(cname, start, end):
    return f"{cname}_{start}_{end}.csv"


def query_data(caddr, start, end):

    # Set query parameters
    params = {
        'collection': caddr.lower(),
        'startTimestamp': start,
        'endTimestamp': end,
        'limit': 1000
    }

    # Pull data
    response = requests.get(URL, params)
    response_dicts = [response.json()]
    i = 0
    while response_dicts[i]['continuation']:
        print(i)
        params['continuation'] = response_dicts[0]['continuation']
        response = requests.get(URL, params)
        response_dicts.append(response.json())
        i += 1

    # Arrange data in pandas df
    dfs = [pd.json_normalize(d['events']) for d in response_dicts]
    df = pd.concat(dfs, ignore_index=True)

    price_df = df[['floorAsk.price', 'event.createdAt']]
    price_df.columns = ['close', 'time']
    price_df.time = pd.to_datetime(price_df.time)
    return price_df


def get_data(caddr, cname, start, end):
    file_name = get_file_name(cname, start, end)
    file_path = os.getcwd() + '/scripts/risk_pipeline/outputs/data/'
    full_path = file_path + file_name
    if helpers.file_exists(full_path):
        print(f"Data {file_name} already exists")
        return pd.read_csv(full_path), full_path
    else:
        df = query_data(caddr, start, end)
        helpers.csv(df, full_path)
        return df, full_path


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        caddr, cname, start, end = get_params()
        get_data(caddr, cname, start, end)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
