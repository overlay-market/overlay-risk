import requests
import pandas as pd
import argparse
import time
import os
import sys
sys.path.insert(0, os.getcwd()+'/scripts/risk_pipeline')
from helpers import helpers  # noqa: E402

URL = "https://api.reservoir.tools/sales/v4"


def epoch_time(ts_string):
    time_struct = time.strptime(ts_string, "%Y-%m-%d-%H-%M")

    # Convert the time struct to Unix time and return
    return int(time.mktime(time_struct))


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
        if response_dicts[-1] == response_dicts[-2]:
            response_dicts = response_dicts[:-1]
            break
        i += 1

    # Arrange data in pandas df
    dfs = [pd.json_normalize(d['sales']) for d in response_dicts]
    df = pd.concat(dfs, ignore_index=True)
    # TODO: Catch when no data is returned
    price_df = df[['price.currency.name', 'price.netAmount.native',
                   'timestamp', 'token.contract']]
    price_df.columns = ['currency', 'price', 'time', 'collection']
    price_df.time = pd.to_datetime(price_df.time, unit='s')
    return price_df


def get_ranges(start, end):
    step = 120 * 86400  # 120 days
    if end - start < step:
        return [(start, end)]
    else:
        steps = list(range(start, end, step))
        if steps[-1] != end:
            steps.append(end)  # Append last timestamp if `range` left it out
        time_range = []
        for i in range(len(steps)-1):
            time_range.append((steps[i]+1, steps[i+1]))
        time_range.reverse()
        return time_range


def get_data(caddr, cname, start, end):
    file_name = get_file_name(cname, start, end)
    file_path = os.getcwd() + '/scripts/risk_pipeline/outputs/data/'
    full_path = file_path + file_name
    if helpers.file_exists(full_path):
        print(f"Data {file_name} already exists")
        return pd.read_csv(full_path), full_path
    else:
        time_ranges = get_ranges(start, end)
        for i, v in enumerate(time_ranges):
            df = query_data(caddr, v[0], v[1])
            if i == 0:
                helpers.csv(df, full_path)
            else:
                helpers.append_csv(df, full_path)
        return full_path


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        caddr, cname, start, end = get_params()
        get_data(caddr, cname, start, end)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
