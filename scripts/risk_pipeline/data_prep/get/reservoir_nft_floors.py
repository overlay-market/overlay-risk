import requests
import pandas as pd


url = "https://api.reservoir.tools/events/collections/floor-ask/v1"

params = {
    'collection': '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D'.lower(),
    'startTimestamp': 1653042910,
    'endTimestamp': 1677235800,
    'limit': 1000
}

response = requests.get(url, params)

response_dicts = [response.json()]
i = 0
while response_dicts[i]['continuation']:
    print(i)
    params['continuation'] = response_dicts[0]['continuation']
    response = requests.get(url, params)
    response_dicts.append(response.json())
    i += 1

dfs = [pd.json_normalize(d['events']) for d in response_dicts]
df = pd.concat(dfs, ignore_index=True)
