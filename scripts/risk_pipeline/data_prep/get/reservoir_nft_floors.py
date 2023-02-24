import requests


url = "https://api.reservoir.tools/events/collections/floor-ask/v1"

params = {'collection': '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D'.lower()}

response = requests.get(url, params)

print(response.text)
