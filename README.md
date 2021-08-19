# overlay-risk

Risk metrics to inform our choices for per-market funding constants.


## Assumptions

We sample data from on-chain oracles and assume the underlying feed is driven by a [Levy process](https://en.wikipedia.org/wiki/L%C3%A9vy_process) of the form

```
P_j = P_0 * e**(mu * j * T + sig * L(j*T))
```

where `L(j*T)` has [Levy stable](https://en.wikipedia.org/wiki/Stable_distribution) increments, `j` is an `int` for the number of update intervals that occur after the initial feed value of `P_0`, and `T` is the update interval of the feed.


## Why?

We need some way to assess the [risk to the system](https://oips.overlay.market/notes/note-4). A good approach is to model the underlying feed value as being driven by a stochastic process, which allows us to estimate "VaR" for passive OVL holders, who act as the counterparty to all unbalanced trades.

Calculation of our per-market risk metrics requires estimating distributional parameters for the underlying stochastic model. This repo aims to provide easy to access views for those parameters.


## Requirements

To run the project you need:

- Python >=3.7.2 local development environment
- [Brownie](https://github.com/eth-brownie/brownie) local environment setup
- Set env variables for [Etherscan API](https://etherscan.io/apis) and [Infura](https://eth-brownie.readthedocs.io/en/stable/network-management.html?highlight=infura%20environment#using-infura): `ETHERSCAN_TOKEN` and `WEB3_INFURA_PROJECT_ID`
- Local Ganache environment installed
- [InfluxDB](https://www.influxdata.com/) set up with local environment variables: `INFLUXDB_TOKEN`, `INFLUXDB_ORG`, `INFLUXDB_URL`


## Installation

Using [Poetry](https://github.com/python-poetry/poetry) for dependencies. Install with `pipx`

```
pipx install poetry
```

Clone the repo, then

```
poetry install
```

within the local dir.


## Scripts

To run, for example, the script to ingest stat parameters for historical risk analysis, do

```
poetry shell
brownie run influx_kv1o --network mainnet
```

### Crons

To save on gas costs for this risk analysis, there are cron schedulers to run Brownie scripts every 10 minutes, fetching cumulative price values from SushiSwap and uploading them to InfluxDB for easy-to-access historical timeseries.

To setup the cron for e.g. fetching from SushiSwap, simply run from the base dir

```
poetry shell
python scripts/cron/schedule_sushi.py
```

which will run every 10 minutes storing new cumulative price data from all quotes in `scripts/constants/quotes.json`.

## Misc
### Generate `requirements.txt`

```
% poetry cache clear --all pypi
> Delete 63 entries? (yes/no) [no] yes

% poetry lock
% poetry install
```

Then regenerate `requirements.txt` with:
```
% poetry export -f requirements.txt --output requirements.txt --without-hashes
```
