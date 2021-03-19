import pytest
import numpy as np

from brownie import Contract, Keep3rV1OracleMetrics


@pytest.fixture
def deployer(accounts):
    yield accounts[0]


@pytest.fixture
def metrics(deployer):
    yield Keep3rV1OracleMetrics.deploy({'from': deployer})


@pytest.fixture
def kv10(metrics):
    yield Contract.from_explorer(metrics.KV1O())


@pytest.mark.require_network("mainnet-fork")
def test_mu(metrics, kv10):
    samples = kv10.sample(
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        1e18,
        "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
        96,  # 2 days
        2,  # 1 h rolling
    )
    t = kv10.periodSize()

    mu = metrics.mu(
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "0xdac17f958d2ee523a2206206994597c13d831ec7",
        96,
        2,
    )

    # Compare log mean calc from sample with mu
    assert mu == (float(np.mean([
        np.log(samples[i]/samples[i-1]) for i in range(1, 96, 1)
    ]) / t))


@pytest.mark.require_network("mainnet-fork")
def test_sigma_sqrd(metrics, kv10):
    samples = kv10.sample(
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        1e18,
        "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
        96,  # 2 days
        2,  # 1 h rolling
    )
    t = kv10.periodSize()

    ss = metrics.sigSqrd(
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "0xdac17f958d2ee523a2206206994597c13d831ec7",
        96,
        2,
    )

    # Compare log var calc from sample with sigma squared
    assert ss == (float(np.var([
        np.log(samples[i]/samples[i-1]) for i in range(1, 96, 1)
    ]) / t))
