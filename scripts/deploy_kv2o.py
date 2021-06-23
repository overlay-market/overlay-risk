import click

from brownie import accounts, network, Contract
from eth_utils import is_checksum_address


def get_address(msg: str, default: str = None) -> str:
    val = click.prompt(msg, default=default)

    while True:
        if is_checksum_address(val):
            return val

        click.echo(f"'{val}' is not a checksummed address")
        val = click.prompt(msg)


def KV2OF() -> Contract:
    # Keep3rV2OracleFactory
    return Contract.from_explorer("0xab26f32ee1e5844d0F99d23328103325E1630700")


def main():
    click.echo(f"You are using the '{network.show_active()}' network")
    deployer = accounts.load(click.prompt(
        "Account", type=click.Choice(accounts.load()))
    )
    click.echo(f"You are using: 'deployer' [{deployer.address}]")

    kv2of = KV2OF()
    pair = get_address("Pair")
    click.echo(f"You are deploying: 'pair' [{pair}]")

    kv2of.deploy(pair, {'from': deployer})
    feed = kv2of.feeds(pair)
    click.echo(f"Deployed: pair [{pair}] to feed [{feed}]")
