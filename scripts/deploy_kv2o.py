from brownie import network, Contract


def KV2O() -> Contract:
    # Keep3rV2OracleFactory
    return Contract.from_explorer("0xab26f32ee1e5844d0F99d23328103325E1630700")


def main():
    print(f"You are using the '{network.show_active()}' network")
    kv2o = KV2O()
