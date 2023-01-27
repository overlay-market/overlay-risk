
import yaml
from pathlib import Path
import os
import sys


# # Global variables
# Get poetry install python and brownie
python_path = str(Path(sys.executable))
brownie_path = str(Path(sys.executable).parent.joinpath('brownie'))


def main():
    # # Read input.yaml
    inputs = yaml.safe_load(Path('scripts/constants/inputs.yaml').read_text())
    print(inputs)

    # # Get data from uni v3
    os.system(brownie_path + f" run univ3_prices.py main "
              f"'{inputs['pool_addr']}','{inputs['start_block']}',"
              f"'{inputs['end_block']}','{inputs['pool_name']}',"
              f"'{inputs['twap_length']}','{inputs['periodicity']}'"
              " --network mainnet")


if __name__ == '__main__':
    main()
