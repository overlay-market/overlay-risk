
import yaml
from pathlib import Path
import os
import sys
import pandas as pd


# # Global variables
# Get poetry install python and brownie
python_path = str(Path(sys.executable))
brownie_path = str(Path(sys.executable).parent.joinpath('brownie'))


def run_command(command):
    print(f"Running {command}")
    os.system(command)


def main():
    # # Read input.yaml
    inputs = yaml.safe_load(Path('scripts/constants/inputs.yaml').read_text())
    print(inputs)

    # # Get data from uni v3
    command = (brownie_path + " run univ3_prices.py main "
               f"'{inputs['pool_addr']}','{inputs['start_block']}',"
               f"'{inputs['end_block']}','{inputs['pool_name']}',"
               f"'{inputs['twap_length']}','{inputs['periodicity']}'"
               " --network mainnet")
    run_command(command)

    # # Plot and save prices
    filename = f"{inputs['pool_name']}-{inputs['twap_length']/60}mTWAP"
    df = pd.read_csv(f"csv/{filename}.csv")
    plt = df[['twap']].plot().get_figure()
    plt.savefig(f"csv/metrics/{filename}.jpg")

    # # Run funding script
    command = (python_path + f" scripts/csv_funding.py --filename {filename}"
               f" --periodicity {inputs['periodicity']}"
               f" --payoffcap {inputs['payoff_cap']}")
    run_command(command)

    # # Run impact script

    # # Run pricedrift script

    # # Run liquidations script



if __name__ == '__main__':
    main()
