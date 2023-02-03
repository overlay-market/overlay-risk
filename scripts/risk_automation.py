
import yaml
from pathlib import Path
import os
import sys
import pandas as pd
import plotly.express as px


# # Global variables
# Get brownie and python installed thru poetry
python_path = str(Path(sys.executable))
brownie_path = str(Path(sys.executable).parent.joinpath('brownie'))


def run_command(command):
    print(f"Running {command}")
    os.system(command)


def main():
    # # Read input.yaml and get metrics
    inputs = yaml.safe_load(Path('scripts/constants/inputs.yaml').read_text())
    print(inputs)

    # Get data from uni v3
    command = (brownie_path + " run univ3_prices.py main "
               f"'{inputs['pool_addr']}','{inputs['start_block']}',"
               f"'{inputs['end_block']}','{inputs['pool_name']}',"
               f"'{inputs['twap_length']}','{inputs['periodicity']}'"
               " --network mainnet")
    # run_command(command)

    # Plot and save prices
    # filename = (f"{inputs['pool_name']}-{inputs['twap_length']/60}mTWAP"
    #             f"-{inputs['start_block']}-to-{inputs['end_block']}")
    filename = 'WBTC-WETH-SPOT_10min'
    df = pd.read_csv(f"csv/{filename}.csv")
    plt = df[['twap']].plot().get_figure()
    plt.savefig(f"csv/metrics/{filename}.jpg")

    # Run funding script
    command = (python_path + f" scripts/csv_funding.py --filename {filename}"
               f" --periodicity {inputs['periodicity']}"
               f" --payoffcap {inputs['payoff_cap']}")
    run_command(command)

    # Run impact script
    command = (python_path + f" scripts/csv_impact.py --filename {filename}"
               f" --periodicity {inputs['periodicity']}"
               f" --payoffcap {inputs['payoff_cap']}"
               f" --short_twap {inputs['short_twap']}")
    # run_command(command)

    # Run pricedrift script
    command = (python_path +
               f" scripts/csv_pricedrift.py --filename {filename}"
               f" --periodicity {inputs['periodicity']}"
               f" --long_twap {inputs['short_twap']}")
    # run_command(command)

    # Run liquidations script
    command = (python_path +
               f" scripts/csv_liquidations.py --filename {filename}"
               f" --periodicity {inputs['periodicity']}")
    # run_command(command)

    # # Recommend parameters and plot charts
    # Funding
    params = {}
    df_ks = pd.read_csv(f"csv/metrics/{filename}-ks.csv")
    params['k'] = df_ks.loc[44, 'alpha=0.05']
    df_ks_plot = df_ks['alpha=0.05'].iloc[20:60] * 2 * 3600 * 24
    breakpoint()
    df_ks_plot = pd.DataFrame(df_ks_plot)
    df_ks_plot['days'] = df_ks_plot.index
    df_ks_plot.columns = ["Percentage of position paid as funding", 'Days']
    fig = px.line(df_ks_plot,
                  x='Days', y='Percentage of position paid as funding')
    fig.update_layout(
        title="Funding % Paid Daily for Various Anchor Times (alpha = 0.05)")
    fig.update_layout(xaxis_title="Days",
                      yaxis_title="Percentage of position paid as funding")
    fig.write_html("Daily funding per anchor time.html")



if __name__ == '__main__':
    main()
