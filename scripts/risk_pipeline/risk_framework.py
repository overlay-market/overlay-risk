import os
import argparse
import numpy as np
import helpers.helpers as helpers
import helpers.visualizations as vis
import risk.parameters.csv_funding as funding
import risk.parameters.csv_impact as impact
import risk.parameters.csv_liquidations as liq
import risk.parameters.csv_pricedrift as drift


def get_params():
    """
    Get parameters from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename', type=str,
        help='Name of the input data file. Without ".csv"'
    )
    parser.add_argument(
        '--periodicity', type=int,
        help='Cadence of input data. In secs'
    )
    parser.add_argument(
        '--payoffcap', type=int,
        help='Cap on pay off chosen by governance. Ex: 10'
    )
    parser.add_argument(
        '--short_twap', type=int,
        help='Shorter TWAP. In secs'
    )
    parser.add_argument(
        '--long_twap', type=int,
        help='Longer TWAP. In secs'
    )

    args = parser.parse_args()
    return args.filename, args.periodicity, args.payoffcap,\
        args.short_twap, args.long_twap


def main(file_name, p, cp, st, lt):
    results_name = file_name.replace('_treated', '')
    helpers.create_dir(results_name)

    # Run risk scripts
    df_ks, df_nvars_long, df_nvars_short, df_ness_long, df_ness_short,\
        df_nevs_long, df_nevs_short = funding.main(file_name, p, cp)
    df_deltas, df_ls = impact.main(file_name, p, cp, st)
    df_mms, df_betas = liq.main(file_name, p)
    df_mus = drift.main(file_name, p, lt)

    # Funding visualizations
    # Funding % Paid Daily for Various Anchor Times
    df_ks['Days'] = df_ks.index
    df_ks['Days'] = df_ks['Days'].apply(
        lambda x: int(x.replace('n=', ''))/86400)
    df_ks["Percentage of position paid as funding"] =\
        df_ks['alpha=0.05'] * 2 * 3600 * 24
    vis.line_chart(
        df_ks,
        "Funding % Paid Daily for Various Anchor Times (alpha = 0.05)",
        "Daily funding per anchor time",
        "Days", "Percentage of position paid as funding",
        helpers.get_results_dir()+results_name
    )

    # Spread visualizations
    # Percentage difference between bid and ask
    


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        name, p, cp, st, lt = get_params()
        main(name, p, cp, st, lt)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
