import os
import argparse
import numpy as np
import helpers.helpers as helpers
import risk.parameters.csv_funding as funding
import risk.parameters.csv_impact as impact
import risk.parameters.csv_liquidations as liq
import risk.parameters.csv_pricedrift as drift
import risk.overlay.pricing as pricing
import visualizations.data.viz_data_prep as vdp
import visualizations.charts.charts as vis


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
    bid = df_deltas['delta'].apply(lambda x: pricing.bid(100, x, 0, 0))
    ask = df_deltas['delta'].apply(lambda x: pricing.ask(100, x, 0, 0))
    df_deltas['spread_perc'] = ask/bid - 1
    vis.line_chart(
        df_deltas,
        "Percentage difference between bid and ask for various alpha",
        "Spread percentage difference",
        'alpha', 'spread_perc',
        helpers.get_results_dir()+results_name
    )

    # Impact visualizations
    # Arrange data
    df_ls_pivot = df_ls.reset_index().melt(
        id_vars='index', var_name='q0', value_name='value'
    )
    df_ls_pivot.columns = ['alpha', 'perc_volume', 'ls']

    # Remove prefixes and make numeric
    df_ls_pivot = vdp.make_numeric(df_ls_pivot, 'alpha=', 'alpha')
    df_ls_pivot = vdp.make_numeric(df_ls_pivot, 'q0=', 'perc_volume')

    # Get all volumes against lambdas
    df_ls_pivot = df_ls_pivot.merge(
        df_ls_pivot.perc_volume.drop_duplicates(), how='cross')

    # Get bid and ask prices and percentage change from TWAP
    df_ls_pivot = vdp.bid_ask_perc_change(df_ls_pivot, 'ls', 'perc_volume_y')

    # Group bid and ask for grouped bar plot
    df_ls_pivot = df_ls_pivot.melt(
        id_vars=['alpha', 'perc_volume_x', 'ls',
                 'perc_volume_y', 'bid', 'ask'],
        var_name='bid_ask',
        value_name='perc_change'
    )

    vis.slider_grouped_bar_chart(
        df_ls_pivot,
        "Abs percentage change in price due to lambda and volume",
        helpers.get_results_dir()+results_name,
        "Price impact",
        'perc_volume_y', 'perc_change', 'bid_ask', 'ls',
        'Volume as percentage of cap (over short TWAP)',
        'Abs percentage change in price'
    )


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        name, p, cp, st, lt = get_params()
        main(name, p, cp, st, lt)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
