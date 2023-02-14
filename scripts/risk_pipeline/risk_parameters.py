import os
import argparse
import helpers.helpers as helpers
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
    '''
    Generates, prints and saves risk parameters for given feed in the results
    directory. Also returns dataframes generated by risk scripts to be used
    downstream for visualizations and further analysis

    Inputs:
        file_name: Name of the data file
        p: Periodicity of the data file
        cp: Cap pay off applied by governance/risk analyst
        st: Length of short TWAP in secs
        lt: Length of long TWAP in secs
    
    Outputs:
        All dataframes returned by risk scripts
    '''
    results_name = file_name.replace('_treated', '')
    helpers.create_dir(results_name)
    results_path = helpers.get_results_dir()+results_name

    # Run risk scripts
    df_ks, _ = funding.get_ks(file_name, p, cp)
    df_deltas, df_ls = impact.main(file_name, p, cp, st)
    df_mms, df_betas = liq.main(file_name, p)
    df_mus = drift.main(file_name, p, lt)

    # Funding recommendation
    # Anchor time = 45 days (3888000 secs); Confidence = 95%
    param_k = int(df_ks.loc['n=3888000', 'alpha=0.05'] * 1e18)

    # Spread recommendation
    # Confidence = 95%
    param_delta = int(float(df_deltas[df_deltas.alpha == 0.05].delta) * 1e18)

    # Impact recommendation
    # Confidence = 95%; rolling volume = 10% of cap
    param_lambda = int(df_ls.loc['alpha=0.05', 'q0=0.01'] * 1e18)

    # Maintenance margin fraction recommendation
    # Time taken for position to go negative = 4 hours; Confidence = 95%
    param_mmf = int(df_mms.loc['t=14400', 'alpha=0.05'] * 1e18)

    # Maintenance margin burn rate recommendation
    # Time taken for position to go negative = 4 hours; Confidence = 95%
    param_mmb = int(df_betas.loc['t=14400', 'alpha=0.05'] * 1e18)

    # Price drift upper limit recommendation
    # Confidence = 99.9% since (alpha should be very small as per WP)
    param_pd = int(float(df_mus[df_mus.alpha == 0.001].mu_max) * 1e18)

    # Print and save parameters
    parameters = {
        'k': param_k,
        'lambda': param_lambda,
        'delta': param_delta,
        'maintenanceMarginFraction': param_mmf,
        'maintenanceMarginBurnRate': param_mmb,
        'priceDriftUpperLimit': param_pd
    }
    print('Recommended parameters: ', parameters)

    with open(f'{results_path}/parameters.txt', 'w') as f:
        print(parameters, file=f)

    return df_ks, df_deltas, df_ls, df_mms, df_betas, df_mus


if __name__ == '__main__':
    root_dir = 'overlay-risk'
    if os.path.basename(os.getcwd()) == root_dir:
        name, p, cp, st, lt = get_params()
        main(name, p, cp, st, lt)
    else:
        print("Run failed")
        print(f"Run this script from the root directory: {root_dir}")
