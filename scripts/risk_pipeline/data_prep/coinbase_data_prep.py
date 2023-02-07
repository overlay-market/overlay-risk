import argparse
import os
import get.coinbase as gc
import treat.coinbase_treatment as tc


def get_params():
    """
    Get parameters from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pair', type=str,
        help='Name of the pair. Ex: ETH-USD'
    )
    parser.add_argument(
        '--pull_periodicity', type=int,
        help='Cadence of data required from Coinbase (in secs). Ex: 300'
    )
    parser.add_argument(
        '--final_periodicity', type=int,
        help='Cadence of data required for risk analysis (in secs). Ex: 600'
    )
    parser.add_argument(
        '--start_time', type=str,
        help='Start time for data pull. Ex: 2016-06-01-00-00'
    )
    parser.add_argument(
        '--end_time', type=str,
        help='End time for data pull. Ex: 2016-06-01-00-00'
    )
    args = parser.parse_args()
    return args.pair, args.pull_periodicity, args.final_periodicity,\
        args.start_time, args.end_time


def main(pair, t, tf, start, end):
    _, data_path = gc.get_data(pair, t, start, end)
    data_name = os.path.basename(data_path).replace('.csv', '')
    _ = tc.treatment(data_name, t, tf)


if __name__ == '__main__':
    pair, t, tf, start_time, end_time = get_params()
    main(pair, t, tf, start_time, end_time)
