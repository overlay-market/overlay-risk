import argparse
import os
import time
import get.reservoir_nft_data as gr
import treat.reservoir_treatment as tr


def epoch_time(ts_string):
    time_struct = time.strptime(ts_string, "%Y-%m-%d-%H-%M")

    # Convert the time struct to Unix time and return
    return int(time.mktime(time_struct))


def get_params():
    """
    Get parameters from command line
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--collection_addr', type=str,
        help=('Contract address of the collection.'
              'Ex, for BAYC: 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D')
    )
    parser.add_argument(
        '--collection_name', type=str,
        help='Name of the collection. Ex: BAYC'
    )
    parser.add_argument(
        '--start_time', type=str,
        help='Start time for data pull. Ex: 2016-06-01-00-00'
    )
    parser.add_argument(
        '--end_time', type=str,
        help='End time for data pull. Ex: 2016-06-01-00-00'
    )
    parser.add_argument(
        '--final_periodicity', type=int,
        help='Cadence of data required for risk analysis (in secs). Ex: 3600'
    )

    args = parser.parse_args()
    start_time = epoch_time(args.start_time)
    end_time = epoch_time(args.end_time)

    return (args.collection_addr, args.collection_name,
            start_time, end_time, args.final_periodicity)


def main(caddr, cname, start, end, tf):
    data_path = gr.get_data(caddr, cname, start, end, 'floors')
    data_name = os.path.basename(data_path).replace('.csv', '')
    df = tr.treatment(data_name, tf)
    return df


if __name__ == '__main__':
    caddr, cname, start, end, tf = get_params()
    main(caddr, cname, start, end, tf)
