from schedule import every, get_jobs, repeat, run_pending
import os
import time


@repeat(every(10).minutes)
def job():
    print(f"Running Brownie script influx_metrics.py at time: {time.time()}")
    os.system("brownie run influx_metrics --network mainnet")


while True:
    run_pending()
    print('jobs:', get_jobs())
    time.sleep(1)
