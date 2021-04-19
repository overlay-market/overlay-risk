from schedule import every, get_jobs, repeat, run_pending
import os
import time

@repeat(every(10).minutes)
def job():
    print(f"Running Brownie script influx_sushi.py at time: {time.time()}")
    # TODO: make sure file path is correct
    os.system("brownie run influx_sushi --network mainnet")

while True:
    run_pending()
    print('jobs:', get_jobs())
    time.sleep(1)
