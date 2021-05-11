from schedule import every, get_jobs, repeat, run_pending
import os
import time

@repeat(every(10).minutes)
def metrics_job():
    print(f"Running influx_metrics.py at time: {time.time()}")
    os.system("python scripts/influx_metrics.py")

while True:
    run_pending()
    print('jobs:', get_jobs())
    time.sleep(1)
