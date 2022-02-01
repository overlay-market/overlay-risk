import os


def job():
    print("Running script influx_metrics_univ3_parallel.py")
    os.system("python scripts/influx_metrics_univ3_parallel.py")


while True:
    job()
