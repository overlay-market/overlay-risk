import os


def job():
    print("Running script influx_metrics_univ3.py")
    os.system("python scripts/influx_metrics_univ3.py")


while True:
    job()
