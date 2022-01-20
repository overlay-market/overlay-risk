import os


def job():
    print("Running script influx_metrics_univ3_rv1.py")
    os.system("python scripts/influx_metrics_univ3_rv1.py")


while True:
    job()
