import os


def job():
    print("Running Brownie script influx_metrics_univ3.py")
    os.system("brownie run influx_metrics_univ3")


while True:
    job()
