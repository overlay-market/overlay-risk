import os


def job():
    print("Running Brownie script influx_univ3_1h_90d.py")
    os.system("brownie run influx_univ3_1h_90d --network alchemynode")


while True:
    job()
