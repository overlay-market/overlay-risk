import os


def job():
    print("Running Brownie script influx_univ3_1m.py")
    os.system("brownie run influx_univ3_1m --network alchemynode")


while True:
    job()
