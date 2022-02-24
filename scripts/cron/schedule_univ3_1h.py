import os


def job():
    print("Running Brownie script influx_univ3_1h.py")
    os.system("brownie run influx_univ3_1h --network alchemynode")


while True:
    job()
