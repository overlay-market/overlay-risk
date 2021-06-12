import os
import typing as tp
import pandas as pd

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, PointSettings


def get_config() -> tp.Dict:
    return {
        "token": os.getenv('INFLUXDB_TOKEN'),
        "org": os.getenv('INFLUXDB_ORG'),
        "bucket": os.getenv('INFLUXDB_BUCKET', "ovl_sushi_dev"), # source bucket to populate from csv
        "url": os.getenv("INFLUXDB_URL"),
    }


def create_client(config: tp.Dict) -> InfluxDBClient:
    return InfluxDBClient(url=config['url'], token=config['token'], debug=False)


def get_point_settings() -> PointSettings:
    point_settings = PointSettings(**{"type": "metrics-hourly"})
    point_settings.add_default_tag("influx-metrics", "ingest-data-frame")
    return point_settings


def get_data_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    dp = 'data/populate_data.csv'
    return os.path.join(base, dp)


def main():
    config = get_config()
    client = create_client(config)
    write_api = client.write_api(
        write_options=SYNCHRONOUS,
        point_settings=get_point_settings(),
    )

    print(f"Loading data from csv ...")
    dp = get_data_path()
    df = pd.read_csv(dp)

    print("records", df)

    bucket = config["bucket"]
    org = config['org']
    print(f"Writing data to {bucket} ...")
    write_api.write(bucket, org, record=df, data_frame_measurement_name="_mem")

    client.close()

if __name__ == '__main__':
    main()
