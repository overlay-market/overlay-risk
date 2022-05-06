#!/bin/sh
# make this directory the working directory
cd "${0%/*}"
# load env variable
export $(cat ../../../.env | xargs)
# kill running script
ps ax | grep influx_metrics_univ3_parallel.py | grep -v grep | awk '{print $1}' | xargs kill -9
# restart script from poetry shell
nohup poetry run python ../../influx_metrics_univ3_parallel.py