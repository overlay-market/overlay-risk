#!/bin/sh
. ~/env_variables.sh
while :
do
    python ../../influx_metrics_univ3.py &
    sleep 600
    echo "KILL AND RESTART influx_metrics_univ3.py NOW"
    kill -KILL `ps -o pid= -C 'python ../../influx_metrics_univ3.py'`
done