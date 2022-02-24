#!/bin/sh
. ~/env_variables.sh
while :
do
    python ../../influx_metrics_univ3.py &
    sleep 20
    echo "KILL NOW"
    kill -KILL `ps -o pid= -C 'python ../../influx_metrics_univ3.py'`
done