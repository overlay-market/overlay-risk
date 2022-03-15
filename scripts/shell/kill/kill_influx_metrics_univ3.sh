#!/bin/sh
pg_id=`ps -o pgid= -C 'python ../../influx_metrics_univ3.py'`
echo $pg_id
kill -- -"$pg_id"
