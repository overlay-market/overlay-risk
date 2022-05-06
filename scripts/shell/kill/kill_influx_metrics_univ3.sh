#!/bin/sh
ps ax | grep influx_metrics_univ3_parallel.py | grep -v grep | awk '{print $1}' | xargs kill -9