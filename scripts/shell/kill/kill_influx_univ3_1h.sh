#!/bin/sh
kill -KILL `ps -o pid=$(ps -ax | grep influx_univ3_1h | awk '{print $1}')`