#!/bin/sh
. ~/env_variables.sh
brownie run influx_univ3_1h --network alchemynode &
echo $! >../PIDs/influx_univ3_1h.pid