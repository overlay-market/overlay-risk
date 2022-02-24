#!/bin/sh
kill -KILL `ps -o pid= -C 'brownie run influx_univ3_1h --network alchemynode'`