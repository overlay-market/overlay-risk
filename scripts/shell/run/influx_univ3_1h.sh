#!/bin/sh
# make this directory the working directory
cd "${0%/*}"
# load env variable
export $(cat ../../../.env | xargs)
# run brownie script
poetry run brownie run influx_univ3_1h --network alchemynode