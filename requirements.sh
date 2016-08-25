#! /bin/bash

# Die on all errors by default
set -e

conda create --yes --name WrfUtils python=2.7 || echo "It's OK"
source activate WrfUtils

conda install --yes \
    pip \
    netcdf4=1.2.2 \
    pyyaml=3 \

# ncdump is broken in netcdf=1.2.4
