#!/bin/bash

# Script to ensure required libraries and external commands for the WOD service.

ENV=wiski

function die
{
    # Echo message and exit script.
    # Arguments: optional message, optional exit code
    echo "${1:-"Unknown Error"}" 1>&2
    exit ${2:-1}
}


# Test that external executables are present
conda -V || die "Please install Anaconda or Miniconda from http://continuum.io/downloads into your PATH"
test -x /usr/bin/gdalinfo || die "Please install gdal, maybe with: sudo apt-get install gdal-bin"
test -x /usr/bin/gm || die "Please install graphicsmagic, maybe with: sudo apt-get install graphicsmagick"

if [ `conda info --envs | grep -c "^$ENV "` == 0 ]
then
  conda create -n $ENV --yes python=2.7
fi

conda install -n $ENV --yes \
  pip \
  numpy=1.9 \
  matplotlib=1.4 \
  scipy=0.15 \
  basemap=1.0 \
  netcdf4=1.1 \
  shapely=1.5 \
  gdal=1.11 \
  pyyaml=3.10  || die "Conda command failed"

. activate $ENV || die
#pip install \
#  gunicorn==18 \
#  alembic==0.6.3 || die "pip install failed"
. deactivate

echo "Requirements Installed"
