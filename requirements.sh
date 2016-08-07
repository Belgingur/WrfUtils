conda create --yes --name WrfUtils python=2.7
source activate WrfUtils

conda install --yes \
    pip \
    netcdf4=1.2 \
    