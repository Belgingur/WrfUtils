# GRIB2NC

This script is intented to convert gfs and cfs grib files to NetCDF4,
using features inspired by DropDigits, the end file is much smaller and 
fully compressed.

## Requirements

It requires python 3.5 and newer(maybe), Pygrib ,Netcdf4, Numpy, PyYaml and some standard libraries from python. 

## To install:
```sh
conda env create -f enviroment.yml
```


## To run:
```sh
 python grib2nc.py -g $PATH_GRIB -o $PATH_OUTPUT -c $MODEL.yml`
```
Note:
 Change file name format and timestep in gfs.yml or cfs.yml 

Where: 

 $PATH_GRIB is the especific folder where the grib files are and 
 
 $PATH_OUTPUT is the folder where the output will be saved
 
 $MODEL.yml is the yaml configuration file(list of vars and atributes)

