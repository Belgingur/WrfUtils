# Upstream2nc

This script is intented to convert gfs and cfs grib files to NetCDF4,
using features inspired by DropDigits, the end file is much smaller and 
fully compressed.

## Requirements

It requires python 2.7 and newer(maybe), and some GDal libraries installed in the system, as also Pygrib, Netcdf4 for python, 
Numpy, and some standard libraries form python. 

## How to use:
`$: python upstream2nc.py $PATH_UPSTREAN $PATH_OUTPUT`

Where 
 $PATH_UPSTREAN is the especific folder where the grib files are and 
 $PATH_OUTPUT is the folder where the output will be saved
