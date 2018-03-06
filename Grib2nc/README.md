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
 Change file name format, data type and timestep in gfs.yml or cfs.yml accordingly to your use.
``` 
NC data types: 
       'S1' : CHAR,
       'i1' : BYTE,
       'u1' : UBYTE,
       'i2' : SHORT,
       'u2' : USHORT,
       'i4' : INT,
       'u4' : UINT,
       'i8' : INT64,
       'u8' : UINT64,
       'f4' : FLOAT,
       'f8' : DOUBLE
```

Where: 

 $PATH_GRIB is the especific folder where the grib files are and 
 
 $PATH_OUTPUT is the folder where the output will be saved
 
 $MODEL.yml is the yaml configuration file(list of vars and atributes)

