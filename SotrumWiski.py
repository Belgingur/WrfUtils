#!/usr/bin/env python
# encoding: utf-8


# Python library imports
import numpy
import pylab
import netCDF4
import sys
import os
import subprocess
from mpl_toolkits.basemap import Basemap
import shapely.geometry
import ogr
import osr


def prePlot(m):
    pylab.figure(figsize=(10, 8))
    shapefiles_path=os.path.join('/', 'opt', 'python', 'Shapefiles', 
                                 'Transformed')
    shapefile_joklarname = 'joklar'
    shapefile_joklar = os.path.join(shapefiles_path, shapefile_joklarname)
    m.drawcoastlines(linewidth=1.5, color='black')
    m.readshapefile(shapefile_joklar, shapefile_joklarname,
                    linewidth=1.5, color='black')
    m.drawparallels(numpy.arange(63.,67.,1.), labels=[1,0,0,0])
    m.drawmeridians(numpy.arange(-26.,-12.,2.), labels=[0,0,0,1])
    m.drawmapscale(-15., 63.5, -19., 65., 100., 
                    barstyle='fancy', fontsize=12)
    iso = numpy.arange(200.,2300.,200.)
    HGT = m.contour(x,y,xhgt,iso,colors='black',linewidths=0.4)


def postPlot(nafn):
    pylab.savefig(nafn, dpi=300)
    pylab.clf()
    pylab.close()
    id1 = subprocess.Popen(['gm', 'convert', '-trim', '+repage', nafn, nafn])
    id1.wait()


def pointsArea(poly):
    print 'Using shapely to find area to scale'
    # Some would first transform data to e.g. km using e.g. lambert-projection

    # Step through elements in grid to evaluate if in the study area
    grid_lons = xlon.flatten()
    grid_lats = xlat.flatten()
    mask = numpy.zeros_like(grid_lons, dtype='bool')
    for i in range(len(grid_lons)):
        grid_point = shapely.geometry.Point(grid_lons[i], grid_lats[i])
        if grid_point.within(poly):
            mask[i] = True
        
    # Stand and deliver
    return mask.reshape(xlon.shape)


def readShapefile(shpfile,tx):
    # Now open the shapefile and start by reading the only layer and
    # the first feature, as well as its geometry.
    source = ogr.Open(shpfile)
    layer =  source.GetLayer()
    counts = layer.GetFeatureCount()
    points = []
    for c in range(counts):
        feature = layer.GetFeature(c)
        geometry = feature.GetGeometryRef()

        # Do the coordinate transformation
        geometry.Transform(tx)

        # Read the polygon (there is just one) and the defining points.
        polygon = geometry.GetGeometryRef(0)
        points.append(polygon.GetPoints())

    # Stand and deliver
    return points


def coordTransform():
    # Remember to set: export GDAL_DATA="/usr/share/gdal/1.10/", or
    # equivalent for version of gdal
    # Define the projections
    isnet = osr.SpatialReference ()
    isnet.ImportFromEPSG ( epsg_isnet )
    latlon = osr.SpatialReference ()
    latlon.ImportFromEPSG ( epsg_wgs84 )
    tx = osr.CoordinateTransformation ( isnet,latlon )
    return tx


#################################################
#
# Script starts here!
#
geofile = './geo_em.d02.nc'
folder = 'Figs/'
nafn = folder+'map.png'
epsg_isnet = 3057 # isnet
epsg_wgs84 = 4326 # wgs84 lat/lon
shpfiles = [ 'Shapefiles/Bla_isn93.shp',
             'Shapefiles/THJ_KVI.shp',
             'Shapefiles/Fljotsdalsstod_vatnasvid.shp',
             'Shapefiles/Thj_Sul_Bud.shp',
             'Shapefiles/THJ_HAG.shp',
             'Shapefiles/TU_SIG.shp' ]
colors = [ 'r','g','b','c','m','y']

# Define the coordinate transformation needed for shapefiles
tx = coordTransform()

# Read input data
print 'Read wrf-file'
dataset = netCDF4.Dataset(geofile)
xhgt = dataset.variables['HGT_M'][0]
xlon = dataset.variables['XLONG_M'][0]
xlat = dataset.variables['XLAT_M'][0]
dataset.close()

# Set up map
print 'Create map projection'
m=Basemap(projection='stere', lat_ts=65.,lat_0=65.,lon_0=-19.,
          llcrnrlon=-24.2, llcrnrlat=63.25,
          urcrnrlon=-13.1, urcrnrlat=66.5,
          resolution='i')
x,y = m(xlon, xlat)

# Plot the map!
print 'Plot map'
prePlot(m)
for shpfile,color in zip(shpfiles,colors):
    print 'Read shape file', shpfile
    points = readShapefile(shpfile,tx)
    print 'Create polygon'
    for p in points:
        poly = shapely.geometry.Polygon(p)
        print 'Find grid points within polygon'
        mask = pointsArea(poly)
        m.plot(x[mask],y[mask], color+'o', ms=2, mec=color)
        x_p,y_p = m(poly.exterior.xy[0],poly.exterior.xy[1])
        m.plot(x_p,y_p, color+'-', lw=2, label=shpfile[11:], alpha=0.5)
pylab.legend(ncol=2,fontsize='x-small')
postPlot(nafn)

