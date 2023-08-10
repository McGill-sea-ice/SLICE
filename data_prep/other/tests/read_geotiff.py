#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:46:33 2020

@author: Amelie
"""

import gdal
import matplotlib.pyplot as plt
import numpy as np
import osr
import matplotlib.path as mpath
import pyproj
import cartopy.crs as ccrs
from os import listdir
from os.path import isfile, join
from cartopy.io import shapereader
#_______________________________________________________________
# ALSO USE gdalinfo <<tiff file>> IN TERMINAL TO SEE THE REFERENCE
# COORDINATE SYSTEM AND MORE INFO ABOUT IMAGE
#_______________________________________________________________
# See Also tutorial here: https://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides4.pdf

#%% CHS 500m Bathymetry
# image = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CHS_500m_Bathy/500m_Gridded_Bathymetry/500m_Gridded_Bathymetry.tif'

# dataset = gdal.Open(image, gdal.GA_ReadOnly)
# # Note GetRasterBand() takes band no. starting from 1 not 0
# band = dataset.GetRasterBand(1)
# arr = band.ReadAsArray()

# arr[arr >= 1e6] = np.nan #Mask land
# plt.figure()
# plt.imshow(arr[6240:6400,14660:14840],vmin=-15,vmax=0)
# # plt.imshow(arr[:,:],vmin=-15,vmax=0)
# plt.colorbar()

#%% ETOPO
# # image = '/Users/Amelie/Desktop/NONNA_product-produit/GeoTiff/NONNA10_4530N07390W.tiff'
# image = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/etopo/exportImage.tiff'

# dataset = gdal.Open(image, gdal.GA_ReadOnly)
# # Note GetRasterBand() takes band no. starting from 1 not 0
# band = dataset.GetRasterBand(1)
# arr = band.ReadAsArray()

# plt.figure()
# plt.imshow(arr,vmin=-100,vmax=100)
# plt.colorbar()

#%% NONNA

# Setup map
# proj = ccrs.Stereographic(central_latitude=90, central_longitude=-45, false_easting=0, false_northing=0, true_scale_latitude=70)
proj = ccrs.PlateCarree()
# Make plot:
fig = plt.figure(figsize=[10, 10])
ax = plt.subplot(1, 1, 1, projection=proj)

# fpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/NONNA10_product-produit/GeoTiff/upper/'
fpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/Bathymetry/NONNA100_product-produit/GeoTiff/'
files = [f for f in listdir(fpath) if isfile(join(fpath, f))]
files = ['NONNA100_4500N07400W.tiff']
for i,im in enumerate(files):
    image = fpath+im
    dataset = gdal.Open(image, gdal.GA_ReadOnly)
    # Note GetRasterBand() takes band no. starting from 1 not 0
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    arr[arr >= 1e6] = np.nan #Mask land
    # plt.figure()
    # plt.imshow(arr,vmin=-15,vmax=0)
    # plt.colorbar()

    ds = gdal.Open(image)
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize

    geotransform = ds.GetGeoTransform()
    # From the GDAL documentation:
    # adfGeoTransform[0] /* top left x */
    # adfGeoTransform[1] /* w-e pixel resolution */
    # adfGeoTransform[2] /* rotation, 0 if image is "north up" */
    # adfGeoTransform[3] /* top left y */
    # adfGeoTransform[4] /* rotation, 0 if image is "north up" */
    # adfGeoTransform[5] /* n-s pixel resolution */
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]

    # !!!!!!CHANGE THIS TO GET COORDINATES IN BOTTOM LEFT CORNER TO FIT WITH PCOLORMESH
    # Get center coordinates for each pixel
    # NOTE: Raw rasyer coordinates are for top left corners of pixels so we add an offset of half a pixel
    # x = np.arange(originX+(0.5*pixelWidth) ,(originX+0.5*pixelWidth) +pixelWidth*(ncols+1),pixelWidth)[0:ncols]
    # y = np.arange(originY+(0.5*pixelHeight),(originY+0.5*pixelHeight)+pixelHeight*(nrows+1),pixelHeight)[0:nrows]
    # These would be the coordinates for the top left corner:
    x = np.arange(originX,(originX)+pixelWidth*ncols,pixelWidth)[0:ncols]
    y = np.arange(originY,(originY)+pixelHeight*nrows,pixelHeight)[0:nrows]

    xplot = np.zeros(x.shape[0]+1)*np.nan
    yplot = np.zeros(y.shape[0]+1)*np.nan
    xplot[0:-1] = x; xplot[-1]=x[-1]+pixelWidth
    yplot[0:-1] = y; yplot[-1]=y[-1]+pixelHeight
    # minx = geotransform[0]
    # miny = geotransform[3] + ncols*geotransform[4] + nrows*geotransform[5]
    # maxx = geotransform[0] + ncols*geotransform[1] + nrows*geotransform[2]
    # maxy = geotransform[3]
    # print(minx,miny, x[0],y[0])
    # print(maxx,maxy,x[-1],y[-1])
    # print(x[500],y[500])

    # Get the EPSG Code for the projection of the raster:
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    print(proj.GetAttrValue('AUTHORITY',1))
    # p = pyproj.Proj('epsg:'+proj.GetAttrValue('AUTHORITY',1))

    plt_im = ax.pcolormesh(x, y, arr, transform=ccrs.PlateCarree(), clim=(-15,0))
    ax.coastlines(resolution='10m')

# from cartopy.feature import NaturalEarthFeature
# coast = NaturalEarthFeature(category='physical', scale='10m',
#                             facecolor='none', name='coastline')
# feature = ax.add_feature(coast, edgecolor='gray')

plt.savefig('/Volumes/SeagateUSB/McGill/Postdoc/group_meeting/qgis_intro/map_with_cartopy', dpi=700)




#%%
# # BELOW IS FOR CHANGING COORDINATE SYSTEM:
# # get the existing coordinate system
# old_cs= osr.SpatialReference()
# old_cs.ImportFromWkt(ds.GetProjectionRef())

# # create the new coordinate system
# # wgs84_wkt = """
# # GEOGCS["WGS 84",
# #     DATUM["WGS_1984",
# #         SPHEROID["WGS 84",6378137,298.257223563,
# #             AUTHORITY["EPSG","7030"]],
# #         AUTHORITY["EPSG","6326"]],
# #     PRIMEM["Greenwich",0,
# #         AUTHORITY["EPSG","8901"]],
# #     UNIT["degree",0.01745329251994328,
# #         AUTHORITY["EPSG","9122"]],
# #     AUTHORITY["EPSG","4326"]]"""


# wgs84_wkt = """
# GEOGCRS["WGS 84",
#     DATUM["World Geodetic System 1984",
#         ELLIPSOID["WGS 84",6378137,298.257223563,
#             LENGTHUNIT["metre",1]]],
#     PRIMEM["Greenwich",0,
#         ANGLEUNIT["degree",0.0174532925199433]],
#     CS[ellipsoidal,2],
#         AXIS["geodetic latitude (Lat)",north,
#             ORDER[1],
#             ANGLEUNIT["degree",0.0174532925199433]],
#         AXIS["geodetic longitude (Lon)",east,
#             ORDER[2],
#             ANGLEUNIT["degree",0.0174532925199433]],
#     USAGE[
#         SCOPE["unknown"],
#         AREA["World"],
#         BBOX[-90,-180,90,180]],
#     ID["EPSG",4326]]
# """

# # """
# # 'GEOGCS["WGS 84",
# # DATUM["WGS_1984",
# #       SPHEROID["WGS 84",6378137,298.257223563,
# #                AUTHORITY["EPSG","7030"]],
# #       AUTHORITY["EPSG","6326"]],
# # PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
# # UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
# # AXIS["Latitude",NORTH],
# # AXIS["Longitude",EAST],
# # AUTHORITY["EPSG","4326"]]'

# # """

# new_cs = osr.SpatialReference()
# new_cs .ImportFromWkt(wgs84_wkt)

# # create a transform object to convert between coordinate systems
# transform = osr.CoordinateTransformation(old_cs,new_cs)

# #get the new coordinates in for a given point
# latlong = transform.TransformPoint(x[500],y[500])
# print(latlong)






