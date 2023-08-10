#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:09:50 2020

@author: Amelie
"""

# Grib files can be converted to netcdf using CDO, e.g.:
# cdo -f nc copy cansips_hindcast_raw_latlon2.5x2.5_TMP_TGL_2m_1986-10_allmembers.grib2 cansips_hindcast_raw_latlon2.5x2.5_TMP_TGL_2m_1986-10_allmembers.nc
import numpy as np
import matplotlib.pyplot as plt
import pygrib
from netCDF4 import Dataset
from functions import ncdump

fp = './'
file = 'cansips_hindcast_raw_latlon2.5x2.5_TMP_TGL_2m_1986-10_allmembers.grib2'
pygrib.open(fp+file)



fp = './'
file_nc = 'cansips_hindcast_raw_latlon2.5x2.5_TMP_TGL_2m_1986-10_allmembers.nc'

ncid = Dataset(fp+file_nc, 'r')
ncid.set_auto_mask(False)
ncdump(ncid)
T2m_tmp = ncid.variables['2t'][:]
time_tmp = ncid.variables['time'][:]
lat_tmp = ncid.variables['lat'][:]
# ncid.close()

