#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:21:23 2022

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import datetime as dt
import calendar
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from functions import ncdump
import os
import cartopy.crs as ccrs


#%%
def plot_pcolormesh_cartopy(var,gridlats,gridlons, proj=ccrs.PlateCarree()):
    plt.figure()
    ax = plt.axes(projection = proj)
    plt.pcolormesh(gridlons, gridlats, var)
    ax.coastlines()
    plt.show()

#%%
fdir_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CMC_GHRSST/'
fdir_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CMC_GHRSST/'

verbose = False

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]


#%%

# monthly_SST_0p2 = np.zeros((2015-1992+1,12,31,351,901))*np.nan

# for iyr,year in enumerate(years[1:25]):
#     print(year)
#     res = '0.2deg'
#     version = 'v2'
#     end = '-fv02.0'

#     # for doy in range(1):
#     for doy in range(365+calendar.isleap(year)):
#         os.chdir(fdir_r +str(year)+'/')

#         doydate = dt.date(year,1,1)+dt.timedelta(days=doy)
#         month = doydate.month
#         day = doydate.day
#         filename = str(year)+str(month).rjust(2, '0')+str(day).rjust(2, '0')+'120000-CMC-L4_GHRSST-SSTfnd-CMC'+res+'-GLOB-v02.0'+end+'.nc'

#         try:
#             ncid = Dataset(filename, 'r')
#             ncid.set_auto_mask(False)
#             if verbose: ncdump(ncid)

#             lat = np.squeeze(ncid.variables['lat'][:])
#             lon = np.squeeze(ncid.variables['lon'][:])
#             mask = np.squeeze(ncid.variables['mask'][:])
#             sst = np.squeeze(ncid.variables['analysed_sst'][:])
#             sst[mask == 2] = np.nan # Mask land

#             sel_lat_min, sel_lat_max = 0,70 # Select northern hemisphere
#             sel_lon_min, sel_lon_max = -180,0
#             ilatmin = np.where(lat == sel_lat_min)[0][0]
#             ilatmax = np.where(lat == sel_lat_max)[0][0]+1
#             ilonmin = np.where(lon == sel_lon_min)[0][0]
#             ilonmax = np.where(lon == sel_lon_max)[0][0]+1
#             monthly_SST_0p2[iyr,month-1,day-1,:,:] = sst[ilatmin:ilatmax,ilonmin:ilonmax]
#             # var = sst
#             # plot_pcolormesh_cartopy(var,lat,lon)
#         except:
#             print(doydate, '- FILE NOT FOUND!')

# np.savez(fdir_p+'daily_SST_0p2',data = monthly_SST_0p2)

# monthly_SST_0p2 = np.squeeze(np.nanmean(monthly_SST_0p2,axis=2))
# np.savez(fdir_p+'monthly_SST_0p2',data = monthly_SST_0p2)


monthly_SST_0p1 = np.zeros((2021-2016+1,12,31,701,1801))*np.nan
for iyr,year in enumerate(years[25:]):
    print(year)
    res = '0.1deg'
    version = 'v3'
    end = '-fv03.0'
    # for doy in range(1):
    for doy in range(365+calendar.isleap(year)):
        os.chdir(fdir_r+str(year)+'/')

        doydate = dt.date(year,1,1)+dt.timedelta(days=doy)
        month = doydate.month
        day = doydate.day
        filename = str(year)+str(month).rjust(2, '0')+str(day).rjust(2, '0')+'120000-CMC-L4_GHRSST-SSTfnd-CMC'+res+'-GLOB-v02.0'+end+'.nc'

        try:
            ncid = Dataset(filename, 'r')
            ncid.set_auto_mask(False)
            if verbose: ncdump(ncid)

            lat = np.squeeze(ncid.variables['lat'][:])
            lon = np.squeeze(ncid.variables['lon'][:])
            mask = np.squeeze(ncid.variables['mask'][:])
            sst = np.squeeze(ncid.variables['analysed_sst'][:])
            sst[mask == 2] = np.nan # Mask land

            sel_lat_min, sel_lat_max = 0,70 # Select northern hemisphere
            sel_lon_min, sel_lon_max = -180,0
            ilatmin = np.where(lat == sel_lat_min)[0][0]
            ilatmax = np.where(lat == sel_lat_max)[0][0]+1
            ilonmin = np.where(lon == sel_lon_min)[0][0]
            ilonmax = np.where(lon == sel_lon_max)[0][0]+1
            monthly_SST_0p1[iyr,month-1,day-1,:,:] = sst[ilatmin:ilatmax,ilonmin:ilonmax]
            # var = sst
            # plot_pcolormesh_cartopy(var,lat,lon)
        except:
            print(doydate, '- FILE NOT FOUND!')

np.savez(fdir_p+'daily_SST_0p1',data = monthly_SST_0p1)

monthly_SST_0p1 = np.squeeze(np.nanmean(monthly_SST_0p1,axis=2))
np.savez(fdir_p+'monthly_SST_0p1',data = monthly_SST_0p1)


