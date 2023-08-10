#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:16:39 2021

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import datetime as dt
from functions import haversine, ncdump
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir='/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/temp_files/') #python


#%%
# FILE PATH:
path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/weather_ERA5/'
save_path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/weather_ERA5/'

# SELECT SEASON AND REGION:
region = 'D'


# # SELECT VARIABLES:
bundle_list  = ['lakes','lakes']
var_list     = ['cl','ltlt']
vartype_list = ['mean', 'mean']
save_varlist = ['lakecover','laketemp']
savename = 'weather_ERA5_lakes_heatfluxes_region'+region

if region == 'A':
    # REGION A: ST_LAWRENCE
    rlon1, rlat1 = -75.25, 44.75
    rlon2, rlat2 = -73.25, 46.00
elif region == 'B':
    # REGION B: LAKE ONTARIO
    rlon1, rlat1 = -77.25, 43.25
    rlon2, rlat2 = -75.50, 44.50
elif region == 'C':
    # REGION C: OTTAWA RIVER + MONTREAL
    rlon1, rlat1 = -77.25, 44.75
    rlon2, rlat2 = -73.25, 46.00
elif region == 'D':
    # REGION D: ALL
    rlon1, rlat1 = -77.25, 43.25
    rlon2, rlat2 = -73.25, 46.00
elif region == 'E':
    # REGION E: OTTAWA RIVER ONLY
    rlon1, rlat1 = -77.25, 44.75
    rlon2, rlat2 = -75.25, 46.00

lon = np.arange(-93,-58+0.25,0.25)
lat = np.arange(40,53+0.25,0.25)

rlon = np.arange(rlon1,rlon2+0.25,0.25)
rlat = np.arange(rlat1,rlat2+0.25,0.25)

# print(haversine(rlon1,rlat1,rlon2,rlat1)) # Box length bottom
# print(haversine(rlon1,rlat2,rlon2,rlat2)) # Box length top
# print(haversine(rlon1,rlat1,rlon1,rlat2)) # Box length left
# print(haversine(rlon2,rlat1,rlon2,rlat2)) # Box length right

verbose = False

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
weather_vars = np.zeros((len(time),len(var_list),12,17))*np.nan
# weather_vars[:,0] = time

# for ivar,var in enumerate(var_list):
#     bundle = bundle_list[ivar]
#     vartype = vartype_list[ivar]

#     for season in ['DJF','MAM','JJA','SON']:
#     # for season in ['SON']:
#         if (season == 'SON'):
#             ifile = path + 'ERA5_1991_2021_'+bundle+'_'+season+'_2.nc'
#         else:
#             ifile = path + 'ERA5_1991_2021_'+bundle+'_'+season+'.nc'

#         print(var+' - '+season)

#         ncid = Dataset(ifile, 'r')
#         ncid.set_auto_mask(False)
#         if verbose: ncdump(ncid)
#         time_tmp = ncid.variables['time'][:]
#         time_tmp = time_tmp[0:time_tmp.size:4]
#         ncid.close()

#         # With CDO:
#         # - select variable (selname)
#         # - select region (sellonlatbox)
#         # - average daily (daymean, daymin, daymax)
#         # ** ALSO CHECK THESE CDO METHODS; MIGHT BE USEFUL TO ADD TO ANALYSIS
#         # eca_hd           Heating degree days per time period
#         # eca_fd           Frost days index per time period

#         vdailymean = np.squeeze(cdo.daymean(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=ifile)),returnArray = var))
#         vdailymin = np.squeeze(cdo.daymin(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=ifile)),returnArray = var))
#         vdailymax = np.squeeze(cdo.daymax(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=ifile)),returnArray = var))

#         if len(vdailymean.shape) > 3:
#             # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
#             # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
#             # equal to 1 for ERA5 and equal to 5 for ERA5T.
#             # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
#             vdailymean = np.nanmean(vdailymean,axis=1)
#             vdailymin = np.nanmean(vdailymin,axis=1)
#             vdailymax = np.nanmean(vdailymax,axis=1)


#         # Then, arrange variables in the same format as weather from NCEI (i.e. [14976,nvars])
#         for it in range(time_tmp.size):
#             date_it = date_ref+dt.timedelta(hours=int(time_tmp[it]))
#             new_time = (date_it - date_ref).days
#             if (new_time <= time[-1]) & (new_time >= time[0]):
#                 itvar = np.where(time == int(new_time))[0][0]
#                 if vartype == 'mean':
#                     weather_vars[itvar,ivar,:,:] = vdailymean[it]
#                 elif vartype == 'min':
#                     weather_vars[itvar,ivar,:,:] = vdailymin[it]
#                 elif vartype == 'max':
#                     weather_vars[itvar,ivar,:,:] = vdailymax[it]

#         cdo.cleanTempDir()

# # Remove data for JF 1990:
# weather_vars[0:3800] = np.nan

#%%
# Tw_montreal = weather_vars[:,1,2,14]
# Tw_LakeOntario = weather_vars[:,1,8,3]
# Tw_AlexandriaBay = weather_vars[:,1,7,5]

Tw_ALXN6 = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/Twater_NOAA/Twater_NOAA_ALXN6.npz', allow_pickle=True)
Tw_ALXN6 = Tw_ALXN6['Twater'][:,1]

Tw_Kingston = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/Twater_SLSMC/Twater_SLSMC_Kingston.npz', allow_pickle=True)
Tw_Kingston = Tw_Kingston['Twater'][:,1]

Tw_Cornwall = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/Twater_SLSMC/Twater_SLSMC_Cornwall.npz', allow_pickle=True)
Tw_Cornwall = Tw_Cornwall['Twater'][:,1]
#%%

plt.figure()
# plt.plot(Tw_ALXN6)
plt.plot(Tw_Kingston)
plt.plot(Tw_Cornwall)
# plt.plot((Tw_AlexandriaBay-273.15))
plt.plot((weather_vars[:,1,8,5]-273.15))
plt.plot((weather_vars[:,1,8,4]-273.15))
# plt.plot((weather_vars[:,1,9,5]-273.15))




