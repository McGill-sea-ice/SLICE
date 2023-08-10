#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:28:54 2021
@author: Amelie

Data from ERA5 hourly data on single levels from 1979 to present
downloaded from:
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview

Downloaded in bundle of 6-8 variables for each season (DJF, MAM, JJA, SON)
for 1991 to 2021, for the region bounded by: W: -93, E: -58, S:40, N:53

Note 1: For the moment, only hours 00:00, 06:00, 12:00, 18:00 are downloaded.
Note 2: For DJF, the period is 1990-2021 to allow matching D-1990 with JF-1991.
Note 3: For SON, the period stops in 2020, since at the time of download data
        for 2021 was not existent.

Below is a description of the variables found in each download bundle.
See ERA5's documentation for more info:
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

BASICS BUNDLE
u10 : u-wind at 10m
v10 : v-wind at 10m
t2m : air temperature at 2m
msl : mean SLP
sst : sea surface temperature
tp : total precipitation

PRECIP BUNDLE
d2m : dewpoint at 2m
fal : forecast albedo
ptype : precipitation type
ro : runoff
siconc : sea-ice cover (concentration)
sf : snowfall
smlt : snowmelt

RATES BUNDLE
mror : mean runoff rate
msdwlwrf : mean LW down flux at surface
msdwswrf : mean SW down flux at surface
mslhf : mean latent heat flux
msshf : mean sensible heat flux
mtpr : mean total precip. rate
tcc : total cloud cover

LAKES BUNDLE
cl : lake cover
dl : lake depth
licd : lake ice depth
lict : lake ice temperature
lmld : lake mixed-layer depth
lmlt : lake mixed-layer temp.
ltlt : lake total layer temp.
lsm : landsea mask

"""

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

# FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
# print(FCT_DIR)
# sys.path.append(FCT_DIR)
# print(sys.path)


#%%
# FILE PATH:
path = '../../data/raw/weather_ERA5/'
save_path = '../../data/processed/weather_ERA5/'

# SELECT SEASON AND REGION:
# region = 'A' # ST_LAWRENCE
# region = 'B' # LAKE ONTARIO
# region = 'C' # OTTAWA RIVER + MONTREAL
# region = 'D' # ALL
# region = 'E' # OTTAWA RIVER ONLY
# region_list = ['D','A','B','E','C']
region_list = ['D']
#%%
for region in region_list:
    print('--------- Region '+region+' ---------')

    # # SELECT VARIABLES:
    # bundle_list  = ['basics','basics','basics','basics','basics','basics','basics','basics','precip','precip','precip','precip','rates']
    # var_list     = ['t2m',   't2m',   't2m',   'tp',    'msl',   'u10',   'v10',   'sst',   'd2m',   'd2m',   'd2m',   'sf',    'tcc']
    # vartype_list = ['max',   'min',   'mean',  'mean',  'mean',  'mean',  'mean',  'mean',  'max',   'min',   'mean',  'mean',  'mean']
    # save_varlist = ['max_Ta','min_Ta','avg_Ta','precip','slp','uwind','vwind','sst','max_Tdew','min_Tdew','avg_Tdew', 'snowfall', 'cloudcover']
    # savename = 'weather_ERA5_region'+region

    # bundle_list  = ['lakes','lakes','lakes','lakes','lakes','rates',   'rates',   'rates','rates']
    # var_list     = ['lict','licd','lmlt', 'ltlt', 'cl',   'msdwlwrf','msdwswrf','mslhf','msshf']
    # vartype_list = ['mean','mean','mean', 'mean', 'mean', 'mean',    'mean',    'mean', 'mean']
    # save_varlist = ['lakeicetemp','lakeicedepth','lakeMLtemp', 'laketemp', 'lakecover', 'LWsfc', 'SWsfc', 'LHflux', 'SHflux']
    # savename = 'weather_ERA5_lakes_heatfluxes_region'+region

    bundle_list  = ['precip','rates','precip','rates']
    var_list     = ['ro','mror', 'smlt', 'mtpr',]
    vartype_list = ['mean','mean', 'mean', 'mean']
    save_varlist = ['runoff','runoffrate','snowmelt', 'preciprate']
    savename = 'weather_ERA5_lakes_runoff_region'+region

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
    weather_vars = np.zeros((len(time),len(var_list)+1))*np.nan
    weather_vars[:,0] = time

    for ivar,var in enumerate(var_list):
        bundle = bundle_list[ivar]
        vartype = vartype_list[ivar]

        for season in ['DJF','MAM','JJA','SON']:
        # for season in ['SON']:
            if (season == 'SON'):
                ifile = path + 'ERA5_1991_2021_'+bundle+'_'+season+'_2.nc'
            else:
                ifile = path + 'ERA5_1991_2021_'+bundle+'_'+season+'.nc'

            print(var+' - '+season)

            ncid = Dataset(ifile, 'r')
            ncid.set_auto_mask(False)
            if verbose: ncdump(ncid)
            time_tmp = ncid.variables['time'][:]
            time_tmp = time_tmp[0:time_tmp.size:4]
            ncid.close()

            # With CDO:
            # - select variable (selname)
            # - select region (sellonlatbox)
            # - average daily (daymean, daymin, daymax)
            # - average the selected region (mermean + zonmean)
            # ** ALSO CHECK THESE CDO METHODS; MIGHT BE USEFUL TO ADD TO ANALYSIS
            # eca_hd           Heating degree days per time period
            # eca_fd           Frost days index per time period

            vdailymean = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymean(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=ifile)))), returnArray = var))
            vdailymin = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymin(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=ifile)))), returnArray = var))
            vdailymax = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymax(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=ifile)))), returnArray = var))

            if len(vdailymean.shape) > 1:
                # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                # equal to 1 for ERA5 and equal to 5 for ERA5T.
                # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                vdailymean = np.nanmean(vdailymean,axis=1)
                vdailymin = np.nanmean(vdailymin,axis=1)
                vdailymax = np.nanmean(vdailymax,axis=1)


            # Then, arrange variables in the same format as weather from NCEI (i.e. [14976,nvars])
            for it in range(time_tmp.size):
                date_it = date_ref+dt.timedelta(hours=int(time_tmp[it]))
                new_time = (date_it - date_ref).days
                if (new_time <= time[-1]) & (new_time >= time[0]):
                    itvar = np.where(time == int(new_time))[0][0]
                    if vartype == 'mean':
                        weather_vars[itvar,ivar+1] = vdailymean[it]
                    elif vartype == 'min':
                        weather_vars[itvar,ivar+1] = vdailymin[it]
                    elif vartype == 'max':
                        weather_vars[itvar,ivar+1] = vdailymax[it]

            cdo.cleanTempDir()

    # Remove data for JF 1990:
    weather_vars[0:3800] = np.nan


    # Finally, save as npy file
    np.savez(save_path+savename,
              weather_data=weather_vars,select_vars = save_varlist)


#%%
plot_2Dfield = True

if plot_2Dfield:
    region = 'D'
    season = 'SON'

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


    bundle_list  = ['lakes','lakes']
    var_list     = ['cl','ltlt']
    vartype_list = ['mean','mean']

    for i,var in enumerate(var_list):
        bundle = bundle_list[i]
        ifile = path + 'ERA5_1991_2021_'+bundle+'_'+season+'.nc'

        v = cdo.selname(var,input=ifile,returnArray = var)
        if len(v.shape) > 3:
            # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
            # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
            # equal to 1 for ERA5 and equal to 5 for ERA5T.
            # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
            v= np.nanmean(v,axis=1)
        v = np.flip(v,axis=1)

        fig = plt.subplots(nrows=1,ncols=1)
        ax = plt.subplot(projection = ccrs.PlateCarree())
        plot_lon = lon-0.25/2.
        plot_lat = lat-0.25/2.
        plt.pcolormesh(plot_lon, plot_lat, v[0])
        ax.coastlines()

        # Plot the selected field at t = 0 for example
        vr = cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=ifile), returnArray = var)
        if len(vr.shape) > 3:
            # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
            # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
            # equal to 1 for ERA5 and equal to 5 for ERA5T.
            # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
            vr= np.nanmean(vr,axis=1)

        vr = np.flip(vr,axis=1)

        fig = plt.subplots(nrows=1,ncols=1)
        ax = plt.subplot(projection = ccrs.PlateCarree())
        plot_rlon = rlon-0.25/2.
        plot_rlat = rlat-0.25/2.
        plt.pcolormesh(plot_rlon, plot_rlat, vr[0])
        ax.coastlines()

#%%

fp = '../../data/processed/'
ERA5_data = np.load(fp+'weather_ERA5/weather_ERA5_lakes_heatfluxes_regionD.npz',allow_pickle='TRUE')
weather_data = ERA5_data['weather_data']
laketemp = weather_data[:,3]
lakecover = weather_data[:,4]


