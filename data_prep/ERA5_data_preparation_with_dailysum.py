#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:28:54 2021
@author: Amelie

Data from ERA5 hourly data on single levels from 1979 to present
downloaded from:
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NOTE: This is now part of the 'update_ERA5_var' function
      in functions_MLR.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'

#%%
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import datetime as dt
from functions import haversine, ncdump
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir=local_path+'slice/prog/temp_files/') #python


#%%
# region = 'A' # ST_LAWRENCE
# region = 'B' # LAKE ONTARIO
# region = 'C' # OTTAWA RIVER + MONTREAL
region = 'D' # ALL
# region = 'E' # OTTAWA RIVER ONLY

# start_year = 1979
# end_year = 2021
start_year = 2022
end_year = 2022

path = local_path+'slice/data/raw/ERA5_hourly/region'+region+'/'
save_path = local_path+'slice/data/processed/ERA5_hourly/region'+region+'/'

verbose = False

date_ref = dt.date(1900,1,1)
date_start = dt.date(1979,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#%%
# var_list      = [ 'ro',   'sf',      'smlt',    'tp']
# savename_list = ['runoff','snowfall','snowmelt','total_precipitation']
# vartype_list  = ['sum',   'sum',     'sum',     'sum']

# var_list      = ['u10',                    'v10',                    't2m',           't2m',           't2m',           'd2m',                    'licd',          'lict',                'ltlt',                        'msl',                    'ro',    'siconc',       'sst',                    'sf',      'smlt',    'tcc',              'tp']
# savename_list = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_temperature','2m_temperature','2m_dewpoint_temperature','lake_ice_depth','lake_ice_temperature','lake_total_layer_temperature','mean_sea_level_pressure','runoff','sea_ice_cover','sea_surface_temperature','snowfall','snowmelt','total_cloud_cover','total_precipitation']
# vartype_list  = ['mean',                   'mean',                   'mean',          'max',           'min',           'mean',                   'mean',          'mean',                'mean',                        'mean',                   'sum',   'mean',         'mean',                   'sum',     'sum',     'mean',             'sum']

# var_list      = ['sf',      'smlt',    'tp',                 'ro']
# savename_list = ['snowfall','snowmelt','total_precipitation','runoff']
# vartype_list  = ['sum',     'sum',     'sum',                'sum'   ]

# var_list      = ['u10',                    'v10',                    't2m',           't2m',           't2m',           'd2m',                        'msl',                    'ro',      'sf',      'smlt',    'tcc',              'tp',                 'windspeed','RH',       'SH',       'FDD',      'TDD',      'licd',          'lict',                'ltlt',                        'siconc',       'sst']
# savename_list = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_temperature','2m_temperature','2m_dewpoint_temperature',    'mean_sea_level_pressure','runoff',  'snowfall','snowmelt','total_cloud_cover','total_precipitation','windspeed','RH',       'SH',       'FDD',      'TDD',      'lake_ice_depth','lake_ice_temperature','lake_total_layer_temperature','sea_ice_cover','sea_surface_temperature']
# vartype_list  = ['dailymean',              'dailymean',              'dailymean',     'dailymax',      'dailymin',      'dailymean',                  'dailymean',              'dailysum','dailysum','dailysum','dailymean',        'dailysum',           'dailymean','dailymean','dailymean','dailymean','dailymean','dailymean',     'dailymean',           'dailymean',                   'dailymean',    'dailymean',]
# var_list      = ['smlt',    'tcc',              'tp',                 'windspeed','RH',       'SH',       'FDD',      'TDD',      'licd',          'lict',                'ltlt',                        'siconc',       'sst']
# savename_list = ['snowmelt','total_cloud_cover','total_precipitation','windspeed','RH',       'SH',       'FDD',      'TDD',      'lake_ice_depth','lake_ice_temperature','lake_total_layer_temperature','sea_ice_cover','sea_surface_temperature']
# vartype_list  = ['dailysum','dailymean',        'dailysum',           'dailymean','dailymean','dailymean','dailymean','dailymean','dailymean',     'dailymean',           'dailymean',                   'dailymean',    'dailymean',]

# var_list      = ['ssrd',                             'slhf',                    'sshf']
# savename_list = ['surface_solar_radiation_downwards','surface_latent_heat_flux','surface_sensible_heat_flux']
# vartype_list  = ['dailymean',                        'dailymean',               'dailymean']

# var_list      = ['tcc']
# savename_list = ['total_cloud_cover']
# vartype_list  = ['dailymean']

var_list      = ['sshf']
savename_list = ['surface_sensible_heat_flux']
vartype_list  = ['dailymean']

# var_list      = ['ssrd']
# savename_list = ['surface_solar_radiation_downwards']
# vartype_list  = ['dailymean']

# var_list      = ['sf']
# savename_list = ['snowfall']
# vartype_list  = ['dailysum']

# var_list      = ['t2m']
# savename_list = ['2m_temperature']
# vartype_list  = ['dailymean']

time_freq = 24

#%%
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

#%%

for ivar,var in enumerate(var_list):

    var_out = np.zeros((len(time)))*np.nan

    for year in range(start_year, end_year+1):
        # for month in range(1,13):
        for month in range(9,10):
            print(month,year)
            month_str = str(month).rjust(2, '0') # '01' instead of '1'
            fdirectory = path+"{}-{}/".format(year, month_str)

            if os.path.isdir(fdirectory):

                filename = fdirectory + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month_str+'.nc'

                if os.path.isfile(filename):
                    print(savename_list[ivar]+'_'+str(year)+month_str)

                    ncid = Dataset(filename, 'r')
                    ncid.set_auto_mask(False)
                    time_tmp = ncid.variables['time'][:]
                    time_tmp = time_tmp[0:time_tmp.size:time_freq]
                    ncid.close()

                    if vartype_list[ivar] == 'dailymean':
                        vdaily = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymean(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=filename)))), returnArray = var))
                        if len(vdaily.shape) > 1:
                            # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                            # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                            # equal to 1 for ERA5 and equal to 5 for ERA5T.
                            # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                            vdaily = np.nanmean(vdaily,axis=1)

                    elif vartype_list[ivar] == 'dailymin':
                        vdaily = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymin(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=filename)))), returnArray = var))
                        if len(vdaily.shape) > 1:
                            # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                            # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                            # equal to 1 for ERA5 and equal to 5 for ERA5T.
                            # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                            vdaily = np.nanmean(vdaily,axis=1)

                    elif vartype_list[ivar] == 'dailymax':
                        vdaily = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymax(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=filename)))), returnArray = var))
                        if len(vdaily.shape) > 1:
                            # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                            # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                            # equal to 1 for ERA5 and equal to 5 for ERA5T.
                            # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                            vdaily = np.nanmean(vdaily,axis=1)

                    elif vartype_list[ivar] == 'dailysum':
                        # Daily accumulated variables need to be summed from 01:00 to 00:00 the next day because the raw variables are accumulated at the end of the record period (i.e. accumulated rain at 00:00 is rain that fell between 23:00 and 00:00)
                        if month < 12:
                            month1 = str(month).rjust(2, '0') # '01' instead of '1'
                            fdirectory1 = path+"{}-{}/".format(year, month1)
                            filename1 = fdirectory1 + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month1+'.nc'
                            ncid1 = Dataset(filename1, 'r')
                            var_nc1 = ncid1.variables[var][:]
                            var_nc1[var_nc1 < 0] = 0
                            if len(var_nc1.shape) > 3:
                                var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))
                            time_tmp = ncid1.variables['time'][:]
                            time_tmp = time_tmp[0:time_tmp.size:time_freq]
                            ncid1.close()

                            month2 = str(month+1).rjust(2, '0') # '01' instead of '1'
                            fdirectory2 = path+"{}-{}/".format(year, month2)
                            filename2 = fdirectory2 + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month2+'.nc'
                            ncid2 = Dataset(filename2, 'r')
                            var_nc2 = ncid2.variables[var][:]
                            var_nc2[var_nc2 < 0] = 0
                            if len(var_nc2.shape) > 3:
                                var_nc2 = np.squeeze(np.nanmean(var_nc2,axis=1))

                            vdaily = np.zeros((int(np.ceil(var_nc1.shape[0]/time_freq)),var_nc1.shape[1],var_nc1.shape[2]))
                            for iday in range(int(var_nc1.shape[0]/time_freq)-1):
                                vdaily[iday,:,:] = np.nansum(var_nc1[1+iday*time_freq:1+(iday+1)*time_freq,:,:],axis=0)
                            vdaily[-1,:,:] = np.squeeze(np.nansum(var_nc1[-(time_freq-1):,:,:],axis=0)+var_nc2[0,:,:])

                            vdaily = np.nanmean(np.nanmean(vdaily,axis=2), axis=1)

                        else:
                            if year == end_year:
                                # Last day sum will be missing one hour of accumulation...
                                month1 = str(month).rjust(2, '0') # '01' instead of '1'
                                fdirectory1 = path+"{}-{}/".format(year, month1)
                                filename1 = fdirectory1 + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month1+'.nc'
                                ncid1 = Dataset(filename1, 'r')
                                var_nc1 = ncid1.variables[var][:]
                                var_nc1[var_nc1 < 0] = 0
                                if len(var_nc1.shape) > 3:
                                    var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))
                                time_tmp = ncid1.variables['time'][:]
                                time_tmp = time_tmp[0:time_tmp.size:time_freq]
                                ncid1.close()

                                vdaily = np.zeros((int(np.ceil(var_nc1.shape[0]/time_freq)),var_nc1.shape[1],var_nc1.shape[2]))*np.nan
                                for iday in range(int(var_nc1.shape[0]/time_freq)-1):
                                    vdaily[iday,:,:] = np.nansum(var_nc1[1+iday*time_freq:1+(iday+1)*time_freq,:,:],axis=0)
                                vdaily[-1,:,:] = np.nansum(var_nc1[1+(iday+1)*time_freq:1+(iday+1+1)*time_freq,:,:],axis=0)
                                vdaily = np.nanmean(np.nanmean(vdaily,axis=2), axis=1)

                            else:
                                # Add Jan. 1st 00:00 of next year to accumulated variable on Dec. 31st.
                                month1 = str(month).rjust(2, '0') # '01' instead of '1'
                                fdirectory1 = path+"{}-{}/".format(year, month1)
                                filename1 = fdirectory1 + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month1+'.nc'
                                ncid1 = Dataset(filename1, 'r')
                                var_nc1 = ncid1.variables[var][:]
                                var_nc1[var_nc1 < 0] = 0
                                if len(var_nc1.shape) > 3:
                                    var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))
                                time_tmp = ncid1.variables['time'][:]
                                time_tmp = time_tmp[0:time_tmp.size:time_freq]
                                ncid1.close()

                                month2 = str(1).rjust(2, '0') # '01' instead of '1'
                                fdirectory2 = path+"{}-{}/".format(year+1, month2)
                                filename2 = fdirectory2 + 'ERA5_'+savename_list[ivar]+'_'+str(year+1)+month2+'.nc'
                                ncid2 = Dataset(filename2, 'r')
                                var_nc2 = ncid2.variables[var][:]
                                var_nc2[var_nc2 < 0] = 0
                                if len(var_nc2.shape) > 3:
                                    var_nc2 = np.squeeze(np.nanmean(var_nc2,axis=1))

                                vdaily = np.zeros((int(np.ceil(var_nc1.shape[0]/time_freq)),var_nc1.shape[1],var_nc1.shape[2]))
                                for iday in range(int(var_nc1.shape[0]/time_freq)-1):
                                    vdaily[iday,:,:] = np.nansum(var_nc1[1+iday*time_freq:1+(iday+1)*time_freq,:,:],axis=0)
                                vdaily[-1,:,:] = np.squeeze(np.nansum(var_nc1[-(time_freq-1):,:,:],axis=0)+var_nc2[0,:,:])

                                vdaily = np.nanmean(np.nanmean(vdaily,axis=2), axis=1)


                    # Correct units:
                    if var == 'msl':
                        vdaily = vdaily/1000. # Convert to kPa
                    if (var == 't2m') | (var == 'd2m') | (var == 'lict') | (var == 'ltlt'):
                        vdaily  = (vdaily-273.15)# Convert Kelvins to Celsius


                    # Then, arrange variables in the same format as weather from NCEI (i.e. [it,var])
                    # for it in range(time_tmp.size):
                    #     date_it = date_ref+dt.timedelta(hours=int(time_tmp[it]))
                    #     new_time = (date_it - date_ref).days
                    #     if (new_time <= time[-1]) & (new_time >= time[0]):
                    #         itvar = np.where(time == int(new_time))[0][0]
                    #         var_out[itvar] = vdaily[it]

                    cdo.cleanTempDir()

    # Finally, save as npy file
    savename ='uERA5_'+vartype_list[ivar]+'_'+savename_list[ivar]
    np.savez(save_path+savename,data=var_out)


#%%
# MAKE NEW VARIABLES FROM THE COMBINATION OF ORIGINAL ONES

# # Load original variables
# u10 = np.load(save_path+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
# v10 = np.load(save_path+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
# avg_Ta = np.load(save_path+'ERA5_dailymean_2m_temperature.npz')['data']
# avg_Td = np.load(save_path+'ERA5_dailymean_2m_dewpoint_temperature.npz')['data']
# slp = np.load(save_path+'ERA5_dailymean_mean_sea_level_pressure.npz')['data']

# # Define new variables
# windspeed = np.sqrt(u10**2 + v10**2)
# e_sat =0.61094*np.exp((17.625*avg_Ta)/(avg_Ta +243.04)) # Saturation vapor pressure (August–Roche–Magnus formula)
# avg_SH = 0.622*e_sat/(slp-0.378*e_sat) # Daily avg. specific humidity
# avg_RH = (np.exp((17.625*avg_Td)/(243.04+avg_Td))/np.exp((17.625*avg_Ta)/(243.04+avg_Ta))) # Daily avg. relative humidity

# mask_FDD = (avg_Ta <= 0)
# FDD = avg_Ta.copy()
# FDD[~mask_FDD] = np.nan

# mask_TDD = (avg_Ta > 0)
# TDD = avg_Ta.copy()
# TDD[~mask_TDD] = np.nan

# # Save new variables
# np.savez(save_path+'uERA5_dailymean_windspeed',data=windspeed)
# np.savez(save_path+'uERA5_dailymean_RH',data=avg_RH)
# np.savez(save_path+'uERA5_dailymean_SH',data=avg_SH)
# np.savez(save_path+'uERA5_dailymean_FDD',data=FDD)
# np.savez(save_path+'uERA5_dailymean_TDD',data=TDD)