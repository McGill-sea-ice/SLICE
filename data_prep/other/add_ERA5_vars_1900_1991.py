#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:45:53 2022

@author: Amelie
"""
import numpy as np
import datetime as dt
import calendar
import matplotlib.pyplot as plt
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir='/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/temp_files/') #python

import statsmodels.api as sm
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

from netCDF4 import Dataset
import cdsapi
from functions_MLR import download_era5

#%%
###
def update_ERA5_var(var,var_type,processed_varname,region,update_startdate,update_enddate,raw_fp,processed_fp,time_freq=24,save=False):

    if (var == 'windspeed') | (var == 'RH') | (var == 'SH') | (var == 'FDD') | (var == 'TDD'):

        # Load past data
        past_data = np.load(processed_fp+'ERA5_dailymean_'+var+'.npz')['data']

        # Load updated base variables
        u10 = np.load(processed_fp+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
        v10 = np.load(processed_fp+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
        avg_Ta = np.load(processed_fp+'ERA5_dailymean_2m_temperature.npz')['data']
        avg_Td = np.load(processed_fp+'ERA5_dailymean_2m_dewpoint_temperature.npz')['data']
        slp = np.load(processed_fp+'ERA5_dailymean_mean_sea_level_pressure.npz')['data']

        # Define new variables
        windspeed = np.sqrt(u10**2 + v10**2)
        e_sat =0.61094*np.exp((17.625*avg_Ta)/(avg_Ta +243.04)) # Saturation vapor pressure (August–Roche–Magnus formula)
        avg_SH = 0.622*e_sat/(slp-0.378*e_sat) # Daily avg. specific humidity
        avg_RH = (np.exp((17.625*avg_Td)/(243.04+avg_Td))/np.exp((17.625*avg_Ta)/(243.04+avg_Ta))) # Daily avg. relative humidity

        mask_FDD = (avg_Ta <= 0)
        FDD = avg_Ta.copy()
        FDD[~mask_FDD] = np.nan

        mask_TDD = (avg_Ta > 0)
        TDD = avg_Ta.copy()
        TDD[~mask_TDD] = np.nan

        if (var == 'windspeed'): new_data=windspeed
        if (var == 'RH'): new_data=avg_RH
        if (var == 'SH'): new_data=avg_SH
        if (var == 'FDD'): new_data=FDD
        if (var == 'TDD'): new_data=TDD

        # Update variables
        updated_data = past_data.copy()
        updated_data[~np.isnan(new_data)] = new_data[~np.isnan(new_data)]

        if save:
            # Save new variables
            np.savez(processed_fp+'ERA5_dailymean_'+var,data=updated_data)

        return updated_data


    else:

        date_ref = dt.date(1900,1,1)
        date_start = dt.date(1979,1,1)
        date_end = dt.date(2021,12,31)

        time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

        # Load previous data
        past_data = np.load(processed_fp+'ERA5_'+var_type+'_'+processed_varname+'.npz')
        past_data = past_data['data']

        # Download new ERA5 file
        download_era5([processed_varname],region,
                      output_dir=raw_fp,
                      start_year=update_startdate.year,
                      end_year=update_enddate.year)

        # # Load newly downloaded data
        # if (update_startdate.day == 1) & (update_enddate.day == 31):
        #     fname = 'ERA5_{}_{}{}.nc'.format(processed_varname, str(update_startdate.year), str(update_startdate.month).rjust(2, '0'))
        # else:
        #     fname = 'ERA5_{}_{}{}{}_{}{}{}.nc'.format(processed_varname,str(update_startdate.year),str(update_startdate.month).rjust(2, '0'),str(update_startdate.day).rjust(2, '0'),str(update_enddate.year),str(update_enddate.month).rjust(2, '0'),str(update_enddate.day).rjust(2, '0'))

        # fpath = raw_fp+str(update_startdate.year)+'-'+str(update_startdate.month).rjust(2, '0')+'/'+fname
        # if os.path.isfile(fpath):
        #     ncid = Dataset(fpath, 'r')
        #     ncid.set_auto_mask(False)
        #     time_tmp = ncid.variables['time'][:]
        #     time_tmp = time_tmp[0:time_tmp.size:time_freq]
        #     ncid.close()

        #     if region == 'A':
        #         # REGION A: ST_LAWRENCE
        #         rlon1, rlat1 = -75.25, 44.75
        #         rlon2, rlat2 = -73.25, 46.00
        #     elif region == 'B':
        #         # REGION B: LAKE ONTARIO
        #         rlon1, rlat1 = -77.25, 43.25
        #         rlon2, rlat2 = -75.50, 44.50
        #     elif region == 'C':
        #         # REGION C: OTTAWA RIVER + MONTREAL
        #         rlon1, rlat1 = -77.25, 44.75
        #         rlon2, rlat2 = -73.25, 46.00
        #     elif region == 'D':
        #         # REGION D: ALL
        #         rlon1, rlat1 = -77.25, 43.25
        #         rlon2, rlat2 = -73.25, 46.00
        #     elif region == 'E':
        #         # REGION E: OTTAWA RIVER ONLY
        #         rlon1, rlat1 = -77.25, 44.75
        #         rlon2, rlat2 = -75.25, 46.00

        #     # Average var in the selected region and for the correct temporal representation
        #     if var_type == 'dailymean':
        #         vdaily_new = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymean(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=fpath)))), returnArray = var))
        #         if len(vdaily_new.shape) > 1:
        #             # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
        #             # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
        #             # equal to 1 for ERA5 and equal to 5 for ERA5T.
        #             # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
        #             vdaily_new = np.nanmean(vdaily_new,axis=1)

        #     elif var_type == 'dailymin':
        #         vdaily_new = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymin(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=fpath)))), returnArray = var))
        #         if len(vdaily_new.shape) > 1:
        #             # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
        #             # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
        #             # equal to 1 for ERA5 and equal to 5 for ERA5T.
        #             # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
        #             vdaily_new = np.nanmean(vdaily_new,axis=1)

        #     elif var_type == 'dailymax':
        #         vdaily_new = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymax(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=fpath)))), returnArray = var))
        #         if len(vdaily_new.shape) > 1:
        #             # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
        #             # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
        #             # equal to 1 for ERA5 and equal to 5 for ERA5T.
        #             # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
        #             vdaily_new = np.nanmean(vdaily_new,axis=1)


        #     elif var_type == 'dailysum':
        #         if (update_startdate.year == update_enddate.year) & (update_startdate.month == update_enddate.month) & (update_startdate.day == update_enddate.day):
        #             # There is only one day in the update file...
        #             ncid = Dataset(fpath, 'r')
        #             ncid.set_auto_mask(False)
        #             var_nc1 = ncid.variables[var][:]
        #             var_nc1[var_nc1 < 0] = 0
        #             if len(var_nc1.shape) > 3:
        #                 var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))

        #             vdaily_new = np.nansum(var_nc1[1:,:,:],axis=0)
        #             vdaily_new = np.nanmean(np.nanmean(vdaily_new,axis=2), axis=1)

        #         else:
        #             ncid = Dataset(fpath, 'r')
        #             ncid.set_auto_mask(False)
        #             var_nc1 = ncid.variables[var][:]
        #             var_nc1[var_nc1 < 0] = 0
        #             if len(var_nc1.shape) > 3:
        #                 var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))

        #             vdaily_new = np.zeros((int(var_nc1.shape[0]/time_freq),var_nc1.shape[1],var_nc1.shape[2]))
        #             for iday in range(int(var_nc1.shape[0]/time_freq)-1):
        #                     vdaily_new[iday,:,:] = np.nansum(var_nc1[1+iday*time_freq:1+(iday+1)*time_freq,:,:],axis=0)
        #             vdaily_new[-1,:,:] = np.nansum(var_nc1[1+(iday+1)*time_freq:1+(iday+1+1)*time_freq,:,:],axis=0)
        #             vdaily_new = np.nanmean(np.nanmean(vdaily_new,axis=2), axis=1)


        #     # Correct units:
        #     if var == 'msl':
        #         vdaily_new = vdaily_new/1000. # Convert to kPa
        #     if (var == 't2m') | (var == 'd2m') | (var == 'lict') | (var == 'ltlt'):
        #         vdaily_new  = (vdaily_new-273.15)# Convert Kelvins to Celsius

        #     # Then, arrange variables in the same format as weather from NCEI (i.e. [it,var])
        #     new_data = np.zeros((len(time)))*np.nan

        #     for it in range(time_tmp.size):
        #         date_it = date_ref+dt.timedelta(hours=int(time_tmp[it]))
        #         new_time = (date_it - date_ref).days
        #         if (new_time <= time[-1]) & (new_time >= time[0]):
        #             itvar = np.where(time == int(new_time))[0][0]
        #             new_data[itvar] = vdaily_new[it]

        #     cdo.cleanTempDir()

        #     # Put all new_data data into updated_data matrix
        #     updated_data = past_data.copy()
        #     updated_data[~np.isnan(new_data)] = new_data[~np.isnan(new_data)]

        #     # Save updated data
        #     if save:
        #         savename = 'ERA5_{}_{}'.format(var_type, processed_varname)
        #         np.savez(processed_fp+savename,
        #                  data = updated_data)

        #     return updated_data

        # else:
        #     print('!!!! PROBLEM !!!!')
        #     print('--> File to update data for selected dates does not exist...')
        #     print('--> DATA NOT UPDATED - CHOOSE ANOTHER UPDATE DATE')



#%%
# years = np.array([1991,1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005,2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020,2021])
years = np.arange(1979,2022)

date_ref = dt.date(1900,1,1)
date_start = dt.date(1979,1,1)
date_end = dt.date(2021,12,31)

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#------------------------------
# Path of raw data
fp_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/'
# Path of processed data
fp_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'




#%%%%%%% UPDATE VARIABLES %%%%%%%%%
update_ERA5_vars = True

update_startdate = dt.date(1979,1,1)
update_enddate = dt.date(1990,12,31)
# update_enddate = dt.date.today()

if update_ERA5_vars:
    # UPDATE ERA5 VARIABLES
    region = 'D'
    fp_p_ERA5 = fp_p + 'ERA5_hourly/region'+region+'/'
    fp_r_ERA5 = fp_r + 'ERA5_hourly/region'+region+'/'
    var_list      = ['u10',                    'v10',                    't2m',           't2m',           't2m',           'd2m',                        'msl',                    'ro',    'sf',      'smlt',    'tcc',              'tp',                 'windspeed','RH',  'SH',  'FDD',  'TDD','licd',          'lict',                'ltlt',                        'siconc',       'sst']
    savename_list = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_temperature','2m_temperature','2m_dewpoint_temperature',    'mean_sea_level_pressure','runoff','snowfall','snowmelt','total_cloud_cover','total_precipitation','windspeed','RH',  'SH',  'FDD',  'TDD','lake_ice_depth','lake_ice_temperature','lake_total_layer_temperature','sea_ice_cover','sea_surface_temperature']
    vartype_list  = ['mean',                   'mean',                   'mean',          'max',           'min',           'mean',                       'mean',                   'sum',   'sum',     'sum',     'mean',             'sum',                'mean',     'mean','mean','mean','mean','mean',          'mean',                'mean',                        'mean',         'mean',]


    for ivar, var in enumerate(var_list):

        var = var_list[ivar]
        processed_varname = savename_list[ivar]
        var_type = 'daily'+vartype_list[ivar]

        print('Updating '+ var +' data... ')
        data = update_ERA5_var(var,var_type,processed_varname,region,update_startdate,update_enddate,fp_r_ERA5,fp_p_ERA5,save=True)


#%%
