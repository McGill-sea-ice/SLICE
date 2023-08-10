#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:36:31 2021

@author: Amelie







!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NOTE: This is now part of the 'load_weather_vars_ERA5' function
      in functions_MLR.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



"""

import numpy as np
import datetime as dt
import calendar
import matplotlib.pyplot as plt
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir='/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/temp_files/') #python


import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)


from functions_MLR import datecheck_var_npz,update_water_level
from functions_MLR import update_ERA5_var
from functions_MLR import update_NAO_index
from functions import running_nanmean,find_freezeup_Tw_all_yrs,season_mask
from functions import linear_fit, rolling_climo, r_confidence_interval
from functions import detrend_ts, get_window_vars, deseasonalize_ts

#%%
years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

fp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)

#%%
end_dates_arr = np.zeros((len(years),4))*np.nan
for iyear,year in enumerate(years):
    doy_sep1 = (dt.date(int(year),9,1)-(dt.date(int(year),1,1))).days + 1
    doy_oct1 = (dt.date(int(year),10,1)-(dt.date(int(year),1,1))).days + 1
    doy_nov1 = (dt.date(int(year),11,1)-(dt.date(int(year),1,1))).days + 1
    doy_dec1 = (dt.date(int(year),12,1)-(dt.date(int(year),1,1))).days + 1
    doy_dec15 = (dt.date(int(year),12,15)-(dt.date(int(year),1,1))).days + 1
    doy_dec30 = (dt.date(int(year),12,30)-(dt.date(int(year),1,1))).days + 1
    # end_dates_arr[iyear,0] = doy_dec15
    # end_dates_arr[iyear,0] = doy_dec30
    end_dates_arr[iyear,0] = doy_dec1
    end_dates_arr[iyear,1] = doy_nov1
    end_dates_arr[iyear,2] = doy_oct1
    end_dates_arr[iyear,3] = doy_sep1
enddate_labels = ['Dec. 1st', 'Nov. 1st', 'Oct. 1st', 'Sept. 1st']

p_critical = 0.05

deseasonalize = False
detrend = True
anomaly = 'linear'

nboot = 1

month_start_day = 1

#window_arr = 2*2**np.arange(0,8) # For powers of 2
window_arr = np.arange(1,39)*7 # For weeks, up to Jan 1st
# window_arr = np.arange(1,3)*7
window_arr = np.arange(1,9)*30 # For months

#%%
save_weather = True
region_list = ['D']

var_list      = ['t2m',           't2m',           't2m',           'msl',                    'sf',      'tcc',              'tp',                 'windspeed','RH',  'SH',  'FDD',  'TDD']
savename_list = ['2m_temperature','2m_temperature','2m_temperature','mean_sea_level_pressure','snowfall','total_cloud_cover','total_precipitation','windspeed','RH',  'SH',  'FDD',  'TDD']
vartype_list  = ['max',           'min',           'mean',          'mean',                   'mean',    'mean',             'mean',               'mean',     'mean','mean','mean','mean']


weather_varnames = ['Avg. Ta_max',
                    'Avg. Ta_min',
                    'Avg. Ta_mean',
                    'Tot. TDD',
                    'Tot. FDD',
                    'Tot. CDD',
                    'Tot. precip.',
                    'Avg. SLP',
                    'Avg. wind speed',
                    'Avg. u-wind',
                    'Avg. v-wind',
                    'Tot. snowfall',
                    'Avg. cloud cover',
                    'Avg. spec. hum.',
                    'Avg. rel. hum.'
                      ]

weather_vars_all = np.zeros((len(weather_varnames),len(years),len(window_arr),end_dates_arr.shape[1],2,len(region_list)))*np.nan


# THEN ADD THE OTHER LOCATIONS FROM ERA5:
for iloc,region in enumerate(region_list):

    fpath_ERA5_processed = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/ERA5/region'+region+'/'

    max_Ta = np.load(fpath_ERA5_processed+'ERA5_dailymax_2m_temperature.npz')['data']
    max_Ta = np.squeeze(max_Ta)

    min_Ta = np.load(fpath_ERA5_processed+'ERA5_dailymin_2m_temperature.npz')['data']
    min_Ta = np.squeeze(min_Ta)

    avg_Ta = np.load(fpath_ERA5_processed+'ERA5_dailymean_2m_temperature.npz')['data']
    avg_Ta = np.squeeze(avg_Ta)

    precip = np.load(fpath_ERA5_processed+'ERA5_dailymean_total_precipitation.npz')['data']
    precip = np.squeeze(precip)

    slp = np.load(fpath_ERA5_processed+'ERA5_dailymean_mean_sea_level_pressure.npz')['data']
    slp = np.squeeze(slp)

    uwind = np.load(fpath_ERA5_processed+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
    uwind = np.squeeze(uwind)

    vwind = np.load(fpath_ERA5_processed+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
    vwind = np.squeeze(vwind)

    snow = np.load(fpath_ERA5_processed+'ERA5_dailymean_snowfall.npz')['data']
    snow = np.squeeze(snow)

    clouds = np.load(fpath_ERA5_processed+'ERA5_dailymean_total_cloud_cover.npz')['data']
    clouds = np.squeeze(clouds)

    windspeed = np.load(fpath_ERA5_processed+'ERA5_dailymean_windspeed.npz')['data']
    windspeed = np.squeeze(windspeed)

    avg_RH = np.load(fpath_ERA5_processed+'ERA5_dailymean_RH.npz')['data']
    avg_RH = np.squeeze(avg_RH)

    avg_SH = np.load(fpath_ERA5_processed+'ERA5_dailymean_SH.npz')['data']
    avg_SH = np.squeeze(avg_SH)

    FDD = np.load(fpath_ERA5_processed+'ERA5_dailymean_FDD.npz')['data']
    FDD = np.squeeze(FDD)

    TDD = np.load(fpath_ERA5_processed+'ERA5_dailymean_TDD.npz')['data']
    TDD = np.squeeze(TDD)

    CDD = avg_Ta.copy()

    weather_vars = np.zeros((len(time),len(weather_varnames)))*np.nan
    weather_vars[:,0] = max_Ta
    weather_vars[:,1] = min_Ta
    weather_vars[:,2] = avg_Ta
    weather_vars[:,3] = TDD
    weather_vars[:,4] = -1*FDD
    weather_vars[:,5] = precip
    weather_vars[:,6] = slp
    weather_vars[:,7] = windspeed
    weather_vars[:,8] = uwind
    weather_vars[:,9] = vwind
    weather_vars[:,10] = snow
    weather_vars[:,11] = clouds
    weather_vars[:,12] = avg_SH
    weather_vars[:,13] = avg_RH

    if deseasonalize:
        Nwindow = 31
        weather_vars = deseasonalize_ts(Nwindow,weather_vars,weather_varnames,'all_time',time,years)

    # Separate in different windows with different end dates
    for iend in range(end_dates_arr.shape[1]):
        weather_vars_all[:,:,:,iend,:,iloc] = get_window_vars(weather_vars,weather_varnames,np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)


    if save_weather:
        save_path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/ERA5/region'+region+'/'
        savename = 'weather_vars_ERA5'
        locnames = ['ERA5\nMLO+OR']
        np.savez(save_path+savename,
                  weather_vars= weather_vars_all,
                  varnames = weather_varnames,
                  locnames = locnames,
                  years = years,
                  window_arr = window_arr,
                  deseasonalize = deseasonalize,
                  end_dates_arr = end_dates_arr,
                  enddate_labels = enddate_labels)

