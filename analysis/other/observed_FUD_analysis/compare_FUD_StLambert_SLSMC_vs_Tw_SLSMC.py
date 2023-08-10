#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:30:29 2022

@author: Amelie
"""
import numpy as np
import datetime as dt
import calendar
import matplotlib.pyplot as plt
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir='/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/temp_files/') #python

# import statsmodels.api as sm
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import pandas as pd
import seaborn as sns

from functions_MLR import datecheck_var_npz,update_water_level,update_monthly_NAO_index
from functions_MLR import update_ERA5_var,load_weather_vars_ERA5
from functions_MLR import update_daily_NAO_index,update_water_discharge
from functions_MLR import get_monthly_vars_from_daily, get_3month_vars_from_daily, get_rollingwindow_vars_from_daily

from functions import detect_FUD_from_Tw,detrend_ts,bootstrap
from functions import r_confidence_interval

#%%%%%%% OPTIONS %%%%%%%%%

plot = False

ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")

#------------------------------
# Period definition
years = np.array([1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021])

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#------------------------------
# Path of raw data
fp_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/'
# Path of processed data
fp_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'

#------------------------------
# Start of forecasts options
# start_doy_arr    = [305,         312,       319,         326,         333,         340]
# start_doy_labels = ['Nov. 1st', 'Nov. 8th', 'Nov. 15th', 'Nov. 22nd', 'Nov. 29th', 'Dec. 6th']
start_doy_arr    = [300,         307,       314,         321,         328,         335]
start_doy_labels = ['Oct. 27th', 'Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st']

#------------------------------
# Correlation analysis
p_critical = 0.05

replace_with_nan = False

detrend_FUD = False
detrend = False
if detrend:
   anomaly = 'linear'


#%%%%%%% LOAD TW FROM SLSMC IN SOUTH SHORE CANAL AND EXTRACT FUD %%%%%%%%%
# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['StLambert']
station_type = 'SLSMC'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)


#%%%%%%% LOAD FUD FROM SLSMC %%%%%%%%%
data_SLSMC = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/freezeup_dates_SLSMC/freezeup_SLSMC_SouthShoreCanal.npz')
fi_SLSMC = data_SLSMC['freezeup_fi'][:]
si_SLSMC = data_SLSMC['freezeup_si'][:]

fi_SLSMC = fi_SLSMC[~np.isnan(fi_SLSMC)]
si_SLSMC = si_SLSMC[~np.isnan(si_SLSMC)]

years_SLSMC = np.arange(1981,2006)
doy_fi_SLSMC = np.zeros((len(years_SLSMC)))*np.nan
doy_si_SLSMC = np.zeros((len(years_SLSMC)))*np.nan

for i in range(len(fi_SLSMC)):
    date_FUD_fi = date_ref + dt.timedelta(days=int(fi_SLSMC[i]))
    if date_FUD_fi.month < 3:
        iyr = np.where(years_SLSMC == date_FUD_fi.year-1 )[0][0]
    else:
        iyr = np.where(years_SLSMC == date_FUD_fi.year)[0][0]
    if date_FUD_fi.year == years_SLSMC[iyr]:
        doy_FUD_fi = (date_FUD_fi-dt.date(years_SLSMC[iyr],1,1)).days + 1
    else:
        doy_FUD_fi = (365 + calendar.isleap(years_SLSMC[iyr]) +
                      (date_FUD_fi-dt.date(years_SLSMC[iyr]+1,1,1)).days + 1)
    doy_fi_SLSMC[iyr] = doy_FUD_fi


for i in range(len(si_SLSMC)):
    date_FUD_si = date_ref + dt.timedelta(days=int(si_SLSMC[i]))
    if date_FUD_si.month < 3:
        iyr = np.where(years_SLSMC == date_FUD_si.year-1 )[0][0]
    else:
        iyr = np.where(years_SLSMC == date_FUD_si.year)[0][0]
    if date_FUD_si.year == years_SLSMC[iyr]:
        doy_FUD_si = (date_FUD_si-dt.date(years_SLSMC[iyr],1,1)).days + 1
    else:
        doy_FUD_si = (365 + calendar.isleap(years_SLSMC[iyr]) +
                      (date_FUD_si-dt.date(years_SLSMC[iyr]+1,1,1)).days + 1)
    doy_si_SLSMC[iyr] = doy_FUD_si

#%%
fig,ax = plt.subplots()

FUD_Tw_Longueuil = np.array([360., 358., 364., 342., 365., 350., 365., 360., 343., 367., 339.,
       348., 354., 350., 381., 341., 347., 352., 357., 364., 358., 347.,
       365., 371., 351., 348., 356., 354.])
years_Longueuil = np.array([1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
       2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
       2014, 2015, 2016, 2017, 2018, 2019])

ax.plot(years_Longueuil, FUD_Tw_Longueuil,'-o',color='k',label='Tw detected at Longueuil')

ax.plot(years_SLSMC,doy_fi_SLSMC,'o-',label='SLSMC - First ice')
ax.plot(years_SLSMC,doy_si_SLSMC,'.-',label='SLSMC - Stable ice')
ax.plot(years[:-3],freezeup_doy[:-3],'o-',label='Tw detected at StLambert')

ax.set_ylabel('FUD')
ax.set_xlabel('Year')
ax.legend()




