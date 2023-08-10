#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:08:34 2022

@author: Amelie
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
import datetime as dt
import calendar
import matplotlib.pyplot as plt
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir=local_path+'slice/prog/temp_files/') #python

import statsmodels.api as sm



from functions_MLR import datecheck_var_npz,update_water_level,update_monthly_NAO_index
from functions_MLR import update_ERA5_var,load_weather_vars_ERA5
from functions_MLR import update_daily_NAO_index,update_water_discharge
from functions_MLR import get_monthly_vars_from_daily, get_3month_vars_from_daily, get_rollingwindow_vars_from_daily

from functions import detect_FUD_from_Tw,detrend_ts,bootstrap
from functions import r_confidence_interval

#%%
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
fp_r = local_path+'slice/data/raw/'
# Path of processed data
fp_p = local_path+'slice/data/processed/'

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

#%%%%%%% LOAD VARIABLES %%%%%%%%%

# Load Twater and FUD data
fp_p_Twater = local_path+'slice/data/processed/'
Twater_loc_list = ['Longueuil','Candiac','Atwater']
station_type = 'cities'
freezeup_opt = 2
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)
# avg_freezeup_doy = freezeup_doy[:,0]

#TEST: THESE ARE THE FUD FROM BEAUHARNOIS!!!!!!!!!
# avg_freezeup_doy = np.array([373., 350., 370., 381., 354., 368., 354., 385., 370., 364., 340.,
#         369., 354., 367., 360., 371., 346., 374., 377., 368., 364., 356.,
#         398., 369., 373., 358., 365., 386., 351., 356., 357., 375., 379.,
#         369., 349., 370., 384., 374., 358., 376., 354.,np.nan,np.nan])[12:]

# Average Twater from all locations:
avg_Twater = np.nanmean(Twater,axis=1)
avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
# avg_Twater = Twater[:,0]
# avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
avg_Twater_varnames = ['Avg. Twater']

# Load ERA5 weather variables:
region = 'D'
ERA5_varlist = [#'dailymean_10m_u_component_of_wind',
                #'dailymean_10m_v_component_of_wind',
                'dailymean_2m_temperature',
                # 'dailymin_2m_temperature',
                # 'dailymax_2m_temperature',
                #'dailymean_2m_dewpoint_temperature',
                # 'dailymean_mean_sea_level_pressure',
                # 'dailysum_runoff',
                # 'dailysum_snowfall',
                #'dailysum_snowmelt',
                # 'dailysum_total_precipitation',
                # 'dailymean_total_cloud_cover',
                # 'dailymean_windspeed',
                # 'daily_theta_wind',
                #'dailymean_RH',
                'dailymean_FDD',
                'dailymean_TDD'
                ]
fp_p_ERA5 = fp_p + 'ERA5_hourly/region'+region+'/'
weather_vars, weather_varnames = load_weather_vars_ERA5(fp_p_ERA5,ERA5_varlist,region,time)
# weather_varnames = ['Avg. Ta_max','Avg. Ta_min','Avg. Ta_mean','Tot. TDD','Tot. FDD','Tot. precip.','Avg. SLP','Avg. wind speed','Avg. u-wind','Avg. v-wind','Tot. snowfall','Avg. cloud cover','Avg. spec. hum.','Avg. rel. hum.']

Ta_mean = weather_vars[:,0]
FDD = weather_vars[:,1]
TDD = weather_vars[:,2]
Tw= avg_Twater_vars

#%%

aFDD = np.zeros((len(years)))*np.nan
aTDD = np.zeros((len(years)))*np.nan
Tw_0 = np.zeros((len(years)))*np.nan
doy_Ta = np.zeros((len(years)))*np.nan
for iyr,year in enumerate(years):
    it_Aug1 = np.where(time == (dt.date(year,8,1)-date_ref).days)[0][0]
    it_Nov1 = np.where(time == (dt.date(year,11,1)-date_ref).days)[0][0]

    time_Aug1_Jan31 = time[it_Aug1:np.min((it_Aug1+185,15340))]
    Ta_mean_Aug1_Jan31 = Ta_mean[it_Aug1:np.min((it_Aug1+185,15340))]
    Tw_Aug1_Jan31 = Tw[it_Aug1:np.min((it_Aug1+185,15340))]
    FDD_Aug1_Jan31 = FDD[it_Aug1:np.min((it_Aug1+185,15340))]
    TDD_Aug1_Jan31 = TDD[it_Aug1:np.min((it_Aug1+185,15340))]

    time_Nov1_Jan31 = time[it_Nov1:np.min((it_Nov1+92,15340))]
    FDD_Nov1_Jan31 = FDD[it_Nov1:np.min((it_Nov1+92,15340))]
    TDD_Nov1_Jan31 = TDD[it_Nov1:np.min((it_Nov1+92,15340))]

    it_Ta = np.where(~np.isnan(FDD_Aug1_Jan31))[0][0]
    Tw_0[iyr] = Tw_Aug1_Jan31[it_Ta-1]
    date_Ta = date_ref+dt.timedelta(days=int(time_Aug1_Jan31[it_Ta]))
    doy_Ta[iyr] = (date_Ta-dt.date(year,1,1)).days+1
    print(date_Ta)

    if ~np.isnan(avg_freezeup_doy[iyr]):
        it_FUD = np.where(time_Aug1_Jan31 == ((dt.date(year,1,1)+dt.timedelta(days=int(avg_freezeup_doy[iyr]-1)))-date_ref).days )[0][0]
        # print(it_FUD)
        # plt.figure()
        # plt.plot(Tw_Aug1_Jan31)
        # plt.plot(it_FUD,Tw_Aug1_Jan31[it_FUD],'*')
        # plt.plot(it_Ta,Tw_0,'*')
        aFDD[iyr] = np.sum(~np.isnan(FDD_Aug1_Jan31[it_Ta:it_FUD]))
        aTDD[iyr] = np.sum(~np.isnan(TDD_Aug1_Jan31[it_Ta:it_FUD]))
        # aFDD[iyr] = np.nansum((FDD_Aug1_Jan31[it_Ta:it_FUD]))

        it_FUD = np.where(time_Nov1_Jan31 == ((dt.date(year,1,1)+dt.timedelta(days=int(avg_freezeup_doy[iyr]-1)))-date_ref).days )[0][0]
        # aFDD[iyr] = np.sum(~np.isnan(FDD_Nov1_Jan31[0:it_FUD]))
        # aTDD[iyr] = np.sum(~np.isnan(TDD_Nov1_Jan31[0:it_FUD]))
        aFDD[iyr] = np.sum(~np.isnan(FDD_Nov1_Jan31[0:62]))
        aTDD[iyr] = np.sum(~np.isnan(TDD_Nov1_Jan31[0:62]))

fig,ax = plt.subplots()
ax.plot(aFDD,avg_freezeup_doy,'o',color='k')
linmodel = sm.OLS(avg_freezeup_doy, sm.add_constant(aFDD,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

#%%
# Water temperature on the first day that the daily avg Ta drops below zero
# VS
# Accumulated FDD between the first day that the daily avg Ta drops below zero and FUD.
fig,ax = plt.subplots()
ax.plot(Tw_0,aFDD,'o',color='k')
linmodel = sm.OLS(aFDD, sm.add_constant(Tw_0,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

#%%
# Water temperature on the first day that the daily avg Ta drops below zero
# VS
# Accumulated TDD between the first day that the daily avg Ta drops below zero and FUD.
fig,ax = plt.subplots()
ax.plot(Tw_0,aTDD,'o',color='k')
linmodel = sm.OLS(aTDD, sm.add_constant(Tw_0,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

#%%
# Accumulated TDD between the first day that the daily avg Ta drops below zero and FUD.
# VS
# FUD
fig,ax = plt.subplots()
ax.plot(aTDD,avg_freezeup_doy,'o',color='k')
linmodel = sm.OLS(avg_freezeup_doy, sm.add_constant(aTDD,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

#%%
# First day that daily average Ta is below zero VS FUD
plt.figure()
plt.plot(doy_Ta,avg_freezeup_doy,'o')
linmodel = sm.OLS(avg_freezeup_doy, sm.add_constant(doy_Ta,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

#%%
# Accumulated TDD between the first day that the daily avg Ta drops below zero and FUD.
# VS
# First day that daily average Ta is below zero
fig,ax = plt.subplots()
ax.plot(aTDD,doy_Ta,'o',color='k')
linmodel = sm.OLS(doy_Ta, sm.add_constant(aTDD,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)


