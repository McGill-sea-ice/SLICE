#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:29:14 2022

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

from functions_MLR import load_weather_vars_ERA5
from functions_MLR import get_monthly_vars_from_daily
from functions_MLR import get_daily_var_from_monthly_cansips_forecasts,get_daily_var_from_seasonal_cansips_forecasts
from functions import detect_FUD_from_Tw,detrend_ts

import statsmodels.api as sm
# import pandas as pd
# import seaborn as sns

#%%%%%%% OPTIONS %%%%%%%%%
ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")

replace_with_nan = False
#------------------------------
# Period definition
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#------------------------------
# Path of processed data
fp_p = local_path+'slice/data/processed/'

#------------------------------
# Choose FUD location:
FUD_loc = 'Longueuil'

# FUD_loc = 'Beauharnois'
# freezeup_type = 'first_ice'
# freezeup_type = 'stable_ice'

p_critical = 0.05
detrend = False
anomaly = 'linear'

save_monthly_vars = True
# save_monthly_vars = False

#%%%%%%% LOAD VARIABLES %%%%%%%%%

# Load Twater and FUD data
if FUD_loc == 'Longueuil':
    years = np.arange(1991,2022)

    fp_p_Twater = local_path+'slice/data/processed/'
    Twater_loc_list = ['Longueuil_updated']
    station_type = 'cities'
    freezeup_opt = 1
    freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)

    # Average (and round) FUD from all locations:
    # avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
    # avg_freezeup_doy = np.round(avg_freezeup_doy)
    avg_freezeup_doy = freezeup_doy[:,0]

    # Average Twater from all locations:
    # avg_Twater = np.nanmean(Twater,axis=1)
    # avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
    avg_Twater = Twater[:,0]
    avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
    avg_Twater_varnames = ['Avg. Twater']

if FUD_loc == 'Beauharnois':
    years = np.arange(1980,2022)

    data = np.load(local_path+'slice/data/processed/freezeup_dates_HQ/freezeup_HQ_BeauharnoisCanal.npz')
    fi = data['freezeup_fi'][:]
    si = data['freezeup_si'][:]

    fi = fi[~np.isnan(fi)]
    si = si[~np.isnan(si)]

    # Convert days since ... into DOY
    years_HQ = np.arange(1960,2020)
    doy_fi_HQ = np.zeros((len(fi)))*np.nan
    doy_si_HQ = np.zeros((len(si)))*np.nan
    avg_freezeup_doy = np.zeros((len(years)))*np.nan
    for i in range(len(fi)):

        date_FUD_fi = date_ref + dt.timedelta(days=int(fi[i]))
        if date_FUD_fi.year == years_HQ[i]:
            doy_FUD_fi = (date_FUD_fi-dt.date(years_HQ[i],1,1)).days + 1
        else:
            doy_FUD_fi = (365 + calendar.isleap(years_HQ[i]) +
                          (date_FUD_fi-dt.date(years_HQ[i]+1,1,1)).days + 1)
        doy_fi_HQ[i] = doy_FUD_fi

        date_FUD_si = date_ref + dt.timedelta(days=int(si[i]))
        if date_FUD_si.year == years_HQ[i]:
            doy_FUD_si = (date_FUD_si-dt.date(years_HQ[i],1,1)).days + 1
        else:
            doy_FUD_si = (365 + calendar.isleap(years_HQ[i]) +
                          (date_FUD_si-dt.date(years_HQ[i]+1,1,1)).days + 1)
        doy_si_HQ[i] = doy_FUD_si

        if years_HQ[i] in years:
            if freezeup_type == 'first_ice':
                avg_freezeup_doy[np.where(years==years_HQ[i])[0][0]] = doy_fi_HQ[i]
            if freezeup_type == 'stable_ice':
                avg_freezeup_doy[np.where(years==years_HQ[i])[0][0]] = doy_si_HQ[i]

# Check if trend is significant:
trend_model = sm.OLS(avg_freezeup_doy, sm.add_constant(years,has_constant='skip'), missing='drop').fit()
if trend_model.pvalues[1] <= p_critical:
    print('*** Warning ***\n Trend for FUD is significant. Consider detrending.')

if detrend:
    if anomaly == 'linear':
        avg_freezeup_doy, [m,b] = detrend_ts(avg_freezeup_doy,years,anomaly)
    if anomaly == 'mean':
        avg_freezeup_doy, mean = detrend_ts(avg_freezeup_doy,years,anomaly)


#%%
# Load ERA5 weather variables:
region = 'D'
ERA5_varlist = [#'dailymean_10m_u_component_of_wind',
                #'dailymean_10m_v_component_of_wind',
                'dailymean_2m_temperature',
                'dailymin_2m_temperature',
                'dailymax_2m_temperature',
                #'dailymean_2m_dewpoint_temperature',
                'dailymean_mean_sea_level_pressure',
                'dailysum_runoff',
                'dailysum_snowfall',
                # 'dailysum_snowmelt',
                'dailysum_total_precipitation',
                'dailymean_total_cloud_cover',
                'dailymean_windspeed',
                'daily_theta_wind',
                # 'dailymean_RH',
                'dailymean_FDD',
                'dailymean_TDD',
                'dailymean_surface_solar_radiation_downwards',
                'dailymean_surface_latent_heat_flux',
                'dailymean_surface_sensible_heat_flux'
                ]
fp_p_ERA5 = fp_p + 'ERA5_hourly/region'+region+'/'
weather_vars, weather_varnames = load_weather_vars_ERA5(fp_p_ERA5,ERA5_varlist,region,time)
# weather_varnames = ['Avg. Ta_max','Avg. Ta_min','Avg. Ta_mean','Tot. TDD','Tot. FDD','Tot. precip.','Avg. SLP','Avg. wind speed','Avg. u-wind','Avg. v-wind','Tot. snowfall','Avg. cloud cover','Avg. spec. hum.','Avg. rel. hum.']

#%%
# Load discharge and level data
loc_discharge = 'Lasalle'
discharge_data = np.load(fp_p+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_discharge+'.npz',allow_pickle=True)
discharge_vars = discharge_data['discharge'][:,1]
discharge_vars = np.expand_dims(discharge_vars, axis=1)
discharge_varnames= ['Avg. discharge St-L. River']

loc_level = 'PointeClaire'
level_data = np.load(fp_p+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_level+'.npz',allow_pickle=True)
level_vars = level_data['level'][:,1]
level_vars = np.expand_dims(level_vars, axis=1)
level_varnames= ['Avg. level St-L. River']

loc_levelO = 'SteAnnedeBellevue'
levelO_data = np.load(fp_p+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_levelO+'.npz',allow_pickle=True)
levelO_vars = levelO_data['level'][:,1]
levelO_vars = np.expand_dims(levelO_vars, axis=1)
levelO_varnames= ['Avg. level Ottawa River']

#%%
# Load climate indices
index_list   = ['AMO',    'SOI',    'NAO',  'PDO',    'ONI',    'AO',   'PNA',  'WP',     'TNH',    'SCAND',  'PT',     'POLEUR', 'EPNP',   'EA']
timerep_list = ['monthly','monthly','daily','monthly','monthly','daily','daily','monthly','monthly','monthly','monthly','monthly','monthly','monthly']
ci_varnames = []
ci_vars = np.zeros((len(time),len(index_list)))*np.nan
for i,iname in enumerate(index_list):
    if iname == 'PDO':
        # vexp = 'ersstv3'
        # vexp = 'ersstv5'
        vexp = 'hadisst1'
        fn = iname+'_index_'+timerep_list[i]+'_'+vexp+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['PDO_data'][365:])
    elif iname == 'ONI':
        fn = iname+'_index_'+timerep_list[i]+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['ONI_data'][365:])
    else:
        fn = iname+'_index_'+timerep_list[i]+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['data'][365:])
    ci_varnames += [iname]

#%%%%%%% GET MONTHLY VARIABLES %%%%%%%%%
# Average for Jan, Feb, Mar, April,...., Dec
monthly_vars_tmp = np.zeros((100,len(years),12))*np.nan
monthly_vars_in =['weather_vars',
                  'ci_vars',
                  'discharge_vars',
                  'level_vars',
                  'levelO_vars'
                  ] + ['avg_Twater_vars']*(FUD_loc=='Longueuil')

nvars_monthly = 0
varnames_monthly_i = []
for i,var in enumerate(monthly_vars_in):
    if var == 'weather_vars':
        monthly_weather_vars = get_monthly_vars_from_daily(weather_vars,weather_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_weather_vars.shape[0],:,:] = monthly_weather_vars
        nvars_monthly += monthly_weather_vars.shape[0]
        varnames_monthly_i += [weather_varnames[i] for i in range(len(weather_varnames))]
    if var == 'avg_Twater_vars':
        monthly_avg_Twater_vars = get_monthly_vars_from_daily(avg_Twater_vars,avg_Twater_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_avg_Twater_vars.shape[0],:,:] = monthly_avg_Twater_vars
        nvars_monthly += monthly_avg_Twater_vars.shape[0]
        varnames_monthly_i += avg_Twater_varnames
    if var == 'ci_vars':
        monthly_ci_vars = get_monthly_vars_from_daily(ci_vars,['Avg.' + index_list[j] for j in range(len(index_list)) ],years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_ci_vars.shape[0],:,:] = monthly_ci_vars
        nvars_monthly += monthly_ci_vars.shape[0]
        varnames_monthly_i += ci_varnames
    if var == 'discharge_vars':
        monthly_discharge_vars = get_monthly_vars_from_daily(discharge_vars,discharge_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_discharge_vars.shape[0],:,:] = monthly_discharge_vars
        nvars_monthly += monthly_discharge_vars.shape[0]
        varnames_monthly_i += discharge_varnames
    if var == 'level_vars':
        monthly_level_vars = get_monthly_vars_from_daily(level_vars,level_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_level_vars.shape[0],:,:] = monthly_level_vars
        nvars_monthly += monthly_level_vars.shape[0]
        varnames_monthly_i += level_varnames
    if var == 'levelO_vars':
        monthly_levelO_vars = get_monthly_vars_from_daily(levelO_vars,levelO_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_levelO_vars.shape[0],:,:] = monthly_levelO_vars
        nvars_monthly += monthly_levelO_vars.shape[0]
        varnames_monthly_i += levelO_varnames

monthly_vars = monthly_vars_tmp[0:nvars_monthly,:,:]

if detrend:
    for ivar in range(nvars_monthly):
        for imonth in range(12):
            if anomaly == 'linear':
                monthly_vars[ivar,:,imonth], [m,b] = detrend_ts(monthly_vars[ivar,:,imonth],years,anomaly)
            if anomaly == 'mean':
                monthly_vars[ivar,:,imonth], mean = detrend_ts(monthly_vars[ivar,:,imonth],years,anomaly)


#%%
if save_monthly_vars:
    save_name = 'monthly_vars_'+FUD_loc+detrend*'_detrended'
    if FUD_loc == 'Longueuil':
        np.savez(local_path+'slice/data/monthly_predictors/'+save_name,
                  data=monthly_vars,
                  labels=varnames_monthly_i,
                  years = years,
                  time = time,
                  anomaly = anomaly,
                  region_ERA5 = region,
                  Twater_loc_list = Twater_loc_list,
                  loc_discharge = loc_discharge,
                  loc_level = loc_level,
                  loc_levelO = loc_levelO,
                  date_ref=date_ref)
    if FUD_loc == 'Beauharnois':
        np.savez(local_path+'slice/data/monthly_predictors/'+save_name,
                  data=monthly_vars,
                  labels=varnames_monthly_i,
                  years = years,
                  time = time,
                  anomaly = anomaly,
                  region_ERA5 = region,
                  loc_discharge = loc_discharge,
                  loc_level = loc_level,
                  loc_levelO = loc_levelO,
                  date_ref=date_ref)