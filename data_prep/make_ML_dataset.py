#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 22:34:33 2023

@author: amelie
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
import pandas as pd
import seaborn as sns

from functions_MLR import datecheck_var_npz,update_water_level,update_monthly_NAO_index
from functions_MLR import update_ERA5_var,load_weather_vars_ERA5
from functions_MLR import update_daily_NAO_index,update_water_discharge
from functions_MLR import get_daily_var_from_monthly_cansips_forecasts,get_daily_var_from_seasonal_cansips_forecasts

from functions import detect_FUD_from_Tw


#%%%%%%% OPTIONS %%%%%%%%%
save = False

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
fp_r = local_path+'slice/data/raw/'
# Path of processed data
fp_p = local_path+'slice/data/processed/'


#%%%%%%% UPDATE VARIABLES %%%%%%%%%
update_ERA5_vars = False
update_level = False
update_discharge = False
update_daily_NAO = False
update_NAO_monthly = False

update_startdate = dt.date(2021,11,30)
# update_enddate = dt.date(2021,11,25)
update_enddate = dt.date.today()

if update_ERA5_vars:
    # UPDATE ERA5 VARIABLES
    region = 'D'
    fp_p_ERA5 = fp_p + 'ERA5_hourly/region'+region+'/'
    fp_r_ERA5 = fp_r + 'ERA5_hourly/region'+region+'/'
    var_list      = ['u10',                    'v10',                    't2m',           't2m',           't2m',           'd2m',                    'licd',          'lict',                'ltlt',                        'msl',                    'ro',    'siconc',       'sst',                    'sf',      'smlt',    'tcc',              'tp',                 'windspeed','RH',  'SH',  'FDD',  'TDD']
    savename_list = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_temperature','2m_temperature','2m_dewpoint_temperature','lake_ice_depth','lake_ice_temperature','lake_total_layer_temperature','mean_sea_level_pressure','runoff','sea_ice_cover','sea_surface_temperature','snowfall','snowmelt','total_cloud_cover','total_precipitation','windspeed','RH',  'SH',  'FDD',  'TDD']
    vartype_list  = ['mean',                   'mean',                   'mean',          'max',           'min',           'mean',                   'mean',          'mean',                'mean',                        'mean',                   'sum',   'mean',         'mean',                   'sum',     'sum',     'mean',             'sum',                'mean',     'mean','mean','mean','mean']

    for ivar, var in enumerate(var_list):

        var = var_list[ivar]
        processed_varname = savename_list[ivar]
        var_type = 'daily'+vartype_list[ivar]

        data_available, n = datecheck_var_npz('data',fp_p_ERA5+'ERA5_'+var_type+'_'+processed_varname,
                            date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print(var, data_available, n)

        if not data_available:
            print('Updating '+ var +' data... ')
            data = update_ERA5_var(var,var_type,processed_varname,region,update_startdate,update_enddate,fp_r_ERA5,fp_p_ERA5,save=True)
            data_available, n = datecheck_var_npz('data',fp_p_ERA5+'ERA5_'+var_type+'_'+processed_varname,
                                                  date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
            print('Done!', data_available, n)

if update_level:
    # UPDATE WATER LEVEL DATA
    # Note: level data from water.gc.ca cannot be downloaded directly from script.
    #       The csv file corresponding to the 'update_datestr' must be available in the
    #       'raw_fp' directory for this update to work.
    fp_r_QL = fp_r+'QL_ECCC/'
    fp_p_QL = fp_p+'water_levels_discharge_ECCC/'
    loc_name_list = ['PointeClaire','SteAnnedeBellevue']
    loc_nb_list = ['02OA039','02OA013']
    update_datestr = update_enddate.strftime("%b")+'-'+str(update_enddate.day)+'-'+str(update_enddate.year)

    for iloc in range(len(loc_name_list)):
        loc_name = loc_name_list[iloc]
        loc_nb = loc_nb_list[iloc]
        data_available, n = datecheck_var_npz('level',fp_p_QL+'water_levels_discharge_'+loc_name,
                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Water level - ' + loc_name , data_available, n)

        if not data_available:
            print('Updating water level data... ')
            level = update_water_level(update_datestr,loc_name,loc_nb,fp_r_QL,fp_p_QL,save=True)
            data_available, n = datecheck_var_npz('level',fp_p_QL+'water_levels_discharge_'+loc_name,
                              date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
            print('Done!', data_available, n)

if update_discharge:
    # UPDATE DISCHARGE DATA
    # Note: level data from water.gc.ca cannot be downloaded directly from script.
    #       The csv file corresponding to the 'update_datestr' must be available in the
    #       'raw_fp' directory for this update to work.
    fp_r_QL = fp_r+'QL_ECCC/'
    fp_p_QL = fp_p+'water_levels_discharge_ECCC/'
    loc_name = 'Lasalle'
    loc_nb = '02OA016'
    update_datestr = update_enddate.strftime("%b")+'-'+str(update_enddate.day)+'-'+str(update_enddate.year)

    data_available, n = datecheck_var_npz('discharge',fp_p_QL+'water_levels_discharge_'+loc_name,
                      date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
    print('Discharge', data_available, n)

    if not data_available:
        print('Updating discharge data... ')
        level = update_water_discharge(update_datestr,loc_name,loc_nb,fp_r_QL,fp_p_QL,save=True)
        data_available, n = datecheck_var_npz('discharge',fp_p_QL+'water_levels_discharge_'+loc_name,
                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Done!', data_available, n)

if update_daily_NAO:
    fp_r_NAOd = fp_r + 'NAO_daily/'
    fp_p_NAOd = fp_p + 'NAO_daily/'
    NAO_daily = update_daily_NAO_index(fp_r_NAOd,fp_p_NAOd,save=True)

    data_available, n = datecheck_var_npz('data',fp_p_NAOd+'NAO_daily',
                      date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
    print('NAO daily index', data_available, n)

    if not data_available:
        print('Updating NAO daily data... ')
        NAO_daily = update_daily_NAO_index(fp_r_NAOd,fp_p_NAOd,save=True)
        data_available, n = datecheck_var_npz('data',fp_p_NAOd+'NAO_daily',
                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Done!', data_available, n)

if update_NAO_monthly:
    # UPDATE MONTHLY NAO INDEX
    fp_p_NAOm = fp_p + 'NAO_index_NOAA/'
    data_available, n = datecheck_var_npz('data',fp_p_NAOm+'NAO_index_NOAA_monthly',
                                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
    print('NAO monthly index',data_available, n)

    if not data_available:
        print('Updating monthly NAO data... ')
        NAO_monthly = update_monthly_NAO_index(fp_p_NAOm, save = True)
        data_available, n = datecheck_var_npz('data',fp_p_NAOm+'NAO_index_NOAA_monthly',
                                              date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Done!', data_available, n)

#%%%%%%% LOAD VARIABLES %%%%%%%%%

# Load Twater and FUD data
fp_p_Twater = local_path+'slice/data/processed/'
# Twater_loc_list = ['Longueuil_updated','Candiac','Atwater']
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 2
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
                #'dailysum_snowmelt',
                'dailysum_total_precipitation',
                'dailymean_total_cloud_cover',
                'dailymean_windspeed',
                'daily_theta_wind',
                #'dailymean_RH',
                'dailymean_FDD',
                'dailymean_TDD',
                'dailymean_surface_solar_radiation_downwards',
                'dailymean_surface_latent_heat_flux',
                'dailymean_surface_sensible_heat_flux'
                ]
fp_p_ERA5 = fp_p + 'ERA5_hourly/region'+region+'/'
weather_vars, weather_varnames = load_weather_vars_ERA5(fp_p_ERA5,ERA5_varlist,region,time)
# weather_varnames = ['Avg. Ta_max','Avg. Ta_min','Avg. Ta_mean','Tot. TDD','Tot. FDD','Tot. precip.','Avg. SLP','Avg. wind speed','Avg. u-wind','Avg. v-wind','Tot. snowfall','Avg. cloud cover','Avg. spec. hum.','Avg. rel. hum.']

# # Load daily NAO data
# NAO_daily_data = np.load(fp_p+'NAO_daily/NAO_daily.npz',allow_pickle='TRUE')
# NAO_daily_vars = NAO_daily_data['data']
# NAO_daily_varnames = ['Avg. daily NAO']

# # Load monthly PDO data
# # fn = 'PDO_index_NOAA_monthly_ersstv3.npz'
# # fn = 'PDO_index_NOAA_monthly_ersstv5.npz'
# fn = 'PDO_index_NOAA_monthly_hadisst1.npz'
# PDO_data = np.load(fp_p+'PDO_index_NOAA/'+fn,allow_pickle='TRUE')
# PDO_vars = PDO_data['PDO_data']
# PDO_varnames = ['Avg. monthly PDO']

# # Load monthly EL NINO data
# fn = 'Nino34_index_NOAA_monthly.npz'
# ENSO_data = np.load(fp_p+'Nino34_index_NOAA/'+fn,allow_pickle='TRUE')
# ENSO_vars = ENSO_data['Nino34_data']
# ENSO_varnames = ['Avg. montlhy Nino34']
# # fn = 'ONI_index_NOAA_monthly.npz'
# # ENSO_data = np.load(fp_p+'ONI_index_NOAA/'+fn,allow_pickle='TRUE')
# # ENSO_vars = ENSO_data['ONI_data']
# # ENSO_varnames = ['ONI']

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


index_list   = ['AMO',    'SOI',    'NAO',  'PDO',    'ONI',    'AO',   'PNA',  'WP',     'TNH',    'SCAND',  'PT',     'POLEUR', 'EPNP',   'EA']
timerep_list = ['monthly','monthly','daily','monthly','monthly','daily','daily','monthly','monthly','monthly','monthly','monthly','monthly','monthly']
ci_varnames = []
ci_vars = np.zeros((len(time),len(index_list)))*np.nan
for i,iname in enumerate(index_list):
    if iname == 'PDO':
        # vexp = 'ersstv3'
        vexp = 'ersstv5'
        # vexp = 'hadisst1'
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


#%%%%%%% SAVE DATSET FOR ML MODELS %%%%%%%%%
if save:
    ntot = (avg_Twater_vars.shape[1]+
           weather_vars.shape[1]+
           # NAO_daily_vars.shape[1]+
           # PDO_vars.shape[1]+
           # ENSO_vars.shape[1]+
           ci_vars.shape[1]+
           discharge_vars.shape[1]+
           level_vars.shape[1]+
           levelO_vars.shape[1]
           )

    ds = np.zeros((len(time),ntot+1))
    labels = np.chararray(ntot+1, itemsize=100)
    save_name = 'ML_dataset'

    # Add time as first column:
    ds[:,0] = time
    labels[0] = 'Days since '+str(date_ref)

    # Add water temperature time series:
    ds[:,1] = avg_Twater_vars[:,0]
    labels[1] = avg_Twater_varnames[0]

    # Add weather data:
    for i in range(weather_vars.shape[1]):
        ds[:,2+i] = weather_vars[:,i]
        labels[2+i] = weather_varnames[i]

    # Add discharge and water level data:
    ds[:,2+weather_vars.shape[1]] = discharge_vars[:,0]
    labels[2+weather_vars.shape[1]] = discharge_varnames[0]

    ds[:,3+weather_vars.shape[1]] = level_vars[:,0]
    labels[3+weather_vars.shape[1]] = level_varnames[0]

    ds[:,4+weather_vars.shape[1]] = levelO_vars[:,0]
    labels[4+weather_vars.shape[1]] = levelO_varnames[0]

    # #Add NAO index:
    # ds[:,5+weather_vars.shape[1]] = NAO_daily_vars[:,0]
    # labels[5+weather_vars.shape[1]] = NAO_daily_varnames[0]

    # #Add PDO index:
    # ds[:,6+weather_vars.shape[1]] = PDO_vars[:,0]
    # labels[6+weather_vars.shape[1]] = PDO_varnames[0]

    # #Add ENSO index:
    # ds[:,7+weather_vars.shape[1]] = ENSO_vars[:,0]
    # labels[7+weather_vars.shape[1]] = ENSO_varnames[0]

    # Add Climate Indices
    for i in range(ci_vars.shape[1]):
        ds[:,5+weather_vars.shape[1]+i] = ci_vars[:,i]
        labels[5+weather_vars.shape[1]+i] = ci_varnames[i]


    # SAVE:
    np.savez(local_path+'slice/data/daily_predictors/ML_timeseries/'+save_name,
              data=ds,
              labels=labels,
              region_ERA5 = region,
              Twater_loc_list = Twater_loc_list,
              loc_discharge = loc_discharge,
              loc_level = loc_level,
              loc_levelO = loc_levelO,
              date_ref=date_ref)

