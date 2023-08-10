#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:44:10 2021

@author: Amelie
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
from functions_MLR import update_monthly_NAO_index

#%%
# UPDATE WATER LEVEL DATA
# Note: level data from water.gc.ca cannot be downloaded directly from script.
#       The csv file corresponding to the 'update_datestr' must be available in the
#       'raw_fp' directory for this update to work.
raw_fp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/QL_ECCC/'
processed_fp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/water_levels_discharge_ECCC/'
loc_name = 'PointeClaire'
loc_nb = '02OA039'
update_date = dt.date(2021,11,30)
update_datestr = update_date.strftime("%b")+'-'+str(update_date.day).rjust(2, '0')+'-'+str(update_date.year)
# update_str = dt.date.today().strftime("%b")+'-'+str(dt.date.today().day)+'-'+str(dt.date.today().year)
# update_datestr = 'Nov-11-2021'

data_available, n = datecheck_var_npz('level',processed_fp+'water_levels_discharge_'+loc_name,
                  date_check = update_date,past_days=30,n=0.75)
print('Water level', data_available, n)

if not data_available:
    print('Updating water level data... ')
    level = update_water_level(update_datestr,loc_name,loc_nb,raw_fp,processed_fp,save=True)
    data_available, n = datecheck_var_npz('level',processed_fp+'water_levels_discharge_'+loc_name,
                      date_check = update_date,past_days=30,n=0.75)
    print('Done!', data_available, n)


#%%
# UPDATE ERA5 VARIABLES
region = 'D'
fpath_ERA5_processed = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/ERA5/region'+region+'/'
fpath_ERA5_raw = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/ERA5_test/region'+region+'/'
var_list      = ['u10',                    'v10',                    't2m',           't2m',           't2m',           'd2m',                    'licd',          'lict',                'ltlt',                        'msl',                    'ro',    'siconc',       'sst',                    'sf',      'smlt',    'tcc',              'tp',                 'windspeed','RH',  'SH',  'FDD',  'TDD']
savename_list = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_temperature','2m_temperature','2m_dewpoint_temperature','lake_ice_depth','lake_ice_temperature','lake_total_layer_temperature','mean_sea_level_pressure','runoff','sea_ice_cover','sea_surface_temperature','snowfall','snowmelt','total_cloud_cover','total_precipitation','windspeed','RH',  'SH',  'FDD',  'TDD']
vartype_list  = ['mean',                   'mean',                   'mean',          'max',           'min',           'mean',                   'mean',          'mean',                'mean',                        'mean',                   'mean',  'mean',         'mean',                   'mean',    'mean',    'mean',             'mean',               'mean',     'mean','mean','mean','mean']

for ivar, var in enumerate(var_list):

    var = var_list[ivar]
    processed_varname = savename_list[ivar]
    var_type = 'daily'+vartype_list[ivar]
    update_startdate = dt.date(2021,11,25)
    update_enddate = dt.date(2021,11,25)

    data_available, n = datecheck_var_npz('data',fpath_ERA5_processed+'ERA5_'+var_type+'_'+processed_varname,
                        date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
    print(var, data_available, n)

    if not data_available:
        print('Updating '+ var +' data... ')
        data = update_ERA5_var(var,var_type,processed_varname,region,update_startdate,update_enddate,fpath_ERA5_raw,fpath_ERA5_processed,save=True)
        data_available, n = datecheck_var_npz('data',fpath_ERA5_processed+'ERA5_'+var_type+'_'+processed_varname,
                                              date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Done!', data_available, n)


#%%
# UPDATE MONTHLY NAO INDEX
NAO_processed_path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/NAO_index_NOAA/'
update_startdate = dt.date(2021,11,19)
update_enddate = dt.date(2021,11,29)
data_available, n = datecheck_var_npz('data',NAO_processed_path+'NAO_index_NOAA_monthly',
                                      date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
# print('NAO index',data_available, n)

if not data_available:
    print('Updating monthly NAO data... ')
    updated_NAO = update_NAO_index(NAO_processed_path, save = True)
    data_available, n = datecheck_var_npz('data',NAO_processed_path+'NAO_index_NOAA_monthly',
                                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
    print('Done!', data_available, n)


