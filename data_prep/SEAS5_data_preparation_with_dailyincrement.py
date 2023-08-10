#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:56:25 2022

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
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import calendar
from netCDF4 import Dataset
import netCDF4

from functions import ncdump
from functions import K_to_C

import warnings

from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir=local_path +'slice/prog/temp_files/') #python

from analysis.SEAS5.SEAS5_forecast_class import SEAS5frcst

import shutil


#%%

# r_dir = local_path + 'slice/data/raw/SEAS5/'
# base = 'SEAS5'
# region = 'D'
# ys = 1994
# ye = 2022

r_dir = local_path + 'slice/data/raw/SEAS51/'
base = 'SEAS51'
region = 'D'
ys = 1981
ye = 2017

feature_list = ['snowfall']#,'runoff','total_precipitation']



for feature in feature_list:
    for year in range(ys,ye):
        for month in range(12,13):

            extension = "{}{}.nc".format(year, str(month).rjust(2, '0'))
            path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month).rjust(2, '0'))
            fname = base + '_' + feature + '_' + extension
            # fname = base + '_' + feature + '_processed_' + extension

            # Initialize forecast class
            if os.path.isfile(path+fname):
                print(year, month)
                s = SEAS5frcst(path+fname)
                spatial_avg = False
                ensemble_avg = False
                time_rep = 'daily'

                # Load all data
                longitude,latitude,number,time_tmp,var = s.read_vars(['longitude', 'latitude', 'number','time',s.var],
                                          spatial_avg=spatial_avg,
                                          ensemble_avg= ensemble_avg,
                                          time_rep=time_rep
                                        )

                # plt.figure()
                # plt.plot(time_tmp,var[:,10,0,0])

                # Get daily increments from daily aggregation:
                var_incr = var.copy()
                var_incr[:-1] = var_incr[1:] - var_incr[0:-1]
                var_incr[-1] = np.ones((var_incr.shape[1],var_incr.shape[2],var_incr.shape[3]))*np.nan
                time = time_tmp.copy()
                # print(var.shape,var_incr.shape)

                # plt.figure()
                # plt.plot(time,var_incr[:,10,0,0])


                # Make new netcdf file with increment data
                fname_p = base + '_' + feature + '_processed_' + extension
                ncfile = netCDF4.Dataset(path+fname_p, 'w',format='NETCDF4')
                lon_dim = ncfile.createDimension('longitude',var.shape[3])
                lat_dim = ncfile.createDimension('latitude', var.shape[2])
                nb_dim = ncfile.createDimension('number', var.shape[1])
                time_dim = ncfile.createDimension('time', var.shape[0])

                lat = ncfile.createVariable('latitude', np.float32, ('latitude',))
                lat.units = 'degrees_north'
                lat.long_name = 'latitude'
                lon = ncfile.createVariable('longitude', np.float32, ('longitude',))
                lon.units = 'degrees_east'
                lon.long_name = 'longitude'
                nb = ncfile.createVariable('number', np.int32, ('number',))
                nb.long_name = 'ensemble_member'
                timev = ncfile.createVariable('time', np.int32, ('time',))
                timev.units = s.time_units
                timev.long_name = 'time'
                timev.calendar = 'gregorian'
                varnc = ncfile.createVariable(s.var, np.float32, ('time', 'number', 'latitude', 'longitude'))
                varnc.units = s.units
                varnc.long_name = s.long_name
                varnc.standard_name = s.std_name

                varnc[:] = var_incr
                timev[:] = time
                ncfile.close()

                # dataset = netCDF4.Dataset(path+fname_p, 'r')
                # var_nc = dataset[s.var][:]
                # time_nc = dataset['time'][:]
                # plt.figure()
                # plt.plot(time_nc,var_nc[:,0,0,0])
                # dataset.close()
