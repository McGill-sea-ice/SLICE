#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:15:10 2021

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
from functions import read_csv, ncdump
from netCDF4 import Dataset

file_path = local_path+'slice/data/raw/NOAA_climate_indices/'

# fn = 'pdo.timeseries.ersstv3b.nc'
# savename = 'PDO_index_monthly_ersstv3'

fn = 'pdo.timeseries.ersstv5.nc'
savename = 'PDO_index_monthly_ersstv5'

# fn = 'pdo.timeseries.hadisst1.1.nc'
# savename = 'PDO_index_monthly_hadisst1'

ncid = Dataset(file_path+fn, 'r')
ncid.set_auto_mask(False)
# ncdump(ncid)
time_tmp = ncid.variables['time'][:]
PDO_tmp = ncid.variables['pdo'][:]

nc_date_ref = dt.date(1800,1,1)

date_ref = dt.date(1900,1,1)
date_start = dt.date(1979,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

PDO = np.zeros((len(time),1))*np.nan

#%%
nc_date = np.zeros(time.shape)
for it in range(len(time)):
    date_it = date_ref + dt.timedelta(days=int(time[it]))
    nc_date[it] = (dt.date(date_it.year,date_it.month,1)-nc_date_ref).days


for it in range(len(time)):
    date_it = date_ref + dt.timedelta(days=int(time[it]))

    if np.any(np.where(time_tmp == nc_date[it])[0]):
        nc_it = np.where(time_tmp == nc_date[it])[0][0]
        PDO[it,0] = PDO_tmp[nc_it]


save_path = local_path+'slice/data/processed/climate_indices_NOAA/'
np.savez(save_path+savename,PDO_data=PDO)



#%%
plt.figure();plt.plot(time,PDO)

