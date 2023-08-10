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
from functions import read_csv

index = 'AMO'
# index = 'SOI'

file_path = local_path+'slice/data/raw/NOAA_climate_indices/'+index+'_monthly.csv'
data_monthly = read_csv(file_path,skip=1)
data_monthly[data_monthly < -99] = np.nan

date_ref = dt.date(1900,1,1)
date_start = dt.date(1979,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.arange(1979,2022)

data = np.zeros((len(time),1))*np.nan

for it in range(len(time)):
    date_it = date_ref + dt.timedelta(days=int(time[it]))
    if date_it.year in years:
        data_year = data_monthly[np.where(data_monthly[:,0] == date_it.year)[0][0]]
        data_month = data_year[date_it.month]
        data[it,0] = data_month


save_path = local_path+'slice/data/processed/climate_indices_NOAA/'
savename = index+'_index_monthly'
np.savez(save_path+savename,data=data)

