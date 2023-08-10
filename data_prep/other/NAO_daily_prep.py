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

#%%

file_path = local_path+'slice/data/raw/NAO_daily/norm.daily.nao.cdas.z500.csv'
NAO_daily = read_csv(file_path,skip=1)

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

NAO = np.zeros((len(time),1))*np.nan

for it in range(NAO_daily.shape[0]):

    date_time = (dt.date(int(NAO_daily[it,0]),int(NAO_daily[it,1]),int(NAO_daily[it,2]))-date_ref).days

    if date_time in time:
        time_it = np.where(time == date_time)[0][0]
        NAO[time_it,0] = NAO_daily[it,3]


save_path = local_path+'slice/data/processed/NAO_daily/'
savename = 'NAO_daily'
np.savez(save_path+'u'+savename,data=NAO)

#%%
# NAO_processed_path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/NAO_index_NOAA/'

# NAO_monthly = np.load(NAO_processed_path+'NAO_index_NOAA_monthly.npz')['data']


