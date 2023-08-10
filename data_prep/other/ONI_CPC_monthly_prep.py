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


file_path = local_path+'slice/data/raw/NOAA_climate_indices/ONI_CPC.csv'
ONI_monthly = read_csv(file_path,skip=1)
ONI_monthly[ONI_monthly < -99] = np.nan

date_ref = dt.date(1900,1,1)
date_start = dt.date(1979,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

# years = [1991,1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005,2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020,2021]
years = np.arange(1979,2022)

ONI = np.zeros((len(time),1))*np.nan


for it in range(len(time)):
    date_it = date_ref + dt.timedelta(days=int(time[it]))
    if date_it.year in years:
        ONI_year = ONI_monthly[np.where(ONI_monthly[:,0] == date_it.year)[0][0]]
        ONI_month = ONI_year[date_it.month]
        ONI[it,0] = ONI_month


save_path = local_path+'slice/data/processed/ONI_index_NOAA/'
savename = 'ONI_index_NOAA_monthly'
np.savez(save_path+'u'+savename,ONI_data=ONI)



