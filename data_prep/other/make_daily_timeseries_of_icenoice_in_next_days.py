#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:37:45 2022

@author: Amelie
"""

import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from functions import detect_FUD_from_Tw

#%%
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

years = np.array([1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021])

# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater, fud_dates = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False,return_FUD_dates = True)
freezeup_doy[np.where(years == 2020)] = np.nan

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)
Twater_mean[14269:14329] = 0.

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)

# Rearrange to get same time series as the one saved by 'save_predictor_daily_timeseries'
yr_start = 1992
date_ref = dt.date(1900,1,1)
it_start = np.where(time == (dt.date(yr_start,1,1)-date_ref).days)[0][0]
Twater_mean = Twater_mean[it_start:]
time = time[it_start:]
avg_freezeup_doy = avg_freezeup_doy[np.where(years==yr_start)[0][0]:]
fud_dates = fud_dates[np.where(years==yr_start)[0][0]:]
years = years[np.where(years==yr_start)[0][0]:]

#%%
fud_0d_10d = np.zeros(time.shape)
fud_0d_15d = np.zeros(time.shape)
fud_0d_20d = np.zeros(time.shape)
fud_0d_30d = np.zeros(time.shape)
fud_0d_40d = np.zeros(time.shape)
fud_0d_45d = np.zeros(time.shape)
fud_0d_50d = np.zeros(time.shape)
fud_0d_60d = np.zeros(time.shape)

for iyr,year in enumerate(years):
    fud_yr = fud_dates[iyr][0]
    fud_mn = fud_dates[iyr][1]
    fud_dy = fud_dates[iyr][2]
    if np.all(~np.isnan(fud_dates[iyr])):
        it_fud = np.where(time == (dt.date(int(fud_yr),int(fud_mn),int(fud_dy))-date_ref).days)[0][0]

        fud_0d_10d[it_fud-10:it_fud+1] = 1
        fud_0d_15d[it_fud-15:it_fud+1] = 1
        fud_0d_20d[it_fud-20:it_fud+1] = 1
        fud_0d_30d[it_fud-30:it_fud+1] = 1
        fud_0d_40d[it_fud-40:it_fud+1] = 1
        fud_0d_45d[it_fud-45:it_fud+1] = 1
        fud_0d_50d[it_fud-50:it_fud+1] = 1
        fud_0d_60d[it_fud-60:it_fud+1] = 1











