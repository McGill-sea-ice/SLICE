#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:25:11 2022

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
import statsmodels.api as sm
import cartopy.crs as ccrs
import cartopy as cartopy
from functions import detect_FUD_from_Tw, detrend_ts
from functions_MLR import get_monthly_vars_from_daily


#%%
fdir_r = local_path+'slice/data/raw/CMC_GHRSST/'
fdir_p = local_path+'slice/data/processed/CMC_GHRSST/'

verbose = False
p_critical = 0.01

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


#%%
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

# Load Twater and FUD data
fp_p_Twater = local_path+'slice/data/processed/'
Twater_loc_list = ['Longueuil','Candiac','Atwater']
station_type = 'cities'
freezeup_opt = 2
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)
years = np.array(years[:-1])
avg_freezeup_doy = avg_freezeup_doy[:-1]

#%%
# Get monthly mean Twater data
monthly_avg_Twater = get_monthly_vars_from_daily(Twater_mean,['Avg. Twater'],years,time,replace_with_nan=False)
monthly_avg_Twater = np.squeeze(monthly_avg_Twater)

x = monthly_avg_Twater[:,11]
xlog = np.log10(monthly_avg_Twater[:,11])

y = avg_freezeup_doy

#%%
# Detrend Twater
detrend = True

if detrend:
    x_detrend, [m,b] = detrend_ts(x,years,'linear')
    xlog_detrend, [m,b] = detrend_ts(xlog,years,'linear')

#%%
# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3))
# ax.hist(x, bins=5)

# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3))
# ax.hist(x_detrend, bins=5)

# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3))
# ax.hist(xlog, bins=5)

# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3))
# ax.hist(xlog_detrend, bins=5)
#%%

# xplot = x; xlabel = 'T$_w$'
# xplot = xlog; xlabel = 'log$_{10}$(T$_w$)'
xplot = xlog_detrend; xlabel = 'log$_{10}$(T$_w$) Detrended Anomaly'


#%%
# WITH STD
fig_log,ax_log = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
ax_log.plot(xplot,y,'*')
ax_log.set_xlabel(xlabel)
ax_log.set_ylabel('FUD (DOY)')

ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*np.nanmean(xplot),np.arange(np.nanmin(y),np.nanmax(y)),'-',color='k')
ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*(np.nanmean(xplot)+np.nanstd(xplot)),np.arange(np.nanmin(y),np.nanmax(y)),':',color='k')
ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*(np.nanmean(xplot)-np.nanstd(xplot)),np.arange(np.nanmin(y),np.nanmax(y)),':',color='k')

ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*np.nanmean(y),'-',color=plt.get_cmap('tab20')(0))
ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*(np.nanmean(y)+np.nanstd(y)),':',color=plt.get_cmap('tab20')(0))
ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*(np.nanmean(y)-np.nanstd(y)),':',color=plt.get_cmap('tab20')(0))


#%%
# With 0.43*STD - LIKE CANSIPS
fig_log,ax_log = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
ax_log.plot(xplot,y,'*')
ax_log.set_xlabel(xlabel)
ax_log.set_ylabel('FUD (DOY)')

ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*np.nanmean(xplot),np.arange(np.nanmin(y),np.nanmax(y)),'-',color='k')
ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*(np.nanmean(xplot)+0.43*np.nanstd(xplot)),np.arange(np.nanmin(y),np.nanmax(y)),':',color='k')
ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*(np.nanmean(xplot)-0.43*np.nanstd(xplot)),np.arange(np.nanmin(y),np.nanmax(y)),':',color='k')


ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*np.nanmean(y),'-',color=plt.get_cmap('tab20')(0))
ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*(np.nanmean(y)+0.43*np.nanstd(y)),':',color=plt.get_cmap('tab20')(0))
ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*(np.nanmean(y)-0.43*np.nanstd(y)),':',color=plt.get_cmap('tab20')(0))
#%%
# With 0.55*STD (+/- ~6 days)
fig_log,ax_log = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
ax_log.plot(xplot,y,'*')
ax_log.set_xlabel(xlabel)
ax_log.set_ylabel('FUD (DOY)')

ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*np.nanmean(xplot),np.arange(np.nanmin(y),np.nanmax(y)),'-',color='k')
ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*(np.nanmean(xplot)+0.55*np.nanstd(xplot)),np.arange(np.nanmin(y),np.nanmax(y)),':',color='k')
ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*(np.nanmean(xplot)-0.55*np.nanstd(xplot)),np.arange(np.nanmin(y),np.nanmax(y)),':',color='k')


ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*np.nanmean(y),'-',color=plt.get_cmap('tab20')(0))
ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*(np.nanmean(y)+0.55*np.nanstd(y)),':',color=plt.get_cmap('tab20')(0))
ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*(np.nanmean(y)-0.55*np.nanstd(y)),':',color=plt.get_cmap('tab20')(0))

#%%
# WITH TERCILES
fig_log,ax_log = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
ax_log.plot(xplot,y,'*')
ax_log.set_xlabel(xlabel)
ax_log.set_ylabel('FUD (DOY)')

ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*np.nanmean(xplot),np.arange(np.nanmin(y),np.nanmax(y)),'-',color='k')
ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*(np.nanquantile(xplot,0.33)),np.arange(np.nanmin(y),np.nanmax(y)),':',color='k')
ax_log.plot(np.ones(len(np.arange(np.nanmin(y),np.nanmax(y))))*(np.nanquantile(xplot,0.66)),np.arange(np.nanmin(y),np.nanmax(y)),':',color='k')

ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*np.nanmean(y),'-',color=plt.get_cmap('tab20')(0))
ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*(np.nanquantile(y,0.33)),':',color=plt.get_cmap('tab20')(0))
ax_log.plot(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1),np.ones(len(np.arange(np.nanmin(xplot),np.nanmax(xplot),0.1)))*(np.nanquantile(y,0.66)),':',color=plt.get_cmap('tab20')(0))



