#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:22:21 2022

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
import matplotlib.pyplot as plt
import datetime as dt
from functions import detect_FUD_from_Tw
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
save = True
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,3))
ax.grid()
ax.plot(years, np.ones(len(years))*np.nanmean(avg_freezeup_doy),'-')
yminus = np.nanmean(avg_freezeup_doy)-np.ones(len(years))*np.nanstd(avg_freezeup_doy)
yplus = np.nanmean(avg_freezeup_doy)+np.ones(len(years))*np.nanstd(avg_freezeup_doy)
ax.fill_between(years,yminus,yplus,alpha=0.1, color=plt.get_cmap('tab20')(0))
ax.plot(years,avg_freezeup_doy,'o-',color='k')
ax.set_xlim(1991,2020)
ax.set_xlabel('Years')
ax.set_ylabel('Freeze-Up Date (DOY)')


if save:
    savepath = local_path+'slice/figures/'
    savename = 'FUD_clim_'
    for i in range(len(Twater_loc_list)):
        savename+=Twater_loc_list[i]
    plt.savefig(savepath+savename+'.eps',bbox_inches='tight')
    plt.savefig(savepath+savename+'.png',dpi=700,bbox_inches='tight')


