#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:40:18 2021

@author: Amelie
"""
import numpy as np
import scipy

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import running_nanmean,find_freezeup_Tw,find_freezeup_Tw_all_yrs

#%%
years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020]

fp = '../../../data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)

water_name_list = ['Atwater_cleaned_filled','DesBaillets_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
station_labels = ['Atwater','DesBaillets','Longueuil','Candiac']
station_type = 'cities'

#%%
# FIND FREEZEUP DATES FROM TWATER TIME SERIES, FROM THRESHOLD ON dTwdt
freezeup_dates = np.zeros((len(years)*4,3,len(water_name_list)))*np.nan
breakup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan


#-------------------------------------------------------
# FINAL OPTIONS:
# THE CHOICE IS BETWEEN OPTION 1B AND OPTION 3C.
# WE CHOOSE OPTION 3C FOR NOW, BUT WE COULD CHECK THE IMPACTS OF CHANGING FOR 1B LATER WITH MODEL ERROR.
# THE ORIGINAL DEFINITION (OPTION 1A) IS HERE ONLY FOR REFERENCE TO SEE IMPROVEMENT IN RESULTS.

# ORIGINAL DEFINITION (OPTION 1A): FOR REFERENCE ONLY - DO NOT USE THIS DEFINITION
# def_opt = 1
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = False
# T_thresh = 0.5
# dTdt_thresh = 0.25
# d2Tdt2_thresh = 0.25
# nd = 7
#-------------------------------------------------------
# OPTION 1B: THRESHOLD ON Tw, BUT BETTER THAN ORIGINAL DEFINITION
def_opt = 1
smooth_T =False; N_smooth = 3; mean_type='centered'
round_T = False; round_type= 'half_unit'
Gauss_filter = False
T_thresh = 0.75
dTdt_thresh = 0.25
d2Tdt2_thresh = 0.25
nd = 1
no_negTw = True
#-------------------------------------------------------
# # # OPTION 3C:- USING THRESHOLD ON DERIVATIVE OF GAUSSIAN FILTER (very similar to B, but even if the differences are bigger between stations, it seems to better capture what we would intuitively define as the freezeup....)
# def_opt = 3
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = True
# sig_dog = 3.5
# T_thresh = 3.
# dTdt_thresh = 0.15
# d2Tdt2_thresh = 0.15
# # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
# # d2Tdt2_thresh = 0.20
# nd = 30
# no_negTw = True
#-------------------------------------------------------

# APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST,
# BEFORE FINDING FREEZEUP DATES FROM WATER TEMP.
Twater = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_DoG1 = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_DoG2 = np.zeros((len(time),len(water_name_list)))*np.nan

fig_tw,ax_tw = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))
# fig_ftw,ax_ftw = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))

for iloc,loc in enumerate(water_name_list):
    print('-------------' + loc + '-------------')

    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][:,1]

    Twater[:,iloc] = Twater_tmp
    if loc == 'Candiac_cleaned_filled':
        Twater[:,iloc] = Twater_tmp-0.8
    if (loc == 'Atwater_cleaned_filled'):
        Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7


    Twater_tmp = Twater[:,iloc].copy()

    if round_T:
        if round_type == 'unit':
            Twater_tmp = np.round(Twater_tmp.copy())
        if round_type == 'half_unit':
            Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.

    if smooth_T:
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

    if no_negTw:
        Twater_tmp[Twater_tmp < 0] = 0.0

    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

    if Gauss_filter:
        Twater_dTdt[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
        Twater_d2Tdt2[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

    # for iyr, year in enumerate(years):
    #     iyr_start = np.where(time == (dt.date(year,10,1)-date_ref).days)[0][0]
    #     if year == 2020:
    #         iyr_end = np.where(time == (dt.date(year,12,31)-date_ref).days)[0][0]
    #     else:
    #         iyr_end = np.where(time == (dt.date(year+1,5,1)-date_ref).days)[0][0]

    #     fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp[iyr_start:iyr_end],Twater_dTdt[iyr_start:iyr_end,iloc],Twater_d2Tdt2[iyr_start:iyr_end,iloc],time[iyr_start:iyr_end],year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
    #     freezeup_dates[iyr,:,iloc] = fd

    #     ax_tw.plot(time[iyr_start:iyr_end],Twater_tmp[iyr_start:iyr_end],color=plt.get_cmap('tab20')(iloc*2+1))
    #     ax_tw.plot(time[iyr_start:iyr_end],T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2))

    label_width = 100
    ifz = 0
    for it in range(0,time.shape[0]):
        i_start = it
        i_end = it+label_width

        date_start = dt.timedelta(days=int(time[i_start])) + date_ref
        if date_start.month < 3:
            year = date_start.year-1
        else:
            year = date_start.year

        if year >= years[0]:
            iyr = np.where(years == year)[0][0]

            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp[i_start:i_end],Twater_dTdt[i_start:i_end,iloc],Twater_d2Tdt2[i_start:i_end,iloc],time[i_start:i_end],year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            # if np.sum(mask_freeze) > 0:
            #     if np.all(freezeup_dates[ifz-1,:,iloc] ==  fd ):
            #         ifz +=0
            #     else:
            #         freezeup_dates[ifz,:,iloc] = fd
            #         ifz += 1

            if (np.sum(mask_freeze) > 0) & (np.isnan(freezeup_dates[iyr,0,iloc])):
                    freezeup_dates[iyr,:,iloc] = fd

            ax_tw.plot(time[i_start:i_end],Twater_tmp[i_start:i_end],color=plt.get_cmap('tab20')(iloc*2+1))
            ax_tw.plot(time[i_start:i_end],T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2))


freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
for iloc,loc in enumerate(water_name_list):
    for iyr,year in enumerate(years):
        if ~np.isnan(freezeup_dates[iyr,0,iloc]):
            fd_yy = int(freezeup_dates[iyr,0,iloc])
            fd_mm = int(freezeup_dates[iyr,1,iloc])
            fd_dd = int(freezeup_dates[iyr,2,iloc])

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365

            freezeup_doy[iyr,iloc]=fd_doy

# PLOT FREEZEUP DOY TIME SERIES
fig_fddoy,ax_fddoy = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))
for iloc,loc in enumerate(water_name_list):
    ax_fddoy.plot(years,freezeup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2+1), label=station_labels[iloc],alpha=0.5)

ax_fddoy.legend(bbox_to_anchor=(0.0, 0.64, 0.3, 0.3),fontsize=8)
ax_fddoy.set_ylim(300,430)
ax_fddoy.set_xlabel('Years')
ax_fddoy.set_ylabel('Freezeup DOY')
ax_fddoy.grid()



#%%
# FIND FREEZEUP DATES FROM TWATER TIME SERIES, FROM THRESHOLD ON dTwdt
freezeup_dates2 = np.zeros((len(years),3,len(water_name_list)))*np.nan


# APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST,
# BEFORE FINDING FREEZEUP DATES FROM WATER TEMP.
Twater = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_DoG1 = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_DoG2 = np.zeros((len(time),len(water_name_list)))*np.nan

fig_tw,ax_tw = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))

for iloc,loc in enumerate(water_name_list):
    print('-------------' + loc + '-------------')

    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][:,1]

    Twater[:,iloc] = Twater_tmp
    if loc == 'Candiac_cleaned_filled':
        Twater[:,iloc] = Twater_tmp-0.8
    if (loc == 'Atwater_cleaned_filled'):
        Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7


    Twater_tmp = Twater[:,iloc].copy()

    if round_T:
        if round_type == 'unit':
            Twater_tmp = np.round(Twater_tmp.copy())
        if round_type == 'half_unit':
            Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.

    if smooth_T:
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

    if no_negTw:
        Twater_tmp[Twater_tmp < 0] = 0.0

    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

    if Gauss_filter:
        Twater_dTdt[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
        Twater_d2Tdt2[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

    fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
    freezeup_dates2[:,:,iloc] = fd

    ax_tw.plot(time,Twater_tmp,color=plt.get_cmap('tab20')(iloc*2+1))
    ax_tw.plot(time,T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2))
    # ax_tw.plot(T_breakup, 'o',color=plt.get_cmap('tab20')(iloc*2))


freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
breakup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
for iloc,loc in enumerate(water_name_list):
    for iyr,year in enumerate(years):
        if ~np.isnan(freezeup_dates2[iyr,0,iloc]):
            fd_yy = int(freezeup_dates2[iyr,0,iloc])
            fd_mm = int(freezeup_dates2[iyr,1,iloc])
            fd_dd = int(freezeup_dates2[iyr,2,iloc])

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365

            freezeup_doy[iyr,iloc]=fd_doy

# PLOT FREEZEUP DOY TIME SERIES
fig_fddoy,ax_fddoy = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))
for iloc,loc in enumerate(water_name_list):
    ax_fddoy.plot(years,freezeup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2+1), label=station_labels[iloc],alpha=0.5)

ax_fddoy.legend(bbox_to_anchor=(0.0, 0.64, 0.3, 0.3),fontsize=8)
ax_fddoy.set_ylim(300,430)
ax_fddoy.set_xlabel('Years')
ax_fddoy.set_ylabel('Freezeup DOY')
ax_fddoy.grid()

