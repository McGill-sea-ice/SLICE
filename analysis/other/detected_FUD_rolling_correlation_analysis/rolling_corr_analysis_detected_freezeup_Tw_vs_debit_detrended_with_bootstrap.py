#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:45:58 2021

@author: Amelie
"""

import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import scipy
from scipy import ndimage

import pandas
from statsmodels.formula.api import ols

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import running_nanmean,find_freezeup_Tw_all_yrs,season_mask
from functions import linear_fit, rolling_climo, r_confidence_interval
from functions import detrend_ts, get_window_vars, deseasonalize_ts

#%%
def bootstrap(xvar_in, yvar_in, nboot=1000):

    nyears = len(xvar_in)
    r_out = np.zeros((nboot))*np.nan

    for n in range(nboot):
        if nboot >1:
            boot_indx = np.random.choice(nyears,size=nyears,replace=True)
        else:
            boot_indx = np.random.choice(nyears,size=nyears,replace=False)


        xvar_boot = xvar_in[boot_indx].copy()
        yvar_boot = yvar_in[boot_indx].copy()

        lincoeff, Rsqr = linear_fit(xvar_boot,yvar_boot)

        r_out[n] = np.sqrt(Rsqr)
        if (lincoeff[0]< 0):
            r_out[n] *= -1

    return r_out


#%%
years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

fp = '../../../data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)

#%%
end_dates_arr = np.zeros((len(years),4))*np.nan
for iyear,year in enumerate(years):
    doy_sep1 = (dt.date(int(year),9,1)-(dt.date(int(year),1,1))).days + 1
    doy_oct1 = (dt.date(int(year),10,1)-(dt.date(int(year),1,1))).days + 1
    doy_nov1 = (dt.date(int(year),11,1)-(dt.date(int(year),1,1))).days + 1
    doy_dec1 = (dt.date(int(year),12,1)-(dt.date(int(year),1,1))).days + 1
    doy_dec15 = (dt.date(int(year),12,15)-(dt.date(int(year),1,1))).days + 1
    # end_dates_arr[iyear,0] = doy_dec15
    end_dates_arr[iyear,0] = doy_dec1
    end_dates_arr[iyear,1] = doy_nov1
    end_dates_arr[iyear,2] = doy_oct1
    end_dates_arr[iyear,3] = doy_sep1
enddate_labels = ['Dec. 1st', 'Nov. 1st', 'Oct. 1st', 'Sept. 1st']

p_critical = 0.05

deseasonalize = False
detrend = True
anomaly = 'linear'

nboot = 1

#window_arr = 2*2**np.arange(0,8) # For powers of 2
# window_arr = np.arange(1,39)*7 # For weeks, up to Jan 1st
# window_arr = np.arange(1,3)*7
window_arr = np.arange(1,9)*30 # For months

#%%
# LOAD FREEZEUP DATES OR FIND FROM TWATER TIME SERIES
water_name_list = ['Longueuil_cleaned_filled']
station_labels = ['Longueuil']
station_type = 'cities'

water_name_list = ['Atwater_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
station_labels = ['Atwater','Longueuil','Candiac']
station_type = 'cities'

# water_name_list = ['Longueuil_cleaned_filled','Candiac_cleaned_filled']
# station_labels = ['Longueuil','Candiac']
# station_type = 'cities'

# water_name_list = ['Atwater_cleaned_filled','Longueuil_cleaned_filled']
# station_labels = ['Atwater','Longueuil']
# station_type = 'cities'

# water_name_list = ['Candiac_cleaned_filled']
# station_labels = ['Candiac']
# station_type = 'cities'

load_freezeup = False
freezeup_opt = 2
month_start_day = 1

# OPTION 1
if freezeup_opt == 1:
    def_opt = 1
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = False
    T_thresh = 0.75
    dTdt_thresh = 0.25
    d2Tdt2_thresh = 0.25
    nd = 1

# OPTION 2
if freezeup_opt == 2:
    def_opt = 3
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = True
    sig_dog = 3.5
    T_thresh = 3.
    dTdt_thresh = 0.15
    d2Tdt2_thresh = 0.15
    # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
    # d2Tdt2_thresh = 0.20
    nd = 30

if load_freezeup:
    print('ERROR: STILL NEED TO DEFINE THIS FROM SAVED ARRAYS...')
else:
    freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
    freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
    freezeup_temp = np.zeros((len(years),len(water_name_list)))*np.nan

    Twater = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_DoG1 = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_DoG2 = np.zeros((len(time),len(water_name_list)))*np.nan

    for iloc,loc in enumerate(water_name_list):
        loc_water_loc = water_name_list[iloc]
        water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
        Twater_tmp = water_loc_data['Twater'][:,1]

        # APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST
        Twater[:,iloc] = Twater_tmp
        if loc == 'Candiac_cleaned_filled':
            Twater[:,iloc] = Twater_tmp-0.8
        if (loc == 'Atwater_cleaned_filled'):
            Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7


        # THEN FIND DTDt, D2TDt2, etc.
        Twater_tmp = Twater[:,iloc].copy()
        if round_T:
            if round_type == 'unit':
                Twater_tmp = np.round(Twater_tmp.copy())
            if round_type == 'half_unit':
                Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
        if smooth_T:
            Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

        dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

        dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
        dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
        dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

        Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
        Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

        if Gauss_filter:
            Twater_DoG1[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
            Twater_DoG2[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

        # THEN FIND FREEZEUP ACCORDING TO CHOSEN OPTION:
        if def_opt == 3:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_DoG1[:,iloc],Twater_DoG2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
            freezeup_temp[:,iloc] = ftw
        else:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
            freezeup_temp[:,iloc] = ftw

        # FINALLY, TRANSFORM FREEZEUP FROM DATE FORMAT TO DOY FORMAT:
        for iyr,year in enumerate(years):
            if ~np.isnan(freezeup_dates[iyr,0,iloc]):
                fd_yy = int(freezeup_dates[iyr,0,iloc])
                fd_mm = int(freezeup_dates[iyr,1,iloc])
                fd_dd = int(freezeup_dates[iyr,2,iloc])

                fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
                if fd_doy < 60: fd_doy += 365

                freezeup_doy[iyr,iloc]=fd_doy

# Average all stations to get mean freezeup DOY for each year
avg_freezeup_doy = np.round(np.nanmean(freezeup_doy,axis=1))

# end_dates_arr = np.zeros((len(years),1))*np.nan
# end_dates_arr[:,0] = avg_freezeup_doy


#%%
# MAKE TWATER INTO AN EXPLANATORY VARIABLE
Twater_varnames = ['Avg. water temp.']
Twater_vars = np.zeros((len(time),len(Twater_varnames)))*np.nan
Twater_vars[:,0] = np.nanmean(Twater,axis=1)
Twater_vars = np.squeeze(Twater_vars)

if deseasonalize:
    Nwindow = 31
    Twater_vars = deseasonalize_ts(Nwindow,Twater_vars,['Twater'],'all_time',time,years)

Twater_vars_all = np.zeros((1,len(years),len(window_arr),end_dates_arr.shape[1],2,1))*np.nan
for iend in range(end_dates_arr.shape[1]):
    Twater_vars_all[:,:,:,iend,:,0] = get_window_vars(np.expand_dims(Twater_vars,axis=1),['Avg. Twater'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)


#%%
# LOAD DISCHARGE AND LEVEL DATA

loc_discharge = 'Lasalle'
loc_level = 'PointeClaire'

discharge_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_discharge+'.npz',allow_pickle=True)
level_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_level+'.npz',allow_pickle=True)

discharge_vars = discharge_data['discharge'][:,1]
level_vars = level_data['level'][:,1]

if deseasonalize:
    Nwindow = 31
    discharge_vars = deseasonalize_ts(Nwindow,discharge_vars,['discharge'],'all_time',time,years)
    level_vars = deseasonalize_ts(Nwindow,level_vars,['level'],'all_time',time,years)

# Separate in different windows with different end dates
discharge_vars_all = np.zeros((1,len(years),len(window_arr),end_dates_arr.shape[1],2,1))*np.nan
level_vars_all = np.zeros((1,len(years),len(window_arr),end_dates_arr.shape[1],2,1))*np.nan
for iend in range(end_dates_arr.shape[1]):
    discharge_vars_all[:,:,:,iend,:,0] = get_window_vars(np.expand_dims(discharge_vars,axis=1),['Avg. discharge'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
    level_vars_all[:,:,:,iend,:,0] = get_window_vars(np.expand_dims(level_vars,axis=1),['Avg. level'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

vars_all = np.zeros((2,len(years),len(window_arr),end_dates_arr.shape[1],2,1))*np.nan
vars_all[0,:,:,:,:,:] = discharge_vars_all.copy()
vars_all[1,:,:,:,:,:] = level_vars_all.copy()

varnames = ['discharge', 'level']

#%%
# LOAD WEATHER DATA

fp2 = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/detected_freezeup_correlation_analysis/'

weather_data1 = np.load(fp2+'weather_vars_all_monthly.npz',allow_pickle='TRUE')
weather_data2 = np.load(fp2+'weather_vars2_all_monthly.npz',allow_pickle='TRUE')

weather_vars_all1 = weather_data1['weather_vars']
weather_varnames1 = weather_data1['varnames']
weather_locnames1 = weather_data1['locnames']

weather_vars_all2 = weather_data2['weather_vars']
weather_varnames2 = weather_data2['varnames']
weather_locnames2 = weather_data2['locnames']

weather_vars_all1 = weather_vars_all1[:,:,:,:,:,0] # Select location: MLO+OR
weather_vars_all2 = weather_vars_all2[:,:,:,:,:,0] # Select location: MLO+OR
weather_locname = weather_locnames1[0]

# MERGE ALL WEATHER DATA TOGETHER:
weather_vars_all = np.zeros((weather_vars_all1.shape[0]+weather_vars_all2.shape[0],weather_vars_all1.shape[1],weather_vars_all1.shape[2],weather_vars_all1.shape[3],2,1))
weather_vars_all[0:weather_vars_all1.shape[0],:,:,:,:,0] = weather_vars_all1[:,:,:,:,:]
weather_vars_all[weather_vars_all1.shape[0]:,:,:,:,:,0] = weather_vars_all2[:,:,:,:,:]

weather_varnames = [n for n in weather_varnames1]+[n for n in weather_varnames2]


#%%
nvars = vars_all.shape[0]
nyears = vars_all.shape[1]
nwindows = vars_all.shape[2]
nend = vars_all.shape[3]
nwindowtype = vars_all.shape[4]
nlocs = vars_all.shape[5]


if detrend:
    vars_all_detrended = np.zeros(vars_all.shape)*np.nan
    Twater_vars_all_detrended = np.zeros(Twater_vars_all.shape)*np.nan
    weather_vars_all_detrended = np.zeros(weather_vars_all.shape)*np.nan
    for ivar in range(nvars):
        for iw in range(nwindows):
            for iend in range(nend):
                for ip in range(nwindowtype):
                    for iloc in range(nlocs):
                        xvar = vars_all[ivar,:,iw,iend,ip,iloc]
                        yvar = avg_freezeup_doy

                        vars_all_detrended[ivar,:,iw,iend,ip,iloc]= detrend_ts(xvar,years,anomaly)
                        avg_freezeup_doy_detrended = detrend_ts(yvar,years,anomaly)

    for iw in range(nwindows):
        for iend in range(nend):
            for ip in range(nwindowtype):
                for iloc in range(nlocs):
                    Txvar = Twater_vars_all[0,:,iw,iend,ip,iloc].copy()

                    Twater_vars_all_detrended[0,:,iw,iend,ip,iloc]= detrend_ts(Txvar,years,anomaly)

    for ivar in range(weather_vars_all.shape[0]):
        for iw in range(nwindows):
            for iend in range(nend):
                for ip in range(nwindowtype):
                    for iloc in range(weather_vars_all.shape[5]):
                        xvar = weather_vars_all[ivar,:,iw,iend,ip,iloc].copy()

                        weather_vars_all_detrended[ivar,:,iw,iend,ip,iloc]= detrend_ts(xvar,years,anomaly)

else:
    vars_all_detrended = vars_all.copy()
    avg_freezeup_doy_detrended = avg_freezeup_doy.copy()
    Twater_vars_all_detrended = Twater_vars_all.copy()
    weather_vars_all_detrended = weather_vars_all.copy()


r_mean = np.zeros((nvars,nwindows,nend,nwindowtype,nlocs))*np.nan
r_std = np.zeros((nvars,nwindows,nend,nwindowtype,nlocs))*np.nan
for ivar in range(nvars):
    for iw in range(nwindows):
        for iend in range(nend):
            for ip in range(nwindowtype):
                for iloc in range(nlocs):

                    print('Bootstrap '+ varnames[ivar]+' - %3i'%(iw)+'/%3i'%(nwindows)+', %3i'%(iend)+'/%3i'%(nend)+', %3i'%(ip)+'/%3i'%(nwindowtype)+', %3i'%(iloc)+'/%3i'%(nlocs))

                    xvar = vars_all_detrended[ivar,:,iw,iend,ip,iloc]
                    yvar = avg_freezeup_doy_detrended

                    r = bootstrap(xvar,yvar,nboot)
                    r_mean[ivar,iw,iend,ip,iloc] = np.nanmean(r)
                    r_std[ivar,iw,iend,ip,iloc] = np.nanstd(r)


Twater_r_mean = np.zeros((1,nwindows,nend,nwindowtype,nlocs))*np.nan
Twater_r_std = np.zeros((1,nwindows,nend,nwindowtype,nlocs))*np.nan
ivar = 0
for iw in range(nwindows):
    for iend in range(nend):
        for ip in range(nwindowtype):
            for iloc in range(nlocs):

                print('Bootstrap '+ Twater_varnames[ivar]+' - %3i'%(iw)+'/%3i'%(nwindows)+', %3i'%(iend)+'/%3i'%(nend)+', %3i'%(ip)+'/%3i'%(nwindowtype)+', %3i'%(iloc)+'/%3i'%(nlocs))

                xvar = Twater_vars_all_detrended[ivar,:,iw,iend,ip,iloc]
                yvar = avg_freezeup_doy_detrended

                Twater_r = bootstrap(xvar,yvar,nboot)
                Twater_r_mean[ivar,iw,iend,ip,iloc] = np.nanmean(Twater_r)
                Twater_r_std[ivar,iw,iend,ip,iloc] = np.nanstd(Twater_r)


weather_r_mean = np.zeros((weather_vars_all.shape[0],nwindows,nend,nwindowtype,weather_vars_all.shape[5]))*np.nan
weather_r_std = np.zeros((weather_vars_all.shape[0],nwindows,nend,nwindowtype,weather_vars_all.shape[5]))*np.nan
for ivar in range(weather_vars_all.shape[0]):
    for iw in range(nwindows):
        for iend in range(nend):
            for ip in range(nwindowtype):
                for iloc in range(weather_vars_all.shape[5]):

                    print('Bootstrap '+ weather_varnames[ivar]+' - %3i'%(iw)+'/%3i'%(nwindows)+', %3i'%(iend)+'/%3i'%(nend)+', %3i'%(ip)+'/%3i'%(nwindowtype)+', %3i'%(iloc)+'/%3i'%(nlocs))

                    xvar = weather_vars_all_detrended[ivar,:,iw,iend,ip,iloc]
                    yvar = avg_freezeup_doy_detrended

                    weather_r = bootstrap(xvar,yvar,nboot)
                    weather_r_mean[ivar,iw,iend,ip,iloc] = np.nanmean(weather_r)
                    weather_r_std[ivar,iw,iend,ip,iloc] = np.nanstd(weather_r)


#%%
# CORRELATION PLOT FOR DISCHARGE & LEVEL:

pc = p_critical

rc_m1, rc_p1 = r_confidence_interval(0,pc,nyears,tailed='one')
rc_m2, rc_p2 = r_confidence_interval(0,pc,nyears,tailed='two')
plot_colors = [plt.get_cmap('tab20b')(0),plt.get_cmap('tab20b')(4),plt.get_cmap('tab20b')(8),plt.get_cmap('tab20b')(12)]

for iend in range(0,1):
# for iend in range(1,2):
# for iend in range(2,3):
# for iend in range(3,4):
    enddate_str = enddate_labels[iend]

    for ivar in range(nvars):
        var1 = varnames[ivar]
        nrows = nlocs
        ncols = 1
        fig,ax = plt.subplots(nrows,figsize=(5,(nlocs)*(10/5.)),sharex=True,sharey=True,squeeze=False)
        if (nrows == 1) | (ncols == 1) :
            ax = ax.reshape(-1)
        plt.suptitle(var1)
        iloc = 0

        for ip in range(nwindowtype):
            r_mean_plot = r_mean[ivar,:,iend,ip,iloc]
            r_std_plot = r_std[ivar,:,iend,ip,iloc]

            ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plot_colors[ip])
            ax[iloc].fill_between(window_arr,r_mean_plot+r_std_plot,r_mean_plot-r_std_plot,color=plot_colors[ip],alpha=0.15)

            ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_p2,':', color='gray')
            ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_m2,':', color='gray')

            plt.subplots_adjust(left=0.2,right=0.9,bottom=0.23)
            ax[iloc].set_xlim(0,np.nanmax(window_arr)+np.nanmax(window_arr)/10.)
            ax[iloc].set_ylim(-1,1)
            ax[iloc].set_ylabel('r\n',fontsize=10)
            ax[iloc].grid()

            if iloc == nlocs-1:
                if (window_arr[1]-window_arr[0]) == 7:
                    ax[iloc].set_xlabel('Previous X weeks (XW) from '+enddate_str,fontsize=10)
                    labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'W' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                    labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                if (window_arr[1]-window_arr[0]) == 30:
                    ax[iloc].set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
                    labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                    labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                ax[iloc].set_xticks(window_arr)
                ax[iloc].set_xticklabels(labels_list)

#%%
# CORRELATION PLOT FOR TWATER:

pc = p_critical

rc_m1, rc_p1 = r_confidence_interval(0,pc,nyears,tailed='one')
rc_m2, rc_p2 = r_confidence_interval(0,pc,nyears,tailed='two')
plot_colors = [plt.get_cmap('tab20b')(0),plt.get_cmap('tab20b')(4),plt.get_cmap('tab20b')(8),plt.get_cmap('tab20b')(12)]
ivar = 0

for iend in range(0,1):
# for iend in range(1,2):
# for iend in range(2,3):
# for iend in range(3,4):
    enddate_str = enddate_labels[iend]

    var1 = Twater_varnames[ivar]
    nrows = nlocs
    ncols = 1
    fig,ax = plt.subplots(nrows,figsize=(5,(nlocs)*(10/5.)),sharex=True,sharey=True,squeeze=False)
    if (nrows == 1) | (ncols == 1) :
        ax = ax.reshape(-1)
    plt.suptitle(var1)
    iloc = 0

    for ip in range(nwindowtype):
        Twater_r_mean_plot = Twater_r_mean[ivar,:,iend,ip,iloc]
        Twater_r_std_plot = Twater_r_std[ivar,:,iend,ip,iloc]

        ax[iloc].plot(window_arr,Twater_r_mean_plot,'.-',color=plot_colors[ip])
        ax[iloc].fill_between(window_arr,Twater_r_mean_plot+Twater_r_std_plot,Twater_r_mean_plot-Twater_r_std_plot,color=plot_colors[ip],alpha=0.15)

        ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_p2,':', color='gray')
        ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_m2,':', color='gray')

        plt.subplots_adjust(left=0.2,right=0.9,bottom=0.23)
        ax[iloc].set_xlim(0,np.nanmax(window_arr)+np.nanmax(window_arr)/10.)
        ax[iloc].set_ylim(-1,1)
        ax[iloc].set_ylabel('r\n',fontsize=10)
        ax[iloc].grid()

        if iloc == nlocs-1:
            if (window_arr[1]-window_arr[0]) == 7:
                ax[iloc].set_xlabel('Previous X weeks (XW) from '+enddate_str,fontsize=10)
                labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'W' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

            if (window_arr[1]-window_arr[0]) == 30:
                ax[iloc].set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
                labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

            ax[iloc].set_xticks(window_arr)
            ax[iloc].set_xticklabels(labels_list)

#%%
# CORRELATION PLOT FOR WEATHER_VARS:

pc = p_critical

rc_m1, rc_p1 = r_confidence_interval(0,pc,nyears,tailed='one')
rc_m2, rc_p2 = r_confidence_interval(0,pc,nyears,tailed='two')
plot_colors = [plt.get_cmap('tab20b')(0),plt.get_cmap('tab20b')(4),plt.get_cmap('tab20b')(8),plt.get_cmap('tab20b')(12)]

for iend in range(0,1):
# for iend in range(1,2):
# for iend in range(2,3):
# for iend in range(3,4):
    enddate_str = enddate_labels[iend]

    for ivar in range(weather_vars_all.shape[0]):
        var1 = weather_varnames[ivar]
        nrows = weather_vars_all.shape[5]
        ncols = 1
        fig,ax = plt.subplots(nrows,figsize=(5,(nlocs)*(10/5.)),sharex=True,sharey=True,squeeze=False)
        if (nrows == 1) | (ncols == 1) :
            ax = ax.reshape(-1)
        plt.suptitle(var1)
        iloc = 0

        for ip in range(nwindowtype):
            weather_r_mean_plot = weather_r_mean[ivar,:,iend,ip,iloc]
            weather_r_std_plot = weather_r_std[ivar,:,iend,ip,iloc]

            ax[iloc].plot(window_arr,weather_r_mean_plot,'.-',color=plot_colors[ip])
            ax[iloc].fill_between(window_arr,weather_r_mean_plot+weather_r_std_plot,weather_r_mean_plot-weather_r_std_plot,color=plot_colors[ip],alpha=0.15)

            ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_p2,':', color='gray')
            ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_m2,':', color='gray')

            plt.subplots_adjust(left=0.2,right=0.9,bottom=0.23)
            ax[iloc].set_xlim(0,np.nanmax(window_arr)+np.nanmax(window_arr)/10.)
            ax[iloc].set_ylim(-1,1)
            ax[iloc].set_ylabel('r\n',fontsize=10)
            ax[iloc].grid()

            if iloc == nlocs-1:
                if (window_arr[1]-window_arr[0]) == 7:
                    ax[iloc].set_xlabel('Previous X weeks (XW) from '+enddate_str,fontsize=10)
                    labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'W' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                    labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                if (window_arr[1]-window_arr[0]) == 30:
                    ax[iloc].set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
                    labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                    labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                ax[iloc].set_xticks(window_arr)
                ax[iloc].set_xticklabels(labels_list)



#%%
def plot_corr_2vars(x_plot,y_plot,x_name,y_name,years=np.arange(1991,2022)):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(years,y_plot,'o-', color= plt.get_cmap('tab10')(0))
    ax.set_xlabel('Year')
    ax.set_ylabel(y_name, color= plt.get_cmap('tab10')(0))
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(years,x_plot,'o--', color= plt.get_cmap('tab10')(1))
    ax2.set_ylabel(x_name, color= plt.get_cmap('tab10')(1))
    lincoeff, Rsqr = linear_fit(x_plot,y_plot)
    r =np.sqrt(Rsqr)
    if lincoeff[0]<0: r*=-1
    ax.text(2000,np.nanmax(y_plot),'r = %3.2f'%(r))

#%%

ip = 0
iend = 0

# Twater in Oct.
x_detrended = Twater_vars_all_detrended[0,:,1,iend,ip,0]
x = Twater_vars_all[0,:,1,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
y = avg_freezeup_doy[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended Tw anomaly in Oct.','Freezeup Anomaly')
# plot_corr_2vars(x,y,'Tw in Oct.','Freezeup DOY')

# Twater in Nov.
x_detrended = Twater_vars_all_detrended[0,:,0,iend,ip,0]
x = Twater_vars_all[0,:,0,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
y = avg_freezeup_doy[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended Tw anomaly in Nov.','Freezeup Anomaly')
# plot_corr_2vars(x,y,'Tw in Nov.','Freezeup DOY')


# Discharge in Oct.
x_detrended = vars_all_detrended[0,:,1,iend,ip,0]
x = vars_all[0,:,1,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
y = avg_freezeup_doy[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Oct.','Freezeup Anomaly')
plot_corr_2vars(x,y,'Discharge in Oct.','Freezeup DOY')

# Discharge in Nov.
x_detrended = vars_all_detrended[0,:,0,iend,ip,0]
x = vars_all[0,:,0,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
y = avg_freezeup_doy[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Nov.','Freezeup Anomaly')
# plot_corr_2vars(x,y,'Discharge in Nov.','Freezeup DOY')


# Level in Oct.
x_detrended = vars_all_detrended[1,:,1,iend,ip,0]
x = vars_all[1,:,1,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
y = avg_freezeup_doy[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended level anomaly in Oct.','Freezeup Anomaly')
# plot_corr_2vars(x,y,'Level in Oct.','Freezeup DOY')

# Level in Nov.
x_detrended = vars_all_detrended[1,:,0,iend,ip,0]
x = vars_all[1,:,0,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
y = avg_freezeup_doy[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended level anomaly in Nov.','Freezeup Anomaly')
# plot_corr_2vars(x,y,'Level in Nov.','Freezeup DOY')


# Discharge vs Tw in Oct.
x_detrended = vars_all_detrended[0,:,1,iend,ip,0]
x = vars_all[0,:,1,iend,ip,0]
y_detrended = Twater_vars_all_detrended[0,:,1,iend,ip,0]
y = Twater_vars_all[0,:,1,iend,ip,0]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Oct.','Detrended Tw anomaly in Oct.')
# plot_corr_2vars(x,y,'Discharge in Oct.','Tw in Oct.')

# Discharge vs Tw in Nov.
x_detrended = vars_all_detrended[0,:,0,iend,ip,0]
x = vars_all[0,:,0,iend,ip,0]
y_detrended = Twater_vars_all_detrended[0,:,0,iend,ip,0]
y = Twater_vars_all[0,:,0,iend,ip,0]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Nov.','Detrended Tw anomaly in Nov.')



#%%
# Discharge vs Windspeed in Oct.
x_detrended = vars_all_detrended[0,:,1,iend,ip,0]
y_detrended = weather_vars_all_detrended[8,:,1,iend,ip,0]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Oct.','Detrended windspeed anomaly in Oct.')

x_detrended = vars_all_detrended[0,11:,1,iend,ip,0]
y_detrended = weather_vars_all_detrended[8,11:,1,iend,ip,0]
years_plot = years[11:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Oct.','Detrended windspeed anomaly in Oct.',years_plot)
#%%

# Discharge vs u-wind in Oct.
x_detrended = vars_all_detrended[0,:,1,iend,ip,0]
y_detrended = weather_vars_all_detrended[9,:,1,iend,ip,0]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Oct.','Detrended u-wind anomaly in Oct.')

# Discharge vs Windspeed in Oct.
x_detrended = vars_all_detrended[0,:,1,iend,ip,0]
y_detrended = weather_vars_all_detrended[10,:,1,iend,ip,0]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Oct.','Detrended v-wind anomaly in Oct.')


#%%
# Discharge in Oct. vs Discharge in Nov.
x_detrended = vars_all_detrended[0,:,1,iend,ip,0]
y_detrended = vars_all_detrended[0,:,0,iend,ip,0]
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Oct.','Detrended discharge anomaly in Nov.')


#%%
# Discharge in Oct. vs summer precip.
x_detrended = vars_all_detrended[0,:,1,iend,ip,0]
y_detrended = np.nansum(weather_vars_all_detrended[6,:,1:6,iend,ip,0],axis=1)
plot_corr_2vars(x_detrended,y_detrended,'Detrended discharge anomaly in Oct.','Tot. summer precip anomaly')


#%%
# Freezeup vs summer precip.
x_detrended = np.nansum(weather_vars_all_detrended[6,:,1:6,iend,ip,0],axis=1)
y_detrended = avg_freezeup_doy_detrended[:]
plot_corr_2vars(x_detrended,y_detrended,'Tot. summer precip anomaly','Freezeup Anomaly')

#%%
# SLP in Nov.
x_detrended = weather_vars_all_detrended[7,:,0,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended SLP anomaly in Nov.','Freezeup Anomaly')

# Ta_mean in Nov.
x_detrended = weather_vars_all_detrended[2,:,0,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended Ta_mean anomaly in Nov.','Freezeup Anomaly')

# Precip in Nov.
x_detrended = weather_vars_all_detrended[6,:,0,iend,ip,0]
y_detrended = avg_freezeup_doy_detrended[:]
plot_corr_2vars(x_detrended,y_detrended,'Detrended precip anomaly in Nov.','Freezeup Anomaly')


