#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:08:00 2021

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

fp = local_path+'slice/data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)

#%%
# TRANSFORM FREEZEUP FROM CHARTS IN DOY FORMAT
freezeup_loc_list = ['Lasalle','Candiac','LaPrairie','Longueuil','StLambert','MontrealOldPort','MontrealPort']
freezeup_type = 'charts'

chart_fd = np.zeros((len(freezeup_loc_list),len(years),3))*np.nan
chart_fd_doy = np.zeros((len(freezeup_loc_list),len(years),2))*np.nan

for iloc,loc in enumerate(freezeup_loc_list):

    ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type+'/freezeup_'+freezeup_type+'_'+loc+'.npz',allow_pickle='TRUE')
    freezeup = ice_data['freezeup_ci'][:,0]
    freezeup_dt = ice_data['freezeup_ci'][:,1]


    for iyr,year in enumerate(years):

        date=(dt.date(year,11,1)-date_ref).days
        i0 = np.where(time==date)[0][0] - 30
        i1 = i0+120

        ci_select = freezeup[i0:i1].copy()
        dt_select = freezeup_dt[i0:i1].copy()

        if np.sum(~np.isnan(ci_select)) > 0:
            i_fd = np.where(~np.isnan(ci_select))[0][0]
            date_chart = date_ref+dt.timedelta(days=int(ci_select[i_fd]))
            fd_dt = dt_select[i_fd]

            fd_yy = int(date_chart.year)
            fd_mm = int(date_chart.month)
            fd_dd = int(date_chart.day)

            chart_fd[iloc,iyr,0] = date_chart.year
            chart_fd[iloc,iyr,1] = date_chart.month
            chart_fd[iloc,iyr,2] = date_chart.day

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365
            chart_fd_doy[iloc,iyr,0] = fd_doy
            chart_fd_doy[iloc,iyr,1] = fd_dt


#%%
# FIND FREEZEUP FROM TWATER AND CHARTS AT THE SAME LOCATIONS

# water_name_list = ['Atwater_cleaned_filled','DesBaillets_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
# station_labels = ['Atwater','DesBaillets','Longueuil','Candiac']
# station_type = 'cities'

# freezeup_loc_list = ['Lasalle','Lasalle','Longueuil','Candiac']
# freezeup_type = 'charts'


water_name_list = ['Longueuil_cleaned_filled','Longueuil_cleaned_filled']
station_labels = ['Longueuil','Longueuil']
station_type = 'cities'

freezeup_loc_list = ['Longueuil','Longueuil']
freezeup_type = 'charts'

chart_fd = np.zeros((len(freezeup_loc_list),len(years),3))*np.nan
chart_fd_doy = np.zeros((len(freezeup_loc_list),len(years),2))*np.nan

for iloc,loc in enumerate(freezeup_loc_list):

    ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type+'/freezeup_'+freezeup_type+'_'+loc+'.npz',allow_pickle='TRUE')
    freezeup = ice_data['freezeup_ci'][:,0]
    freezeup_dt = ice_data['freezeup_ci'][:,1]

    for iyr,year in enumerate(years):

        date=(dt.date(year,11,1)-date_ref).days
        i0 = np.where(time==date)[0][0] - 30
        i1 = i0+120

        ci_select = freezeup[i0:i1].copy()
        dt_select = freezeup_dt[i0:i1].copy()

        if np.sum(~np.isnan(ci_select)) > 0:
            i_fd = np.where(~np.isnan(ci_select))[0][0]
            date_chart = date_ref+dt.timedelta(days=int(ci_select[i_fd]))
            fd_dt = dt_select[i_fd]

            fd_yy = int(date_chart.year)
            fd_mm = int(date_chart.month)
            fd_dd = int(date_chart.day)

            chart_fd[iloc,iyr,0] = date_chart.year
            chart_fd[iloc,iyr,1] = date_chart.month
            chart_fd[iloc,iyr,2] = date_chart.day

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365
            chart_fd_doy[iloc,iyr,0] = fd_doy
            chart_fd_doy[iloc,iyr,1] = fd_dt




#%%
# FIND FREEZEUP DATES FROM TWATER TIME SERIES, FROM THRESHOLD ON dTwdt
freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
breakup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan

freezeup_temp = np.zeros((len(years),len(water_name_list)))*np.nan

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
# no_negTw = False

# OPTION 1B: THRESHOLD ON Tw, BUT BETTER THAN ORIGINAL DEFINITION
def_opt = 1
smooth_T =False; N_smooth = 3; mean_type='centered'
round_T = False; round_type= 'half_unit'
Gauss_filter = False
T_thresh = 0.75
dTdt_thresh = 0.25
d2Tdt2_thresh = 0.25
nd = 1
no_negTw = False

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

# #TEST:
# def_opt = 1
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = False
# T_thresh = 0.0
# dTdt_thresh = 0.25
# d2Tdt2_thresh = 0.25
# nd = 1
# no_negTw = True



# BELOW ARE ALL OPTIONS THAT WERE TESTED:
#-------------------------------------------------------
#OPTION 2 - A
# def_opt = 2
# smooth_T =True; N_smooth = 7; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = False
# T_thresh = 4.0
# dTdt_thresh = 0.2
# d2Tdt2_thresh = 0.15
# nd = 30

#OPTION 2 - B
# def_opt = 2
# smooth_T =True; N_smooth = 30; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = False
# T_thresh = 3.0
# dTdt_thresh = 0.15
# d2Tdt2_thresh = 0.15
# nd = 1

# #OPTION 2  - C SAME AS OPT.3A, BUT WITH DTDt INSTEAD OF DoG
# def_opt = 2
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = False
# T_thresh = 1.0
# dTdt_thresh = 0.25
# d2Tdt2_thresh = 0.25
# nd = 1

#-------------------------------------------------------
# OPTION 3 - A
# def_opt = 3
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = True
# sig_dog = 1
# T_thresh = 1.0
# dTdt_thresh = 0.25
# d2Tdt2_thresh = 0.25
# nd = 1

# # # OPTION 3 - B
# def_opt = 3
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = True
# sig_dog = 3.
# T_thresh = 3.
# dTdt_thresh = 0.2
# d2Tdt2_thresh = 0.2
# nd = 30

# # OPTION 3 - C
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


def_opt = 3
smooth_T =False; N_smooth = 3; mean_type='centered'
round_T = False; round_type= 'half_unit'
Gauss_filter = True
sig_dog = 30
T_thresh = 3.
dTdt_thresh = 0.2
d2Tdt2_thresh = 0.2
# dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
# d2Tdt2_thresh = 0.20
nd = 30
no_negTw = True

#-------------------------------------------------------


# APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST,
# BEFORE FINDING FREEZEUP DATES FROM WATER TEMP.
Twater = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_DoG1 = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_DoG2 = np.zeros((len(time),len(water_name_list)))*np.nan

fig_tw,ax_tw = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))
fig_ftw,ax_ftw = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))

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
    freezeup_dates[:,:,iloc] = fd
    freezeup_temp[:,iloc] = ftw

    ax_tw.plot(time,Twater_tmp,color=plt.get_cmap('tab20')(iloc*2+1))
    ax_tw.plot(time,T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2))
    # ax_tw.plot(T_breakup, 'o',color=plt.get_cmap('tab20')(iloc*2))

    ax_ftw.plot(years,freezeup_temp[:,iloc],'>',color=plt.get_cmap('tab20')(iloc*2+1), label=station_labels[iloc])

    for iyr in range(chart_fd.shape[1]):
        if ~np.isnan(chart_fd[iloc,iyr,0]):
            d = (dt.date(int(chart_fd[iloc,iyr,0]),int(chart_fd[iloc,iyr,1]),int(chart_fd[iloc,iyr,2]))-date_ref).days
            fu_i = np.where(time == d)[0][0]
            ax_tw.plot(time[int(fu_i)], Twater_tmp[fu_i], '*', color='black')

ax_ftw.plot(np.arange(years[0],years[-1]+1),np.zeros(len(years)),'-',color = [0.9,0.9,0.9],linewidth=0.75)
ax_ftw.set_xlim([years[0],years[-1]+1])
ax_ftw.plot(np.arange(years[0],years[-1]+1),np.ones(len(years))*np.nanmean(freezeup_temp),':',color=[0.5,0.5,0.5])
ax_ftw.fill_between(np.arange(years[0],years[-1]+1), np.ones(len(years))*np.nanmean(freezeup_temp)+np.nanstd(freezeup_temp),np.ones(len(years))*np.nanmean(freezeup_temp)-np.nanstd(freezeup_temp),facecolor=[0.9,0.9,0.9], interpolate=True, alpha=0.65)
ax_ftw.set_ylim([-0.5,5])
ax_ftw.set_ylabel('Water temp. at detected freezeup date ($^{\circ}$C)')
ax_ftw.set_xlabel('Year')
ax_ftw.set_xticks(np.arange(years[0],years[-1]))
# ax_ftw.set_xticklabels([str(years[i])[-2:] for i in range(len(years))],fontsize=10)
ax_ftw.legend()



freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
breakup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
for iloc,loc in enumerate(water_name_list):
    for iyr,year in enumerate(years):
        if ~np.isnan(freezeup_dates[iyr,0,iloc]):
            fd_yy = int(freezeup_dates[iyr,0,iloc])
            fd_mm = int(freezeup_dates[iyr,1,iloc])
            fd_dd = int(freezeup_dates[iyr,2,iloc])

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365

            freezeup_doy[iyr,iloc]=fd_doy

        if ~np.isnan(breakup_dates[iyr,0,iloc]):
            bd_yy = int(breakup_dates[iyr,0,iloc])
            bd_mm = int(breakup_dates[iyr,1,iloc])
            bd_dd = int(breakup_dates[iyr,2,iloc])

            bd_doy = (dt.date(bd_yy,bd_mm,bd_dd)-dt.date(bd_yy,1,1)).days + 1

            breakup_doy[iyr,iloc]=bd_doy


# PLOT FREEZEUP DOY TIME SERIES
fig_fddoy,ax_fddoy = plt.subplots(nrows=1,ncols=2,figsize=(12,3.5))
for iloc,loc in enumerate(water_name_list):
    ax_fddoy[0].plot(years,freezeup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2+1), label=station_labels[iloc],alpha=0.5)
    # y_plot = freezeup_doy[:,iloc].copy()
    # lincoeff, Rsqr = linear_fit(years[~np.isnan(y_plot)],y_plot[~np.isnan(y_plot)])
    # ax_fddoy.plot(years,lincoeff[0]*years+lincoeff[1],'-',color=plt.get_cmap('tab20')(iloc*2+1) )
    # ax_fddoy.text(2010,413-6*iloc,'%3.2f'%lincoeff[0]+'x '+'%3.2f'%lincoeff[1] +' (R$^{2}$: %3.2f'%(Rsqr)+')',color=plt.get_cmap('tab20')(iloc*2+1) )

    fd_chart_plot = chart_fd_doy[iloc,:,0][~np.isnan(freezeup_doy[:,iloc])]
    dt_chart_plot = chart_fd_doy[iloc,:,1][~np.isnan(freezeup_doy[:,iloc])]
    years_plot = years[~np.isnan(freezeup_doy[:,iloc])]

    lower_yerror = []
    upper_yerror = []
    for iyr,year in enumerate(years_plot):
        if ~np.isnan(dt_chart_plot[iyr]):
            lower_yerror.append(dt_chart_plot[iyr]-1)
            upper_yerror.append(0)
        else:
            lower_yerror.append(0)
            upper_yerror.append(0)

    yerror = [lower_yerror, upper_yerror]
    ax_fddoy[0].errorbar(years_plot,fd_chart_plot, yerr=yerror, fmt='*',color=plt.get_cmap('tab20')(iloc*2),alpha=0.5)

ax_fddoy[0].legend(bbox_to_anchor=(0.0, 0.64, 0.3, 0.3),fontsize=8)
ax_fddoy[0].set_ylim(300,430)
ax_fddoy[0].set_xlabel('Years')
ax_fddoy[0].set_ylabel('Freezeup DOY')


# PLOT FREEZEUP TEMP VS FREEZEUP DOY
fig_ftwdoy,ax_ftwdoy = plt.subplots(nrows=1,ncols=1,figsize=(6,3.5))
for iloc,loc in enumerate(water_name_list):
    ax_ftwdoy.plot(freezeup_doy[:,iloc],freezeup_temp[:,iloc], 'o',color=plt.get_cmap('tab20')(iloc*2+1),alpha=0.5, label=station_labels[iloc])

ax_ftwdoy.legend(bbox_to_anchor=(0.0, 0.64, 0.3, 0.3),fontsize=8)
ax_ftwdoy.set_xlim(300,430)
ax_ftwdoy.set_xlabel('Freezeup DOY')
ax_ftwdoy.set_ylabel('Freezeup water temp. (detected)')


#ADD FREEZEUP OBSERVED FROM SLSMC
freezeup_loc_slsmc = ['SouthShoreCanal']
freezeup_type2 = 'SLSMC'
slsmc_fi = np.zeros((len(freezeup_loc_slsmc),len(years),3))*np.nan
slsmc_fi_doy = np.zeros((len(freezeup_loc_slsmc),len(years),2))*np.nan
slsmc_si = np.zeros((len(freezeup_loc_slsmc),len(years),3))*np.nan
slsmc_si_doy = np.zeros((len(freezeup_loc_slsmc),len(years),2))*np.nan

for iloc,loc in enumerate(freezeup_loc_slsmc):

    ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type2+'/freezeup_'+freezeup_type2+'_'+loc+'.npz',allow_pickle='TRUE')
    freezeup_fi = ice_data['freezeup_fi'][:,0]
    freezeup_si = ice_data['freezeup_si'][:,0]

    for iyr,year in enumerate(years):

        date=(dt.date(year,11,1)-date_ref).days
        i0 = np.where(time==date)[0][0] - 30
        i1 = i0+120

        fi_select = freezeup_fi[i0:i1].copy()
        si_select = freezeup_si[i0:i1].copy()

        if np.sum(~np.isnan(fi_select)) > 0:
            i_fd = np.where(~np.isnan(fi_select))[0][0]
            date_chart = date_ref+dt.timedelta(days=int(fi_select[i_fd]))

            fd_yy = int(date_chart.year)
            fd_mm = int(date_chart.month)
            fd_dd = int(date_chart.day)

            slsmc_fi[iloc,iyr,0] = date_chart.year
            slsmc_fi[iloc,iyr,1] = date_chart.month
            slsmc_fi[iloc,iyr,2] = date_chart.day

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365
            slsmc_fi_doy[iloc,iyr,0] = fd_doy

        if np.sum(~np.isnan(si_select)) > 0:
            i_fd = np.where(~np.isnan(si_select))[0][0]
            date_chart = date_ref+dt.timedelta(days=int(si_select[i_fd]))

            fd_yy = int(date_chart.year)
            fd_mm = int(date_chart.month)
            fd_dd = int(date_chart.day)

            slsmc_si[iloc,iyr,0] = date_chart.year
            slsmc_si[iloc,iyr,1] = date_chart.month
            slsmc_si[iloc,iyr,2] = date_chart.day

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365
            slsmc_si_doy[iloc,iyr,0] = fd_doy



for iloc,loc in enumerate(freezeup_loc_slsmc):
    fi_plot = slsmc_fi_doy[iloc,:,0]
    si_plot = slsmc_si_doy[iloc,:,0]
    years_plot = years

    ax_fddoy[0].plot(years_plot,fi_plot, '*',color='gray',alpha=0.5)
    ax_fddoy[0].plot(years_plot,si_plot, '*',color='black',alpha=0.5)


ax2=ax_fddoy[1].twinx()
for iloc,loc in enumerate(water_name_list):
    mask_water = ~np.isnan(freezeup_doy[17:28,iloc])
    mask_chart = ~np.isnan(chart_fd_doy[iloc,17:28,0])
    mask = mask_water & mask_chart
    # ax_fddoy[1].boxplot(freezeup_doy[14:29,iloc][mask_water], positions = [iloc],whis=[5, 95],showfliers=True,labels = [station_labels[iloc]])
    # ax_fddoy[1].boxplot(chart_fd_doy[iloc,14:29,0][mask_chart], positions = [iloc+0.3],whis=[5, 95],showfliers=True,labels = [freezeup_loc_list[iloc]])
    # # print(np.nanmedian(freezeup_doy[:,iloc][mask]))
    bp_doy = ax_fddoy[1].boxplot(freezeup_doy[17:28,iloc][mask], positions = [iloc],whis=[5, 95],showfliers=True,labels = [station_labels[iloc]])
    bp_diff = ax2.boxplot(freezeup_doy[17:28,iloc][mask]-chart_fd_doy[iloc,17:28,0][mask], positions = [iloc+0.3],whis=[5, 95],showfliers=True,labels = [station_labels[iloc]])

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp_diff[element], color=plt.get_cmap('tab10')(0))


# bp = ax.boxplot(data, patch_artist=True)

# for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
#     plt.setp(bp[element], color=edge_color)

ax_fddoy[1].grid(True)
ax2.set_ylabel('Difference with charts (days)', color=plt.get_cmap('tab10')(0))
ax_fddoy[1].set_ylabel('Freezeup DOY')
ax_fddoy[1].set_ylim(336,395)
ax2.set_ylim(-18,18)

plt.subplots_adjust(bottom=0.3)
# ax_fddoy[1].set_xticklabels(labels = [station_labels[0],freezeup_loc_list[0],station_labels[1],freezeup_loc_list[1],station_labels[2],freezeup_loc_list[2],station_labels[3],freezeup_loc_list[3]],rotation=45)
# ax_fddoy[1].set_xticklabels(labels = [station_labels[0],'',station_labels[1],'',station_labels[2],'',station_labels[3],''],rotation=45)

posorig = ax_fddoy[0].get_position()
posnew = [posorig.x0-0.05, posorig.y0+0.1, posorig.width*1.5, posorig.height*0.9]
ax_fddoy[0].set_position(posnew)

posorig = ax_fddoy[1].get_position()
posnew = [posorig.x0+0.12, posorig.y0+0.1, posorig.width*0.75, posorig.height*0.9]
ax_fddoy[1].set_position(posnew)

#%%
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,5), sharex=True)
for iloc,loc in enumerate(water_name_list):
    ax.boxplot(freezeup_doy[:,iloc][~np.isnan(freezeup_doy[:,iloc])], positions = [iloc],whis=[5, 95],showmeans=True,showfliers=True,labels = [station_labels[iloc]])
    print(np.nanmedian(freezeup_doy[:,iloc]), np.nanmean(freezeup_doy[:,iloc]))
plt.suptitle('ALL')
ax.set_ylim((335,390))

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,5), sharex=True)
for iloc,loc in enumerate(water_name_list):
    ax.boxplot(freezeup_doy[14:29,iloc][~np.isnan(freezeup_doy[14:29,iloc])], positions = [iloc],whis=[5, 95],showmeans=True,showfliers=True,labels = [station_labels[iloc]])
    print(np.nanmedian(freezeup_doy[14:29,iloc][~np.isnan(freezeup_doy[14:29,iloc])]), np.nanmean(freezeup_doy[14:29,iloc]))
plt.suptitle('SUB-PERIOD')
ax.set_ylim((335,390))



fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,5), sharex=True)
for iloc,loc in enumerate(water_name_list):
    ax.boxplot(freezeup_doy[17:28,iloc][~np.isnan(freezeup_doy[17:28,iloc])], positions = [iloc],whis=[5, 95],showmeans=True,showfliers=True,labels = [station_labels[iloc]])
    print(np.nanmedian(freezeup_doy[17:28,iloc][~np.isnan(freezeup_doy[17:28,iloc])]), np.nanmean(freezeup_doy[17:28,iloc]))
plt.suptitle('SUB-PERIOD')
ax.set_ylim((335,390))


# AND PLOT DIFFERENCES BETWEEN STATIONS
show_plot = True

if show_plot:
    for iloc,loc in enumerate(water_name_list):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5), sharex=True)

        plt.suptitle(loc)

        fd_main = freezeup_doy[:,iloc]

        for jloc in range(len(water_name_list)):
            if jloc != iloc:
                fd_diff = fd_main - freezeup_doy[:,jloc]

                ax.boxplot(fd_diff[~np.isnan(fd_diff)], positions = [jloc*0.25],whis=[5, 95],showfliers=True,labels = [station_labels[jloc]+'\n(%2i'%np.sum(~np.isnan(fd_diff))+')'])

                plt.xticks(rotation=45)
                plt.subplots_adjust(bottom=0.24)
                ax.set_ylabel('Diff. (days) in freezeup\n ('+loc+' - other stations)')
                ax.grid(True)
                ax.text(-0.25, -15, 'Earlier',color='blue',rotation=90)
                ax.text(-0.25, 4, 'Later',color='red',rotation=90)
                ax.set_ylim(-44,44)
                ax.set_xlim(-0.35,0.25*7)




#%%
# ADD FREEZE-UP DATES DETECTED FROM MEAN TWATER OF LONGUEUIL, CANDIAC, AND ATWATER

# Twater_mean = np.nanmean(Twater[:,[0,2,3]].copy(),axis=1)
# freezeup_dates_mean = np.zeros((len(years),3,1))*np.nan

# # APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST,
# # BEFORE FINDING FREEZEUP DATES FROM WATER TEMP.
# Twater_dTdt_mean = np.zeros((len(time),1))*np.nan
# Twater_d2Tdt2_mean = np.zeros((len(time),1))*np.nan
# Twater_DoG1_mean = np.zeros((len(time),1))*np.nan
# Twater_DoG2_mean = np.zeros((len(time),1))*np.nan

# for iloc,loc in enumerate([0]):

#     Twater_tmp = Twater_mean.copy()

#     if round_T:
#         if round_type == 'unit':
#             Twater_tmp = np.round(Twater_tmp.copy())
#         if round_type == 'half_unit':
#             Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.

#     if smooth_T:
#         Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

#     if no_negTw:
#         Twater_tmp[Twater_tmp < 0] = 0.0

#     dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

#     dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
#     dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
#     dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

#     Twater_dTdt_mean[:,0] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
#     Twater_d2Tdt2_mean[:,0] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)


#     if Gauss_filter:
#         Twater_DoG1_mean[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
#         Twater_DoG2_mean[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

#     if def_opt == 3:
#         fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_DoG1_mean[:,iloc],Twater_DoG2_mean[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
#         freezeup_dates_mean[:,:,iloc] = fd
#     else:
#         fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt_mean[:,iloc],Twater_d2Tdt2_mean[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
#         freezeup_dates_mean[:,:,iloc] = fd


#     ax_tw.plot(time,Twater_tmp,color=plt.get_cmap('tab20b')(iloc*2+1))
#     ax_tw.plot(time,T_freezeup, '*',color=plt.get_cmap('tab20b')(iloc*2))
#     # ax_tw.plot(T_breakup, 'o',color=plt.get_cmap('tab20')(iloc*2))

#     for iyr in range(chart_fd.shape[1]):
#         if ~np.isnan(chart_fd[iloc,iyr,0]):
#             d = (dt.date(int(chart_fd[iloc,iyr,0]),int(chart_fd[iloc,iyr,1]),int(chart_fd[iloc,iyr,2]))-date_ref).days
#             fu_i = np.where(time == d)[0][0]
#             ax_tw.plot(time[int(fu_i)], Twater_tmp[fu_i], '*', color='black')



# freezeup_doy_mean = np.zeros((len(years),1))*np.nan
# breakup_doy_mean = np.zeros((len(years),1))*np.nan
# for iloc,loc in enumerate([0]):
#     for iyr,year in enumerate(years):
#         if ~np.isnan(freezeup_dates_mean[iyr,0,iloc]):
#             fd_yy = int(freezeup_dates_mean[iyr,0,iloc])
#             fd_mm = int(freezeup_dates_mean[iyr,1,iloc])
#             fd_dd = int(freezeup_dates_mean[iyr,2,iloc])

#             fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
#             if fd_doy < 60: fd_doy += 365

#             freezeup_doy_mean[iyr,iloc]=fd_doy


# # PLOT FREEZEUP DOY TIME SERIES
# for iloc,loc in enumerate([0]):
#     ax_fddoy[0].plot(years,freezeup_doy_mean[:,iloc],'o',color=plt.get_cmap('tab20b')(iloc*2+1), label=station_labels[iloc],alpha=0.5)



# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,5), sharex=True)


# chart_loc = 0
# mask_water = ~np.isnan(freezeup_doy_mean[17:28,0])
# mask_chart = ~np.isnan(chart_fd_doy[chart_loc,17:28,0])
# mask = mask_water & mask_chart

# ax.boxplot(freezeup_doy_mean[17:28,0][mask]-chart_fd_doy[chart_loc,17:28,0][mask], positions = [iloc],whis='range',showfliers=True,labels = [station_labels[iloc]])

# ax.grid(True)
# ax.set_ylabel('Freezeup DOY difference with charts')
# ax.set_ylim(-18,18)

# plt.subplots_adjust(bottom=0.24)
# # ax_fddoy[1].set_xticklabels(labels = [station_labels[0],freezeup_loc_list[0],station_labels[1],freezeup_loc_list[1],station_labels[2],freezeup_loc_list[2],station_labels[3],freezeup_loc_list[3]],rotation=45)
# # ax.set_xticklabels(labels = [station_labels[0],station_labels[1],station_labels[2],station_labels[3]],rotation=45)





#%%

# ADD FREEZEUP FROM TW FROM ECCC
water_name_list = ['LaPrairie_cleaned_filled','Lasalle_cleaned_filled']
station_type = 'eccc'

freezeup_loc_list = ['LaPrairie','Lasalle']
freezeup_type = 'charts'

# FIRST FIND FREEZEUP FROM CHARTS AT THE DESIRED LOCATIONS TO MATCH WATER DATA
chart_fd = np.zeros((len(freezeup_loc_list),len(years),3))*np.nan
chart_fd_doy = np.zeros((len(freezeup_loc_list),len(years),2))*np.nan

for iloc,loc in enumerate(freezeup_loc_list):

    ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type+'/freezeup_'+freezeup_type+'_'+loc+'.npz',allow_pickle='TRUE')
    freezeup = ice_data['freezeup_ci'][:,0]
    freezeup_dt = ice_data['freezeup_ci'][:,1]

    for iyr,year in enumerate(years):

        date=(dt.date(year,11,1)-date_ref).days
        i0 = np.where(time==date)[0][0] - 30
        i1 = i0+120

        ci_select = freezeup[i0:i1].copy()
        dt_select = freezeup_dt[i0:i1].copy()

        if np.sum(~np.isnan(ci_select)) > 0:
            i_fd = np.where(~np.isnan(ci_select))[0][0]
            date_chart = date_ref+dt.timedelta(days=int(ci_select[i_fd]))
            fd_dt = dt_select[i_fd]

            fd_yy = int(date_chart.year)
            fd_mm = int(date_chart.month)
            fd_dd = int(date_chart.day)

            chart_fd[iloc,iyr,0] = date_chart.year
            chart_fd[iloc,iyr,1] = date_chart.month
            chart_fd[iloc,iyr,2] = date_chart.day

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365
            chart_fd_doy[iloc,iyr,0] = fd_doy
            chart_fd_doy[iloc,iyr,1] = fd_dt


# THEN APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST, BEFORE
# FINDING FREEZEUP DATES FROM WATER TEMP.
Twater = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan

for iloc,loc in enumerate(water_name_list):
    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][:,1]

    Twater[:,iloc] = Twater_tmp
    if loc == 'Candiac_cleaned_filled':
        Twater[:,iloc] = Twater_tmp-0.8
    if (loc == 'Atwater_cleaned_filled'):
        Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7




    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)


# FIND FREEZEUP DATES FROM TWATER TIME SERIES
freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
breakup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan

station_labels = ['La Prairie ECCC','Lasalle ECCC']


for iloc,loc in enumerate(water_name_list):
    Twater_tmp = Twater[:,iloc].copy()

    if round_T:
        if round_type == 'unit':
            Twater_tmp = np.round(Twater_tmp.copy())
        if round_type == 'half_unit':
            Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.

    if smooth_T:
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

    # if Gauss_filter:
    #     Twater_tmp = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=1,order=2)

    # fd, T_freezeup, mask_freeze = find_freezeup_Tw(Twater_tmp,time,years,thresh_T = T_thresh, ndays = nd)
    # freezeup_dates[:,:,iloc] = fd

    if no_negTw:
        Twater_tmp[Twater_tmp < 0] = 0.0

    if Gauss_filter:
        Twater_DoG1[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
        Twater_DoG2[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

    if def_opt == 3:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_DoG1[:,iloc],Twater_DoG2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_dates[:,:,iloc] = fd
        freezeup_temp[:,iloc] = ftw
    else:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_dates[:,:,iloc] = fd
        freezeup_temp[:,iloc] = ftw


    ax_tw.plot(time,Twater_tmp,color=plt.get_cmap('tab20')(iloc*2+1+8))
    ax_tw.plot(time,T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2+8))

    for iyr in range(chart_fd.shape[1]):
        if ~np.isnan(chart_fd[iloc,iyr,0]):
            d = (dt.date(int(chart_fd[iloc,iyr,0]),int(chart_fd[iloc,iyr,1]),int(chart_fd[iloc,iyr,2]))-date_ref).days
            fu_i = np.where(time == d)[0][0]
            ax_tw.plot(fu_i, Twater_tmp[fu_i], '*', color='black')


    # mean_winter_temp = np.zeros(len(years))*np.nan
    # for iyr,year in enumerate(years[:-1]):
    #     if ~np.isnan(fd[iyr,0]):

    #         i_fr = np.where(time == ( dt.date(int(fd[iyr,0]),int(fd[iyr,1]),int(fd[iyr,2]))-date_ref).days)[0][0]
    #         i_br = np.where(time == ( dt.date(int(bd[iyr+1,0]),int(bd[iyr+1,1]),int(bd[iyr+1,2]))-date_ref).days)[0][0]
    #         Twater_winter = Twater[i_fr:i_br,iloc]
    #         mean_winter_temp[iyr] = np.nanmean(Twater_winter)


freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
breakup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
for iloc,loc in enumerate(water_name_list):
    for iyr,year in enumerate(years):
        if ~np.isnan(freezeup_dates[iyr,0,iloc]):
            fd_yy = int(freezeup_dates[iyr,0,iloc])
            fd_mm = int(freezeup_dates[iyr,1,iloc])
            fd_dd = int(freezeup_dates[iyr,2,iloc])

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365

            freezeup_doy[iyr,iloc]=fd_doy

        if ~np.isnan(breakup_dates[iyr,0,iloc]):
            bd_yy = int(breakup_dates[iyr,0,iloc])
            bd_mm = int(breakup_dates[iyr,1,iloc])
            bd_dd = int(breakup_dates[iyr,2,iloc])

            bd_doy = (dt.date(bd_yy,bd_mm,bd_dd)-dt.date(bd_yy,1,1)).days + 1

            breakup_doy[iyr,iloc]=bd_doy



for iloc,loc in enumerate(water_name_list):
    ax_fddoy[0].plot(years,freezeup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2+1+8), label=station_labels[iloc],alpha=0.5)
    fd_chart_plot = chart_fd_doy[iloc,:,0][~np.isnan(freezeup_doy[:,iloc])]
    dt_chart_plot = chart_fd_doy[iloc,:,1][~np.isnan(freezeup_doy[:,iloc])]
    years_plot = years[~np.isnan(freezeup_doy[:,iloc])]

    lower_yerror = []
    upper_yerror = []
    for iyr,year in enumerate(years_plot):
        if ~np.isnan(dt_chart_plot[iyr]):
            lower_yerror.append(dt_chart_plot[iyr]-1)
            upper_yerror.append(0)
        else:
            lower_yerror.append(0)
            upper_yerror.append(0)

    yerror = [lower_yerror, upper_yerror]
    ax_fddoy[0].errorbar(years_plot,fd_chart_plot, yerr=yerror, fmt='*',color=plt.get_cmap('tab20')(iloc*2+8),alpha=0.5)

ax_fddoy[0].legend(bbox_to_anchor=(0.0, 0.64, 0.3, 0.3),fontsize=8)
ax_fddoy[0].set_ylim(300,430)
ax_fddoy[0].set_xlabel('Years')
ax_fddoy[0].set_ylabel('Freezeup DOY')



# for iloc,loc in enumerate(water_name_list):
#     mask_water = ~np.isnan(freezeup_doy[:,iloc])
#     mask_chart = ~np.isnan(chart_fd_doy[iloc,:,0])
#     mask = mask_water & mask_chart
#     ax_fddoy[1].boxplot(freezeup_doy[:,iloc][mask], positions = [iloc],whis=[5, 95],showfliers=True,labels = [station_labels[iloc]])
#     ax_fddoy[1].boxplot(chart_fd_doy[iloc,:,0][mask], positions = [iloc+0.3],whis=[5, 95],showfliers=True,labels = [freezeup_loc_list[iloc]])








