#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:48:24 2021

@author: Amelie
"""
import numpy as np
import scipy

import pandas as pd
import statsmodels.api as sm

import datetime as dt
import calendar
import itertools
import matplotlib.pyplot as plt


import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)


from functions import running_nanmean,find_freezeup_Tw_all_yrs
from functions import linear_fit,r_confidence_interval, detrend_ts
from functions import get_window_monthly_vars, get_window_vars, deseasonalize_ts
from functions import predicted_r2
from functions_MLR import freezeup_multiple_linear_regression_model, MLR_model_analysis
# from functions_MLR import freezeup_multiple_linear_regression_model
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
if deseasonalize:
    Nwindow = 31

detrend = True
if detrend:
   anomaly = 'linear'


nboot = 1

#window_arr = 2*2**np.arange(0,8) # For powers of 2
# window_arr = np.arange(1,39)*7 # For weeks, up to Jan 1st
# window_arr = np.arange(1,3)*7
window_arr = np.arange(1,9)*30 # For months

#%%
# LOAD FREEZEUP DATES OR FIND FROM TWATER TIME SERIES

water_name_list = ['Longueuil_cleaned_filled','Candiac_cleaned_filled','Atwater_cleaned_filled']
station_labels = ['Longueuil','Candiac','Atwater']

# water_name_list = ['Longueuil_cleaned_filled']
# station_labels = ['Longueuil']

# water_name_list = ['Longueuil_cleaned_filled','Atwater_cleaned_filled']
# station_labels = ['Longueuil','Atwater']

# water_name_list = ['Longueuil_cleaned_filled','Candiac_cleaned_filled']
# station_labels = ['Longueuil','Candiac']

station_type = 'cities'

load_freezeup = False
freezeup_opt = 2
month_start_day = 1

fig, ax = plt.subplots()

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

# OPTION 3 = TEST!!!!!!!!!!!
if freezeup_opt == 3:
    def_opt = 3
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = True
    sig_dog = 3.5
    T_thresh = 1.0
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

        ax.plot(Twater_tmp)
        ax.plot(T_freezeup,'o')

# Average all stations to get mean freezeup DOY for each year
avg_freezeup_doy = np.round(np.nanmean(freezeup_doy,axis=1))
# avg_freezeup_doy = freezeup_doy[:,0]
# avg_freezeup_doy[np.isnan(avg_freezeup_doy)] = freezeup_doy[:,1][np.isnan(avg_freezeup_doy)]

# if station_labels[1] == 'Candiac':
#     avg_freezeup_doy[27] = 360

# end_dates_arr = np.zeros((len(years),1))*np.nan
# end_dates_arr[:,0] = avg_freezeup_doy

#%%
# MAKE TWATER INTO VARIABLE

Twater_varnames = ['Avg. water temp.']
Twater_vars = np.zeros((len(time),len(Twater_varnames)))*np.nan
Twater_vars[:,0] = np.nanmean(Twater,axis=1)
# Twater_vars[:,0] = Twater[:,0]
# Twater_vars[:,0][np.isnan(Twater_vars[:,0])] = Twater[:,1][np.isnan(Twater_vars[:,0])]
Twater_vars = np.squeeze(Twater_vars)

#%%
# FIND TWATER EVERY DEC 1ST

T1 = np.zeros((len(years)))*np.nan
T2 = np.zeros((len(years)))*np.nan
T3 = np.zeros((len(years)))*np.nan
Tm = np.zeros((len(years)))*np.nan

for iyr, year in enumerate(years):

    dec1 = (dt.date(int(year),12,1)-date_ref).days
    # dec1 = (dt.date(int(year),12,5)-date_ref).days
    # dec1 = (dt.date(int(year),12,10)-date_ref).days
    # dec1 = (dt.date(int(year),12,15)-date_ref).days
    it = np.where(time == dec1)[0][0]

    T1[iyr] = Twater[it,0]
    T2[iyr] = Twater[it,1]
    T3[iyr] = Twater[it,2]
    Tm[iyr] = np.nanmean(Twater[it])


fig1,ax1 = plt.subplots()
plt.title('Twater Dec 1st')
# plt.title('Twater Dec 5th')
# plt.title('Twater Dec 10th')
# plt.title('Twater Dec 15th')
ax1.plot(T1,avg_freezeup_doy,'o',color= plt.get_cmap('tab10')(0))
ax1.plot(T2,avg_freezeup_doy,'o',color= plt.get_cmap('tab10')(1))
ax1.plot(T3,avg_freezeup_doy,'o',color= plt.get_cmap('tab10')(2))
ax1.plot(Tm,avg_freezeup_doy,'o',color= plt.get_cmap('tab10')(3))

_, Rsqr1 = linear_fit(T1,avg_freezeup_doy)
_, Rsqr2 = linear_fit(T2,avg_freezeup_doy)
_, Rsqr3 = linear_fit(T3,avg_freezeup_doy)
_, Rsqrm = linear_fit(Tm,avg_freezeup_doy)

ax1.text(3,380,'Rsqr = {:03.2f}'.format(Rsqr1),color= plt.get_cmap('tab10')(0))
ax1.text(3,377,'Rsqr = {:03.2f}'.format(Rsqr2),color= plt.get_cmap('tab10')(1))
ax1.text(3,374,'Rsqr = {:03.2f}'.format(Rsqr3),color= plt.get_cmap('tab10')(2))
ax1.text(3,371,'Rsqr = {:03.2f}'.format(Rsqrm),color= plt.get_cmap('tab10')(3))


# fig2,ax2 = plt.subplots()
# plt.title('Twater Dec 1st')
# # plt.title('Twater Dec 5th')
# plt.title('Twater Dec 10th')
# # plt.title('Twater Dec 15th')
# ax2.plot(np.round(T1.copy()* 2)/2.,avg_freezeup_doy,'o',color= plt.get_cmap('tab10')(0))
# ax2.plot(np.round(T2.copy()* 2)/2.,avg_freezeup_doy,'o',color= plt.get_cmap('tab10')(1))
# ax2.plot(np.round(T3.copy()* 2)/2.,avg_freezeup_doy,'o',color= plt.get_cmap('tab10')(2))
# ax2.plot(np.round(Tm.copy()* 2)/2.,avg_freezeup_doy,'o',color= plt.get_cmap('tab10')(3))


# _, Rsqr1 = linear_fit(np.round(T1.copy()* 2)/2.,avg_freezeup_doy)
# _, Rsqr2 = linear_fit(np.round(T2.copy()* 2)/2.,avg_freezeup_doy)
# _, Rsqr3 = linear_fit(np.round(T3.copy()* 2)/2.,avg_freezeup_doy)
# _, Rsqrm = linear_fit(np.round(Tm.copy()* 2)/2.,avg_freezeup_doy)

# ax2.text(3,380,'Rsqr = {:03.2f}'.format(Rsqr1),color= plt.get_cmap('tab10')(0))
# ax2.text(3,377,'Rsqr = {:03.2f}'.format(Rsqr2),color= plt.get_cmap('tab10')(1))
# ax2.text(3,374,'Rsqr = {:03.2f}'.format(Rsqr3),color= plt.get_cmap('tab10')(2))
# ax2.text(3,371,'Rsqr = {:03.2f}'.format(Rsqrm),color= plt.get_cmap('tab10')(3))
