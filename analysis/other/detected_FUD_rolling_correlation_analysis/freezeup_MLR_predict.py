#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:09:08 2021

@author: Amelie
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 06:54:46 2021

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

from statsmodels.formula.api import ols
import pandas as pd
import statsmodels.api as sm

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import running_nanmean,find_freezeup_Tw_all_yrs,season_mask
from functions import linear_fit, rolling_climo, r_confidence_interval, detrend_ts
from functions import get_window_monthly_vars, get_window_vars, deseasonalize_ts
from functions_MLR import freezeup_multiple_linear_regression_model


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

if deseasonalize:
    Twater_vars = deseasonalize_ts(Nwindow,Twater_vars,['Twater'],'all_time',time,years)

Twater_vars_all = np.zeros((1,len(years),len(window_arr),end_dates_arr.shape[1],2,1))*np.nan
for iend in range(end_dates_arr.shape[1]):
    Twater_vars_all[:,:,:,iend,:,0] = get_window_vars(np.expand_dims(Twater_vars,axis=1),['Avg. Twater'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

Twater_vars_all = np.squeeze(Twater_vars_all[:,:,:,:,0,:])

#%%
# LOAD WEATHER DATA
fp2 = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/detected_freezeup_correlation_analysis/'

weather_data1 = np.load(fp2+'weather_vars_all_monthly.npz',allow_pickle='TRUE')
weather_data2 = np.load(fp2+'weather_vars2_all_monthly.npz',allow_pickle='TRUE')

vars_all1 = weather_data1['weather_vars']
varnames1 = weather_data1['varnames']
locnames1 = weather_data1['locnames']

vars_all2 = weather_data2['weather_vars']
varnames2 = weather_data2['varnames']
locnames2 = weather_data2['locnames']

vars_all1 = vars_all1[:,:,:,:,:,0] # Select location: MLO+OR
vars_all2 = vars_all2[:,:,:,:,:,0] # Select location: MLO+OR
locname = locnames1[0]

# MERGE ALL WEATHER DATA TOGETHER:
vars_all = np.zeros((vars_all1.shape[0]+vars_all2.shape[0],vars_all1.shape[1],vars_all1.shape[2],vars_all1.shape[3]))
vars_all[0:vars_all1.shape[0],:,:,:] = vars_all1[:,:,:,:,0]
vars_all[vars_all1.shape[0]:,:,:,:] = vars_all2[:,:,:,:,0]

varnames = [n for n in varnames1]+[n for n in varnames2]

# MAKE NEW VARIABLES:
MJJA_precip_all = np.nansum(vars_all[6,:,3:7,0],axis=1)
MJJAS_precip_all = np.nansum(vars_all[6,:,2:7,0],axis=1)
JJA_Ta_mean_all = np.nanmean(vars_all[2,:,3:6,0],axis=1)

SON_snow_all = np.nansum(vars_all[11,:,0:3,0],axis=1)


# LOAD NAO DATA

NAO_data = np.load(fp+'NAO_index_NOAA/NAO_index_NOAA_monthly.npz',allow_pickle='TRUE')
NAO_vars = NAO_data['data']
NAO_varnames = ['NAO']

if deseasonalize:
    NAO_vars = deseasonalize_ts(Nwindow,NAO_vars,NAO_varnames,'all_time',time,years)

NAO_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
# boot_iyears_all = np.zeros((len(years),end_dates_arr.shape[1],nboot))*np.nan
for iend in range(end_dates_arr.shape[1]):
    NAO_vars_all[:,:,iend,:]= get_window_monthly_vars(NAO_vars,['Avg. monthly NAO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

NAO_vars_all = np.expand_dims(np.expand_dims(NAO_vars_all[:,:,:,:],axis=-1),axis=0)
NAO_vars_all = np.squeeze(NAO_vars_all[:,:,:,:,0,:])


# LOAD PDO DATA
fn = 'PDO_index_NOAA_monthly_ersstv3.npz'
# fn = 'PDO_index_NOAA_monthly_ersstv5.npz'
# fn = 'PDO_index_NOAA_monthly_hadisst1.npz'
PDO_data = np.load(fp+'PDO_index_NOAA/'+fn,allow_pickle='TRUE')
PDO_vars = PDO_data['PDO_data']
PDO_varnames = ['PDO']

if deseasonalize:
    PDO_vars = deseasonalize_ts(Nwindow,PDO_vars,PDO_varnames,'all_time',time,years)

PDO_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
# boot_iyears_all = np.zeros((len(years),end_dates_arr.shape[1],nboot))*np.nan
for iend in range(end_dates_arr.shape[1]):
    PDO_vars_all[:,:,iend,:]= get_window_monthly_vars(PDO_vars,['Avg. monthly PDO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

PDO_vars_all = np.expand_dims(np.expand_dims(PDO_vars_all[:,:,:,:],axis=-1),axis=0)
PDO_vars_all = np.squeeze(PDO_vars_all[:,:,:,:,0,:])


# LOAD EL NINO DATA
fn = 'Nino34_index_NOAA_monthly.npz'
ENSO_data = np.load(fp+'Nino34_index_NOAA/'+fn,allow_pickle='TRUE')
ENSO_vars = ENSO_data['Nino34_data']
ENSO_varnames = ['nino34']
# fn = 'ONI_index_NOAA_monthly.npz'
# ENSO_data = np.load(fp+'ONI_index_NOAA/'+fn,allow_pickle='TRUE')
# ENSO_vars = ENSO_data['ONI_data']
# ENSO_varnames = ['ONI']

if deseasonalize:
    ENSO_vars = deseasonalize_ts(Nwindow,ENSO_vars,ENSO_varnames,'all_time',time,years)

ENSO_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
# boot_iyears_all = np.zeros((len(years),end_dates_arr.shape[1],nboot))*np.nan
for iend in range(end_dates_arr.shape[1]):
    ENSO_vars_all[:,:,iend,:]= get_window_monthly_vars(ENSO_vars,['Avg. monthly '+ENSO_varnames[0]],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

ENSO_vars_all = np.expand_dims(np.expand_dims(ENSO_vars_all[:,:,:,:],axis=-1),axis=0)
ENSO_vars_all = np.squeeze(ENSO_vars_all[:,:,:,:,0,:])


# LOAD DISCHARGE AND LEVEL DATA
loc_discharge = 'Lasalle'
loc_level = 'PointeClaire'

discharge_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_discharge+'.npz',allow_pickle=True)
level_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_level+'.npz',allow_pickle=True)

discharge_vars = discharge_data['discharge'][:,1]
level_vars = level_data['level'][:,1]

if deseasonalize:
    discharge_vars = deseasonalize_ts(Nwindow,discharge_vars,['discharge'],'all_time',time,years)
    level_vars = deseasonalize_ts(Nwindow,level_vars,['level'],'all_time',time,years)

# Separate in different windows with different end dates
discharge_vars_all = np.zeros((1,len(years),len(window_arr),end_dates_arr.shape[1],2,1))*np.nan
level_vars_all = np.zeros((1,len(years),len(window_arr),end_dates_arr.shape[1],2,1))*np.nan
for iend in range(end_dates_arr.shape[1]):
    discharge_vars_all[:,:,:,iend,:,0] = get_window_vars(np.expand_dims(discharge_vars,axis=1),['Avg. discharge'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
    level_vars_all[:,:,:,iend,:,0] = get_window_vars(np.expand_dims(level_vars,axis=1),['Avg. level'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

water_levels_discharge_vars_all = np.zeros((2,len(years),len(window_arr),end_dates_arr.shape[1]))*np.nan
water_levels_discharge_vars_all[0,:,:,:] = discharge_vars_all[0,:,:,:,0,0].copy()
water_levels_discharge_vars_all[1,:,:,:] = level_vars_all[0,:,:,:,0,0].copy()
water_levels_discharge_varnames = ['discharge', 'level']



# KEEP ONLY SPECIFIED YEARS
year_start = 1992
year_end = 2021

iy_start = np.where(years == year_start)[0][0]
iy_end = np.where(years == year_end)[0][0] + 1

vars_all = vars_all[:,iy_start:iy_end,:,:]
NAO_vars_all = NAO_vars_all[iy_start:iy_end,:,:]
PDO_vars_all = PDO_vars_all[iy_start:iy_end,:,:]
ENSO_vars_all = ENSO_vars_all[iy_start:iy_end,:,:]
water_levels_discharge_vars_all = water_levels_discharge_vars_all[:,iy_start:iy_end,:,:]
Twater_vars_all = Twater_vars_all[iy_start:iy_end,:,:]
years = years[iy_start:iy_end]
avg_freezeup_doy = avg_freezeup_doy[iy_start:iy_end]

MJJA_precip_all = MJJA_precip_all[iy_start:iy_end]
MJJAS_precip_all = MJJAS_precip_all[iy_start:iy_end]
JJA_Ta_mean_all = JJA_Ta_mean_all[iy_start:iy_end]
SON_snow_all = SON_snow_all[iy_start:iy_end]


nvars = vars_all.shape[0]
nyears = vars_all.shape[1]
nwindows = vars_all.shape[2]
nend = vars_all.shape[3]


# DETREND VARIABLES
if detrend:
    vars_all_detrended = np.zeros(vars_all.shape)*np.nan
    NAO_vars_all_detrended = np.zeros(NAO_vars_all.shape)*np.nan
    PDO_vars_all_detrended = np.zeros(PDO_vars_all.shape)*np.nan
    ENSO_vars_all_detrended = np.zeros(ENSO_vars_all.shape)*np.nan
    Twater_vars_all_detrended = np.zeros(Twater_vars_all.shape)*np.nan
    water_levels_discharge_vars_all_detrended = np.zeros(water_levels_discharge_vars_all.shape)*np.nan
    MJJA_precip_detrended = np.zeros(MJJA_precip_all.shape)*np.nan
    MJJAS_precip_detrended = np.zeros(MJJAS_precip_all.shape)*np.nan
    JJA_Ta_mean_detrended = np.zeros(JJA_Ta_mean_all.shape)*np.nan
    SON_snow_detrended = np.zeros(SON_snow_all.shape)*np.nan

    for ivar in range(nvars):
        for iw in range(nwindows):
            for iend in range(nend):
                xvar = vars_all[ivar,:,iw,iend]
                yvar = avg_freezeup_doy

                vars_all_detrended[ivar,:,iw,iend]= detrend_ts(xvar,years,anomaly)
                avg_freezeup_doy_detrended = detrend_ts(yvar,years,anomaly)
                MJJA_precip_detrended = detrend_ts(MJJA_precip_all,years,anomaly)
                MJJAS_precip_detrended = detrend_ts(MJJAS_precip_all,years,anomaly)
                JJA_Ta_mean_detrended = detrend_ts(JJA_Ta_mean_all,years,anomaly)
                SON_snow_detrended = detrend_ts(SON_snow_all,years,anomaly)

    for ivar in range(2):
        for iw in range(nwindows):
            for iend in range(nend):
                Qvar = water_levels_discharge_vars_all[ivar,:,iw,iend]

                water_levels_discharge_vars_all_detrended[ivar,:,iw,iend]= detrend_ts(Qvar,years,anomaly)

    for iw in range(nwindows):
         for iend in range(nend):
             xvar = NAO_vars_all[:,iw,iend]
             yvar = PDO_vars_all[:,iw,iend]
             zvar = ENSO_vars_all[:,iw,iend]
             Tvar = Twater_vars_all[:,iw,iend]

             NAO_vars_all_detrended[:,iw,iend] = detrend_ts(xvar,years,anomaly)
             PDO_vars_all_detrended[:,iw,iend] = detrend_ts(yvar,years,anomaly)
             ENSO_vars_all_detrended[:,iw,iend] = detrend_ts(zvar,years,anomaly)
             Twater_vars_all_detrended[:,iw,iend] = detrend_ts(Tvar,years,anomaly)

else:
    vars_all_detrended = vars_all.copy()
    NAO_vars_all_detrended = NAO_vars_all.copy()
    PDO_vars_all_detrended = PDO_vars_all.copy()
    ENSO_vars_all_detrended = ENSO_vars_all.copy()
    avg_freezeup_doy_detrended = avg_freezeup_doy.copy()
    Twater_vars_all_detrended = Twater_vars_all.copy()
    water_levels_discharge_vars_all_detrended = water_levels_discharge_vars_all.copy()
    MJJA_precip_detrended = MJJA_precip_all.copy()
    MJJAS_precip_detrended = MJJAS_precip_all.copy()
    JJA_Ta_mean_detrended = JJA_Ta_mean_all.copy()
    SON_snow_detrended = SON_snow_all.copy()


# RESHAPE VARS TO HAVE 11 MONTHS DATA IN CORRECT ORDER:
weather_monthly = np.zeros((nvars,nyears,11))
NAO_monthly = np.zeros((nyears,11))
ENSO_monthly = np.zeros((nyears,11))
PDO_monthly = np.zeros((nyears,11))
Twater_monthly = np.zeros((nyears,11))
QL_monthly = np.zeros((2,nyears,11))

for ivar in range(nvars):
    weather_monthly[ivar,:,3:]= np.fliplr(vars_all_detrended[ivar,:,:,0])
    weather_monthly[ivar,:,0:3]= np.fliplr(vars_all_detrended[ivar,:,5:8,-1])

NAO_monthly[:,3:]= np.fliplr(NAO_vars_all_detrended[:,:,0])
NAO_monthly[:,0:3]= np.fliplr(NAO_vars_all_detrended[:,5:8,-1])

PDO_monthly[:,3:]= np.fliplr(PDO_vars_all_detrended[:,:,0])
PDO_monthly[:,0:3]= np.fliplr(PDO_vars_all_detrended[:,5:8,-1])

ENSO_monthly[:,3:]= np.fliplr(ENSO_vars_all_detrended[:,:,0])
ENSO_monthly[:,0:3]= np.fliplr(ENSO_vars_all_detrended[:,5:8,-1])

Twater_monthly[:,3:]= np.fliplr(Twater_vars_all_detrended[:,:,0])
Twater_monthly[:,0:3]= np.fliplr(Twater_vars_all_detrended[:,5:8,-1])

for ivar in range(2):
    QL_monthly[ivar,:,3:]= np.fliplr(water_levels_discharge_vars_all_detrended[ivar,:,:,0])
    QL_monthly[ivar,:,0:3]= np.fliplr(water_levels_discharge_vars_all_detrended[ivar,:,5:8,-1])


# SELECT PREDICTORS
Jan_PDO = PDO_monthly[:,0]

Feb_PDO = PDO_monthly[:,1]

Apr_snow = weather_monthly[11,:,3]
Apr_NAO = NAO_monthly[:,3]
Apr_FDD = weather_monthly[4,:,3]
Apr_Ta_mean = weather_monthly[2,:,3]
Apr_discharge = QL_monthly[0,:,3]
Apr_level = QL_monthly[1,:,3]

May_Twater = Twater_monthly[:,4]
May_Ta_mean = weather_monthly[2,:,4]

Aug_NAO = NAO_monthly[:,7]

Sept_RH = weather_monthly[14,:,8]
Sept_clouds = weather_monthly[12,:,8]
Sept_NAO = NAO_monthly[:,8]
Sept_SLP = weather_monthly[7,:,8]
Sept_precip = weather_monthly[6,:,8]
Sept_snow = weather_monthly[11,:,8]

Oct_Twater = Twater_monthly[:,9]
Oct_wind = weather_monthly[8,:,9]
Oct_RH = weather_monthly[14,:,9]
Oct_clouds = weather_monthly[12,:,9]
Oct_SLP = weather_monthly[7,:,9]
Oct_precip = weather_monthly[6,:,9]
Oct_discharge = QL_monthly[0,:,9]
Oct_level = QL_monthly[1,:,9]
Oct_snow = weather_monthly[11,:,9]

Nov_Twater = Twater_monthly[:,10]
Nov_snow = weather_monthly[11,:,10]
Nov_Ta_mean = weather_monthly[2,:,10]
Nov_Ta_min = weather_monthly[1,:,10]
Nov_Ta_max = weather_monthly[0,:,10]
Nov_SLP = weather_monthly[7,:,10]
Nov_TDD = weather_monthly[3,:,10]
Nov_FDD = weather_monthly[4,:,10]
Nov_RH = weather_monthly[14,:,10]
Nov_clouds = weather_monthly[12,:,10]
Nov_precip = weather_monthly[6,:,10]
Nov_discharge = QL_monthly[0,:,10]
Nov_level = QL_monthly[1,:,10]
Nov_NAO = NAO_monthly[:,10]

MJJA_precip = MJJA_precip_detrended
MJJAS_precip = MJJAS_precip_detrended
JJA_Ta_mean = JJA_Ta_mean_detrended

SON_snow = SON_snow_detrended


# Prepare predictor data in DataFrame
data = {'Year': years,

        'Freeze-up': avg_freezeup_doy,
        'Freeze-up Anomaly': avg_freezeup_doy_detrended,

        'Jan. PDO Anomaly': Jan_PDO,

        'Feb. PDO Anomaly': Feb_PDO,

        'Apr. Snowfall Anomaly': Apr_snow,
        'Apr. NAO Anomaly': Apr_NAO,
        'Apr. aFDD Anomaly': Apr_FDD,
        'Apr. Ta_avg Anomaly': Apr_Ta_mean,
        'Apr. discharge Anomaly': Apr_discharge,
        'Apr. level Anomaly': Apr_level,

        'May Ta_avg Anomaly': May_Ta_mean,
        'May Twater Anomaly': May_Twater,

        'Aug. NAO Anomaly': Aug_NAO,

        'Sept. RH Anomaly': Sept_RH,
        'Sept. Cloud Anomaly': Sept_clouds,
        'Sept. NAO Anomaly': Sept_NAO,
        'Sept. SLP Anomaly': Sept_SLP,
        'Sept. precip Anomaly': Sept_precip,
        'Sept. Snowfall Anomaly': Sept_snow,

        'Oct. windspeed Anomaly': Oct_wind,
        'Oct. Twater Anomaly': Oct_Twater,
        'Oct. RH Anomaly': Oct_RH,
        'Oct. Cloud Anomaly': Oct_clouds,
        'Oct. SLP Anomaly': Oct_SLP,
        'Oct. precip Anomaly': Oct_precip,
        'Oct. discharge Anomaly': Oct_discharge,
        'Oct. level Anomaly': Oct_level,
        'Oct. Snowfall Anomaly': Oct_snow,

        'Nov. Ta_avg Anomaly': Nov_Ta_mean,
        'Nov. Ta_min Anomaly': Nov_Ta_min,
        'Nov. Ta_max Anomaly': Nov_Ta_max,
        'Nov. Snowfall Anomaly': Nov_snow,
        'Nov. SLP Anomaly': Nov_SLP,
        'Nov. aTDD Anomaly': Nov_TDD,
        'Nov. aFDD Anomaly': Nov_FDD,
        'Nov. RH Anomaly': Nov_RH,
        'Nov. Cloud Anomaly': Nov_clouds,
        'Nov. precip Anomaly': Nov_precip,
        'Nov. discharge Anomaly': Nov_discharge,
        'Nov. level Anomaly': Nov_level,
        'Nov. Twater Anomaly': Nov_Twater,
        'Nov. NAO Anomaly': Nov_NAO,

        'Summer precip Anomaly': MJJA_precip,
        'Summer+Sept precip Anomaly': MJJAS_precip,
        'Summer Ta_avg Anomaly': JJA_Ta_mean,

        'Fall Snowfall Anomaly': SON_snow
        }

columns=['Year',
        'Freeze-up',
        'Freeze-up Anomaly',
        'Jan. PDO Anomaly',
        'Feb. PDO Anomaly',
        'Apr. Snowfall Anomaly',
        'Apr. NAO Anomaly',
        'Apr. aFDD Anomaly',
        'Apr. Ta_avg Anomaly',
        'Apr. discharge Anomaly',
        'Apr. level Anomaly',
        'May Ta_avg Anomaly',
        'May Twater Anomaly',
        'Aug. NAO Anomaly',
        'Sept. RH Anomaly',
        'Sept. Cloud Anomaly',
        'Sept. NAO Anomaly',
        'Sept. SLP Anomaly',
        'Sept. precip Anomaly',
        'Sept. Snowfall Anomaly',
        'Oct. windspeed Anomaly',
        'Oct. Twater Anomaly',
        'Oct. RH Anomaly',
        'Oct. Cloud Anomaly',
        'Oct. SLP Anomaly',
        'Oct. precip Anomaly',
        'Oct. discharge Anomaly',
        'Oct. level Anomaly',
        'Oct. Snowfall Anomaly',
        'Nov. Ta_avg Anomaly',
        'Nov. Ta_min Anomaly',
        'Nov. Ta_max Anomaly',
        'Nov. Snowfall Anomaly',
        'Nov. SLP Anomaly',
        'Nov. aTDD Anomaly',
        'Nov. aFDD Anomaly',
        'Nov. RH Anomaly',
        'Nov. Cloud Anomaly',
        'Nov. precip Anomaly',
        'Nov. discharge Anomaly',
        'Nov. level Anomaly',
        'Nov. Twater Anomaly',
        'Nov. NAO Anomaly',
        'Summer precip Anomaly',
        'Summer+Sept precip Anomaly',
        'Summer Ta_avg Anomaly',
        'Fall Snowfall Anomaly'
        ]
df = pd.DataFrame(data,columns=columns)
#%%
nyears = years.shape[0]-1
training_size = 15

rolling_training = True
# rolling_training = False

# anomaly = True
anomaly = False

# xall = All years
# yall =  Observed FUD for all years
# xf = Years of forecast period
# yf_true = Observed FUD during forecast period
# yf = Predictions during forecast period
# xh = Years of hindcast period
# yh = Predictions during hindcast period
#%%
# PREDICTIONS INCLUDING OCT. VARIABLES (BUT NOT NOV.)
fig, ax = plt.subplots()
ax.set_xlabel('Year')
ax.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))

# ORANGE
x_model = ['Nov. level Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(2))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(3))
print('ORANGE - 2021 prediction:  ')

#ADD LAST 28-YRS AVERAGE OF FREEZE-UP DATE:
if not anomaly:
    ax.plot(xall,np.ones(xall.size)*np.nanmean(avg_freezeup_doy),'-',color=[0.7, 0.7 ,0.7])
    ax.fill_between(years, np.nanpercentile(avg_freezeup_doy,25), np.nanpercentile(avg_freezeup_doy,75),color=[0.7, 0.7,0.7], alpha= 0.15)


#%%
# PREDICTIONS INCLUDING OCT. VARIABLES (BUT NOT NOV.)
fig, ax = plt.subplots()
ax.set_xlabel('Year')
ax.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))

# ORANGE
x_model = ['Oct. windspeed Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(2))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(3))
df_x_predict = pd.DataFrame({'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(2))
print('ORANGE - 2021 prediction:  '+str(y_predict))

#ADD LAST 28-YRS AVERAGE OF FREEZE-UP DATE:
if not anomaly:
    ax.plot(xall,np.ones(xall.size)*np.nanmean(avg_freezeup_doy),'-',color=[0.7, 0.7 ,0.7])
    ax.fill_between(years, np.nanpercentile(avg_freezeup_doy,25), np.nanpercentile(avg_freezeup_doy,75),color=[0.7, 0.7,0.7], alpha= 0.15)

# # PINK
# x_model = ['Apr. Snowfall Anomaly']
# [model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
# mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
# mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
# ax.plot(xall,yall,'o-',color='k')
# ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(12))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(13))
# df_x_predict = pd.DataFrame({'Apr. Snowfall Anomaly':[df['Apr. Snowfall Anomaly'][29]]})
# x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
# y_predict = model.predict(x_predict)
# ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(12))
# print('2021 prediction:  '+str(y_predict))

# GREEN ****************************
x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(4))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(5))
df_x_predict = pd.DataFrame({'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Apr. Snowfall Anomaly':[df['Apr. Snowfall Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(4))
print('GREEN - 2021 prediction:  '+str(y_predict))

# PURPLE
x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(8))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(9))
df_x_predict = pd.DataFrame({'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Apr. Snowfall Anomaly':[df['Apr. Snowfall Anomaly'][29]],'Apr. NAO Anomaly':[df['Apr. NAO Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(8))
print('PURPLE - 2021 prediction:  '+str(y_predict))

# # BROWN
# x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly','May Ta_avg Anomaly']
# [model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
# mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
# mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
# ax.plot(xall,yall,'o-',color='k')
# ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(10))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(11))
# df_x_predict = pd.DataFrame({'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Apr. Snowfall Anomaly':[df['Apr. Snowfall Anomaly'][29]],'May Ta_avg Anomaly':[df['May Ta_avg Anomaly'][29]]})
# x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
# y_predict = model.predict(x_predict)
# ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(10))
# print('BROWN - 2021 prediction:  '+str(y_predict))

# BLUE
x_model = ['Oct. windspeed Anomaly','Oct. level Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(0))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(1))
df_x_predict = pd.DataFrame({'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Oct. level Anomaly':[df['Oct. level Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(0))
print('BLUE - 2021 prediction:  '+str(y_predict))

# # BROWN
# x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Oct. level Anomaly']
# [model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
# mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
# mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
# ax.plot(xall,yall,'o-',color='k')
# ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(10))
# # ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(11))
# print('BROWN - 2021 prediction:  ')


#%%
#PREDICTIONS INCLUDING NOV. VARIABLES

fig, ax = plt.subplots()
ax.set_xlabel('Year')
ax.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))
# xall = All years
# yall =  Observed FUD for all years
# xf = Years of forecast period
# yf_true = Observed FUD during forecast period
# yf = Predictions during forecast period
# xh = Years of hindcast period
# yh = Predictions during hindcast period



#ADD LAST 28-YRS AVERAGE OF FREEZE-UP DATE:
if not anomaly:
    ax.plot(xall,np.ones(xall.size)*np.nanmean(avg_freezeup_doy),'-',color=[0.7, 0.7 ,0.7])
    ax.fill_between(years, np.nanpercentile(avg_freezeup_doy,25), np.nanpercentile(avg_freezeup_doy,75),color=[0.7, 0.7,0.7], alpha= 0.15)


# GREEN
x_model = ['Nov. NAO Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# x_model = ['Nov. aTDD Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(4))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(5))
df_x_predict = pd.DataFrame({'Nov. NAO Anomaly':[df['Nov. NAO Anomaly'][29]],'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Apr. Snowfall Anomaly':[df['Apr. Snowfall Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(4))
print('GREEN - 2021 prediction:  '+str(y_predict))

# SICK GREEN
x_model = ['Nov. NAO Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# x_model = ['Nov. aTDD Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(16))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(17))
df_x_predict = pd.DataFrame({'Nov. NAO Anomaly':[df['Nov. NAO Anomaly'][29]],'Nov. precip Anomaly':[df['Nov. precip Anomaly'][29]],'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Apr. Snowfall Anomaly':[df['Apr. Snowfall Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(16))
print('SICK GREEN - 2021 prediction:  '+str(y_predict))


# PURPLE
x_model = ['Nov. NAO Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# x_model = ['Nov. aTDD Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(8))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(9))
df_x_predict = pd.DataFrame({'Nov. NAO Anomaly':[df['Nov. NAO Anomaly'][29]],'Apr. NAO Anomaly':[df['Apr. NAO Anomaly'][29]],'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Apr. Snowfall Anomaly':[df['Apr. Snowfall Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(8))
print('PURPLE - 2021 prediction:  '+str(y_predict))


# PINK
x_model = ['Nov. NAO Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# x_model = ['Nov. aTDD Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(12))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(13))
df_x_predict = pd.DataFrame({'Nov. NAO Anomaly':[df['Nov. NAO Anomaly'][29]],'Nov. precip Anomaly':[df['Nov. precip Anomaly'][29]],'Apr. NAO Anomaly':[df['Apr. NAO Anomaly'][29]],'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Apr. Snowfall Anomaly':[df['Apr. Snowfall Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(12))
print('PinK - 2021 prediction:  '+str(y_predict))


# BROWN ****************************
x_model = ['Nov. NAO Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
# x_model = ['Nov. aTDD Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(10))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(11))
df_x_predict = pd.DataFrame({'Nov. NAO Anomaly':[df['Nov. NAO Anomaly'][29]],'Nov. precip Anomaly':[df['Nov. precip Anomaly'][29]],'Oct. windspeed Anomaly':[df['Oct. windspeed Anomaly'][29]],'Oct. level Anomaly':[df['Oct. level Anomaly'][29]]})
x_predict = sm.add_constant(df_x_predict, has_constant='add') # adding a constant
y_predict = model[-1].predict(x_predict)
ax.plot(2021,y_predict, '*', color= plt.get_cmap('tab20')(10))
print('BROWN - 2021 prediction:  '+str(y_predict))



#%%
# PREDICTIONS INCLUDING NOV. VARIABLES #2

fig, ax = plt.subplots()
ax.set_xlabel('Year')
ax.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))


#ADD LAST 28-YRS AVERAGE OF FREEZE-UP DATE:
if not anomaly:
    ax.plot(xall,np.ones(xall.size)*np.nanmean(avg_freezeup_doy),'-',color=[0.7, 0.7 ,0.7])
    ax.fill_between(years, np.nanpercentile(avg_freezeup_doy,25), np.nanpercentile(avg_freezeup_doy,75),color=[0.7, 0.7,0.7], alpha= 0.15)

#  ***** 2 SICK GREEN (better than RED)
x_model = ['Nov. NAO Anomaly','Oct. windspeed Anomaly']
# x_model = ['Nov. aTDD Anomaly','Oct. windspeed Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Oct. windspeed Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Oct. windspeed Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Oct. windspeed Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(16))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(17))
print('2SICK GREEN - 2021 prediction:  ')

# ***** 2 BLUE ****************************
x_model = ['Nov. NAO Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly']
# x_model = ['Nov. aTDD Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(0))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(1))
print('2BLUE - 2021 prediction:  ')


# 2 BROWN ****************************
x_model = ['Nov. NAO Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
# x_model = ['Nov. aTDD Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(10))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(11))
print('2BROWN - 2021 prediction:  ')

#  ----> 2 GRAY****************************
x_model = ['Nov. NAO Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Nov. level Anomaly']
# x_model = ['Nov. aTDD Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Nov. level Anomaly']
# x_model = ['Nov. Ta_avg Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Nov. level Anomaly']
# x_model = ['Nov. Ta_max Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Nov. level Anomaly']
# x_model = ['Nov. Ta_min Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Nov. level Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(14))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(15))
print('2GRAY- 2021 prediction:  ')

# # GREEN ****************************
x_model = ['Nov. aTDD Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly','Nov. NAO Anomaly']
# # x_model = ['Nov. Ta_avg Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly','Nov. NAO Anomaly']
# # x_model = ['Nov. Ta_max Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly','Nov. NAO Anomaly']
# # x_model = ['Nov. Ta_min Anomaly','Nov. precip Anomaly','Oct. windspeed Anomaly','Oct. level Anomaly','Nov. NAO Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(4))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(5))
print('GREEN - 2021 prediction:  ')

#%%

fig, ax = plt.subplots()
ax.set_xlabel('Year')
ax.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))


x_model = ['Feb. PDO Anomaly','Apr. aFDD Anomaly']
[model,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f] = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(4))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(5))
print('GREEN - 2021 prediction:  ')


#%%
# Dec1_vars = columns
# Dec1_df = df[Dec1_vars]

# Nov1_vars = []
# for ic,c in enumerate(Dec1_vars):
#     if 'Nov' not in c:
#         Nov1_vars += [c]
# Nov1_df = df[Nov1_vars]

# Oct1_vars = []
# for ic,c in enumerate(Nov1_vars):
#     if 'Oct' not in c:
#         Oct1_vars += [c]
# Oct1_df = df[Oct1_vars]

# #%%
# MAE_Nov1_f = np.zeros(len(Nov1_vars)-3)*np.nan
# RMSE_Nov1_f = np.zeros(len(Nov1_vars)-3)*np.nan
# Rsqr_Nov1_f = np.zeros(len(Nov1_vars)-3)*np.nan
# Rsqr_adj_Nov1_f = np.zeros(len(Nov1_vars)-3)*np.nan
# sigma_err_Nov1_f = np.zeros(len(Nov1_vars)-3)*np.nan

# MAE_Nov1_h = np.zeros(len(Nov1_vars)-3)*np.nan
# RMSE_Nov1_h = np.zeros(len(Nov1_vars)-3)*np.nan
# Rsqr_Nov1_h = np.zeros(len(Nov1_vars)-3)*np.nan
# Rsqr_adj_Nov1_h = np.zeros(len(Nov1_vars)-3)*np.nan
# sigma_err_Nov1_h = np.zeros(len(Nov1_vars)-3)*np.nan

# for i in range(len(Nov1_vars)-3):
#     x_model = [Nov1_vars[i+3]]
#     [model,xall,yall,xf,yf_true,yf,xh,yh,
#     MAE_Nov1_h[i],RMSE_Nov1_h[i],Rsqr_Nov1_h[i],Rsqr_adj_Nov1_h[i],sigma_err_Nov1_h[i],
#     MAE_Nov1_f[i],RMSE_Nov1_f[i],Rsqr_Nov1_f[i],Rsqr_adj_Nov1_f[i],sigma_err_Nov1_f[i]] = freezeup_multiple_linear_regression_model(Nov1_df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)


# stats_Nov1_f = pd.DataFrame(np.array([MAE_Nov1_f,RMSE_Nov1_f,Rsqr_Nov1_f,Rsqr_adj_Nov1_f,sigma_err_Nov1_f]).T,
#                   columns=['MAE', 'RMSE','Rsqr','Rsqr_adj','sig_err'],
#                   index=Nov1_vars[3:])

# #%%
# MAE_Dec1_f = np.zeros(len(Dec1_vars)-3)*np.nan
# RMSE_Dec1_f = np.zeros(len(Dec1_vars)-3)*np.nan
# Rsqr_Dec1_f = np.zeros(len(Dec1_vars)-3)*np.nan
# Rsqr_adj_Dec1_f = np.zeros(len(Dec1_vars)-3)*np.nan
# sigma_err_Dec1_f = np.zeros(len(Dec1_vars)-3)*np.nan

# MAE_Dec1_h = np.zeros(len(Dec1_vars)-3)*np.nan
# RMSE_Dec1_h = np.zeros(len(Dec1_vars)-3)*np.nan
# Rsqr_Dec1_h = np.zeros(len(Dec1_vars)-3)*np.nan
# Rsqr_adj_Dec1_h = np.zeros(len(Dec1_vars)-3)*np.nan
# sigma_err_Dec1_h = np.zeros(len(Dec1_vars)-3)*np.nan

# for i in range(len(Dec1_vars)-3):
#     x_model = [Dec1_vars[i+3]]
#     [model,xall,yall,xf,yf_true,yf,xh,yh,
#     MAE_Dec1_h[i],RMSE_Dec1_h[i],Rsqr_Dec1_h[i],Rsqr_adj_Dec1_h[i],sigma_err_Dec1_h[i],
#     MAE_Dec1_f[i],RMSE_Dec1_f[i],Rsqr_Dec1_f[i],Rsqr_adj_Dec1_f[i],sigma_err_Dec1_f[i]] = freezeup_multiple_linear_regression_model(Dec1_df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training)


# stats_Dec1_f = pd.DataFrame(np.array([MAE_Dec1_f,RMSE_Dec1_f,Rsqr_Dec1_f,Rsqr_adj_Dec1_f,sigma_err_Dec1_f]).T,
#                   columns=['MAE', 'RMSE','Rsqr','Rsqr_adj','sig_err'],
#                   index=Dec1_vars[3:])
