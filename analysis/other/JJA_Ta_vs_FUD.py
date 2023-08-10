#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:30:02 2022

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import statsmodels.api as sm

from functions import detect_FUD_from_Tw, detrend_ts

#%%
# p_critical = 0.01
p_critical = 0.05

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

ignore_warnings = False
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")


start_doy_arr    = [300,         307,       314,         321,         328,         335]
start_doy_labels = ['Oct. 27th', 'Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st']

replace_with_nan = False

detrend_FUD = False
detrend = False
if detrend:
   anomaly = 'linear'


#%%
# LOAD DATA AND GET FUD TIME SERIES

# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
freezeup_doy[np.where(years == 2020)] = np.nan

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)
Twater_mean[14269:14329] = 0.

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)

# Load monthly predictor data
fpath_mp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

df_May = pd.DataFrame(monthly_pred_data[:,:,4].T, columns = pred_names)
df_Jun = pd.DataFrame(monthly_pred_data[:,:,5].T, columns = pred_names)
df_Jul = pd.DataFrame(monthly_pred_data[:,:,6].T, columns = pred_names)
df_Aug = pd.DataFrame(monthly_pred_data[:,:,7].T, columns = pred_names)
df_Sep = pd.DataFrame(monthly_pred_data[:,:,8].T, columns = pred_names)
df_Oct = pd.DataFrame(monthly_pred_data[:,:,9].T, columns = pred_names)
df_Nov = pd.DataFrame(monthly_pred_data[:,:,10].T, columns = pred_names)

#%%
# ACCUMULATE DEGREES:
TDD_MJJ = df_May['Tot. TDD']+df_Jun['Tot. TDD']+df_Jul['Tot. TDD']
TDD_JJA = df_Jun['Tot. TDD']+df_Jul['Tot. TDD']+df_Aug['Tot. TDD']
TDD_JJAS = df_Jun['Tot. TDD']+df_Jul['Tot. TDD']+df_Aug['Tot. TDD']+df_Sep['Tot. TDD']
TDD_JJASO = df_Jun['Tot. TDD']+df_Jul['Tot. TDD']+df_Aug['Tot. TDD']+df_Sep['Tot. TDD']+df_Oct['Tot. TDD']
TDD_JJASON = df_Jun['Tot. TDD']+df_Jul['Tot. TDD']+df_Aug['Tot. TDD']+df_Sep['Tot. TDD']+df_Oct['Tot. TDD']+df_Nov['Tot. TDD']

TDD_SO = df_Sep['Tot. TDD']+df_Oct['Tot. TDD']
TDD_ASO = df_Aug['Tot. TDD']+df_Sep['Tot. TDD']+df_Oct['Tot. TDD']
TDD_SON = df_Sep['Tot. TDD']+df_Oct['Tot. TDD']+df_Nov['Tot. TDD']

FDD_SO = df_Sep['Tot. FDD']+df_Oct['Tot. FDD']
FDD_SON = df_Sep['Tot. FDD']+df_Oct['Tot. FDD']+df_Nov['Tot. FDD']

#%%
# AVERAGE TA_MEAN
avgTa_MJJ = np.nanmean(pd.DataFrame([df_May['Tot. TDD'],df_Jun['Tot. TDD'],df_Jul['Tot. TDD']]),axis=0)
avgTa_JJA = np.nanmean(pd.DataFrame([df_Aug['Tot. TDD'],df_Jun['Tot. TDD'],df_Jul['Tot. TDD']]),axis=0)
avgTa_JJAS = np.nanmean(pd.DataFrame([df_Sep['Tot. TDD'],df_Aug['Tot. TDD'],df_Jun['Tot. TDD'],df_Jul['Tot. TDD']]),axis=0)

avgTa_JAS = np.nanmean(pd.DataFrame([df_Sep['Tot. TDD'],df_Aug['Tot. TDD'],df_Jul['Tot. TDD']]),axis=0)
avgTa_ASO = np.nanmean(pd.DataFrame([df_Sep['Tot. TDD'],df_Aug['Tot. TDD'],df_Oct['Tot. TDD']]),axis=0)
avgTa_SON = np.nanmean(pd.DataFrame([df_Sep['Tot. TDD'],df_Nov['Tot. TDD'],df_Oct['Tot. TDD']]),axis=0)
avgTa_SO = np.nanmean(pd.DataFrame([df_Sep['Tot. TDD'],df_Oct['Tot. TDD']]),axis=0)

#%%
# OPTIONAL:
# DETREND TIME SERIES BEFORE CORRELATING
if detrend:
    TDD_MJJ,_ = detrend_ts(TDD_MJJ,years,anomaly)
    TDD_JJA,_ = detrend_ts(TDD_JJA,years,anomaly)
    TDD_JJAS,_ = detrend_ts(TDD_JJAS,years,anomaly)
    TDD_JJASO,_ = detrend_ts(TDD_JJASO,years,anomaly)
    TDD_JJASON,_ = detrend_ts(TDD_JJASON,years,anomaly)

    TDD_SO,_ = detrend_ts(TDD_SO,years,anomaly)
    TDD_ASO,_ = detrend_ts(TDD_ASO,years,anomaly)
    TDD_SON,_ = detrend_ts(TDD_SON,years,anomaly)

    FDD_SO,_ = detrend_ts(FDD_SO,years,anomaly)
    FDD_SON,_ = detrend_ts(FDD_SON,years,anomaly)

    avgTa_MJJ,_ = detrend_ts(avgTa_MJJ,years,anomaly)
    avgTa_JJA,_ = detrend_ts(avgTa_JJA,years,anomaly)
    avgTa_JJAS,_ = detrend_ts(avgTa_JJAS,years,anomaly)
    avgTa_JAS,_ = detrend_ts(avgTa_JAS,years,anomaly)

    avgTa_ASO,_ = detrend_ts(avgTa_ASO,years,anomaly)
    avgTa_SON,_ = detrend_ts(avgTa_SON,years,anomaly)
    avgTa_SO,_ = detrend_ts(avgTa_SO,years,anomaly)

if detrend_FUD:
    avg_freezeup_doy,_ = detrend_ts(avg_freezeup_doy,years,anomaly)

#%%
# CHECK LINEAR CORRELATION BETWEEN ACCUMULATED VARIABLES
# 1) SUMMER TDD
y = avg_freezeup_doy
fig,ax = plt.subplots()

x = TDD_MJJ
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)


x = TDD_JJA
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = TDD_JJAS
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = TDD_JJASO
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = TDD_JJASON
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)



#%%
# CHECK LINEAR CORRELATION BETWEEN ACCUMULATED VARIABLES
# 2) FALL TDD
y = avg_freezeup_doy
fig,ax = plt.subplots()

x = TDD_ASO
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = TDD_SON
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = TDD_SO
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)


#%%
# CHECK LINEAR CORRELATION BETWEEN ACCUMULATED VARIABLES
# 3) FALL FDD
y = avg_freezeup_doy
fig,ax = plt.subplots()

x = FDD_SON
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = FDD_SO
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)


#%%
# CHECK LINEAR CORRELATION BETWEEN ACCUMULATED VARIABLES
# 4) SUMMER Ta
y = avg_freezeup_doy
fig,ax = plt.subplots()

x = avgTa_MJJ
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = avgTa_JJA
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = avgTa_JJAS
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)


x = avgTa_JAS
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)


#%%
# CHECK LINEAR CORRELATION BETWEEN ACCUMULATED VARIABLES
# 4) Fall Ta
y = avg_freezeup_doy
fig,ax = plt.subplots()

x = avgTa_ASO
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = avgTa_SON
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)

x = avgTa_SO
ax.plot(x,y,'o')
linmodel = sm.OLS(y, sm.add_constant(x,has_constant='skip'), missing='drop').fit()
print(linmodel.rsquared,linmodel.f_pvalue)





