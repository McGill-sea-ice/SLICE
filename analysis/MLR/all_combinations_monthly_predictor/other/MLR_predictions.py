#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:18:34 2022

@author: Amelie
"""
#%%
# local_path = '/storage/amelie/'
local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'

import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt
import itertools
import calendar

import statsmodels.api as sm
from netCDF4 import Dataset

from functions import K_to_C
from functions import detect_FUD_from_Tw, detrend_ts
from functions_MLR import get_monthly_vars_from_daily, remove_collinear_features
from functions_MLR import find_models,eval_accuracy_multiple_models,make_metric_df
from functions_MLR import find_all_column_combinations

from analysis.SEAS5.SEAS5_forecast_class import SEAS5frcst
from analysis.MLR.all_combinations_monthly_predictor.MLR_monthly_predictors_forecast_mode import loo_eval

import sklearn.metrics as metrics
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#%%
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

ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")

start_yr = 1992
end_yr = 2019

train_yr_start = start_yr # [1992 - 2007] = 16 years
valid_yr_start = 2008 # [2008 - 2013] = 6 years
test_yr_start = 2014  # [2014 - 2019] = 6 years
nsplits = end_yr-valid_yr_start+1
nfolds = 5

anomaly = True
# anomaly = False
freezeup_opt = 1

#%% Load Twater and FUD data

fp_p_Twater = local_path+'slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
freezeup_doy[np.where(years > end_yr)] = np.nan

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)
Twater_mean[14269:14329] = 0.

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)

# Get FUD categories for accuracy measure:
it_1992 = np.where(years == 1992)[0][0]
it_2008= np.where(years == 2008)[0][0]
mean_FUD = np.nanmean(avg_freezeup_doy[it_1992:it_2008])
std_FUD = np.nanstd(avg_freezeup_doy[it_1992:it_2008])
tercile1_FUD = np.nanpercentile(avg_freezeup_doy[it_1992:it_2008],(1/3.)*100)
tercile2_FUD = np.nanpercentile(avg_freezeup_doy[it_1992:it_2008],(2/3.)*100)

#%% Load all monthly predictor data
fpath_mp = local_path+'slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars'+'_'+'Longueuil'+'.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

# Keep only 1991 to 2020 (inclusive)
it_start = np.where(years == start_yr)[0][0]
it_end = np.where(years == end_yr)[0][0]

years = years[it_start:it_end+1]
avg_freezeup_doy = avg_freezeup_doy[it_start:it_end+1]
monthly_pred_data = monthly_pred_data[:,it_start:it_end+1,:]

p_list = ['Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','NAO','NAO','NAO','NAO','NAO','PDO','PDO','PDO','PDO','PDO','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Tot. snowfall','Tot. snowfall','Tot. snowfall', 'Avg. windspeed', 'Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)', 'Avg. LH (sfc)', 'Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)' ]
m_list = [12,                11,                 10,               9,                8,                   12,                       11,                       10,                      9,                        8,                        12,   11,    10,     9,  8,    12,    11,   10,    9, 8,     12,                         11,                           10,                         9,                           8,                            12,           11,            10,           9,              8,            12,               11,              10,             8,                9,                10,             11,             12,              8,                   9 ,                  10,                  11,                  12,                   8,                9,              10,           11,              12,              8,              9,              10,            11,             12,             ]
# p_list = ['Avg. Ta_mean','Tot. snowfall','Tot. snowfall']
# m_list = [12,             11,             12            ]

# Make dataframe with all predictors
month_str = ['Jan. ', 'Feb. ', 'Mar. ', 'Apr. ', 'May ', 'Jun. ','Jul. ', 'Aug. ', 'Sep. ', 'Oct. ', 'Nov. ','Dec. ']
pred_arr = np.zeros((monthly_pred_data.shape[1],len(p_list)))*np.nan
col = []
for i in range(len(p_list)):
    ipred = np.where(pred_names == p_list[i])[0][0]
    pred_arr[:,i] = monthly_pred_data[ipred,:,m_list[i]-1]
    col.append(month_str[m_list[i]-1]+p_list[i])
pred_df =  pd.DataFrame(pred_arr,columns=col)

#%%
# List of 2022 monthly predictors
# Obtained from 'ERA5_data_preparation_with_dailysum.py' (except discharge and water level - obtained from raw txt files downloaded from web)

SeptNAO2022 = -0.702
SeptSH2022 = -76754.74194477081
OctSH2022 = -34120.33740253003
SeptCloudCover2022 = 0.66412
OctCloudCover2022 = 0.557011
OctSLRdischarge2022 = 7554.41
NovTamean2022 = 4.275217950635346
Novsnowfall2022 = 0.0003518938100026512
SeptSW2022 = 544177.479283211
AugSW2022 = 748539.3214844235

# pred = ['Dec. Avg. Ta_mean','Nov. Tot. snowfall','Dec. Tot. snowfall']

#%%
frcst_start = 'Nov. 1st'
pred = ['Oct. Avg. cloud cover', 'Sep. NAO','Sep. Avg. SH (sfc)',]
pred_2022 = [OctCloudCover2022, SeptNAO2022,  SeptSH2022]

#Make MLR model in LOO mode with IDEAL FORECAST = REALITY:
X, y = pred_df[pred], pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])
y_pred_test, model, Xscaler = loo_eval(LinearRegression(),'mlr',X,y,years,show_coeff=False,plot=False)

# Make prediction with trained model
y_pred_2022 = model.predict(Xscaler.transform(pd.DataFrame(np.expand_dims(np.array(pred_2022),0))))
plt.figure()
plt.plot(years,y, 'o-', color='k')
plt.plot(years,y_pred_test, 'o-', color = plt.get_cmap('tab20')(0), label=pred)
plt.plot(2022,y_pred_2022,'*', markersize = 15,color = plt.get_cmap('tab20')(1))
plt.title(frcst_start)
# plt.legend()
print(y_pred_2022)

#%%
frcst_start = 'Dec. 1st'
pred = ['Oct. Avg. cloud cover', 'Sep. Avg. cloud cover', 'Sep. NAO','Nov. Avg. Ta_mean',]
pred_2022 = [OctCloudCover2022, SeptCloudCover2022, SeptNAO2022,  NovTamean2022]
# pred = ['Oct. Avg. cloud cover', 'Sep. NAO', 'Oct. Avg. discharge St-L. River','Nov. Avg. Ta_mean',]
# pred_2022 = [OctCloudCover2022, SeptNAO2022, OctSLRdischarge2022, NovTamean2022]
# pred = ['Oct. Avg. cloud cover', 'Sep. Avg. cloud cover', 'Sep. NAO']
# pred_2022 = [OctCloudCover2022, SeptCloudCover2022, SeptNAO2022]
# pred = ['Oct. Avg. discharge St-L. River','Nov. Avg. Ta_mean',]
# pred_2022 = [OctSLRdischarge2022, NovTamean2022]
# pred = ['Oct. Avg. cloud cover', 'Sep. Avg. cloud cover', 'Nov. Tot. snowfall',]
# pred_2022 = [OctCloudCover2022, SeptCloudCover2022, Novsnowfall2022]

#Make MLR model in LOO mode with IDEAL FORECAST = REALITY:
X, y = pred_df[pred], pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])
y_pred_test, model, Xscaler = loo_eval(LinearRegression(),'mlr',X,y,years,show_coeff=False,plot=False)

# Make prediction with trained model
y_pred_2022 = model.predict(Xscaler.transform(pd.DataFrame(np.expand_dims(np.array(pred_2022),0))))
plt.figure()
plt.plot(years,y, 'o-', color='k')
plt.plot(years,y_pred_test, 'o-', color = plt.get_cmap('tab20')(0), label=pred)
plt.plot(2022,y_pred_2022,'*', markersize = 15,color = plt.get_cmap('tab20')(1))
plt.title(frcst_start)
# plt.legend()
print(y_pred_2022)
