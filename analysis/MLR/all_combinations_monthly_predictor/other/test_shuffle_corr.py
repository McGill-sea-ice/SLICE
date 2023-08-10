#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:58:16 2023

@author: amelie
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
import random
import pandas as pd
import datetime as dt
import itertools
import calendar
import matplotlib.pyplot as plt
import statsmodels.api as sm
from functions import detect_FUD_from_Tw, detrend_ts
from functions_MLR import get_monthly_vars_from_daily, remove_collinear_features
from functions_MLR import find_models,eval_accuracy_multiple_models,make_metric_df
from functions_MLR import find_all_column_combinations
import sklearn.metrics as metrics
from functions import linear_fit
#%%

valid_scheme = 'LOOk'
# file_name = './output/MLR_monthly_pred_Jan1st_maxpred5_valid_scheme_LOOk'
# file_name = './output/MLR_monthly_pred_Dec1st_maxpred5_valid_scheme_LOOk'
# file_name = './output/MLR_monthly_pred_Nov1st_maxpred6_valid_scheme_LOOk'
file_name = './output/MLR_monthly_pred_p05_Jan1st_maxpred4_valid_scheme_LOOk'

df_valid_all = pd.read_pickle(file_name+'_df_valid_all')
df_test_all = pd.read_pickle(file_name+'_df_test_all')
df_clim_valid_all = pd.read_pickle(file_name+'_df_clim_valid_all')
df_clim_test_all = pd.read_pickle(file_name+'_df_clim_test_all')
df_select_valid = pd.read_pickle(file_name+'_df_select_valid')
df_select_test = pd.read_pickle(file_name+'_df_select_test')
pred_df_clean = pd.read_pickle(file_name+'_pred_df_clean')


data = np.load(file_name+'.npz', allow_pickle=True)

valid_years=data['valid_years']
test_years=data['test_years']
plot_label = data['plot_label']
years = data['years']
avg_freezeup_doy = data['avg_freezeup_doy']
p_critical = data['p_critical']
date_ref = data['date_ref']
date_start = data['date_start']
date_end = data['date_end']
time = data['time']
start_yr = data['start_yr']
end_yr = data['end_yr']
train_yr_start = data['train_yr_start']
valid_yr_start = data['valid_yr_start']
test_yr_start = data['test_yr_start']
nsplits = data['nsplits']
nfolds = data['nfolds']
max_pred = data['max_pred']
valid_metric = data['valid_metric']


istart_labels = ['Sept. 1st', 'Oct. 1st','Nov. 1st','Dec. 1st','Jan. 1st']
istart_savelabels = ['Sept1st', 'Oct1st','Nov1st','Dec1st','Jan1st']
istart = 4
ind = 0

best_n = np.min((8,len(df_valid_all)))
#%%


#%%
years = np.array([1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021])

start_yr = 1992
end_yr = 2019

# Load Twater and FUD data
fp_p_Twater = local_path+'slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
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

# Load monthly predictor data
fpath_mp = local_path+'slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars'+'_'+'Longueuil'+'.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

# Load monthly predictor data
fpath_mp = local_path+'slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars'+'_'+'Longueuil'+'.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

#%%
# Keep only 1992 to 2020 (inclusive)
it_start = np.where(years == start_yr)[0][0]
it_end = np.where(years == end_yr)[0][0]

years = years[it_start:it_end+1]
avg_freezeup_doy = avg_freezeup_doy[it_start:it_end+1]
monthly_pred_data = monthly_pred_data[:,it_start:it_end+1,:]

#%%
p_list = ['Avg. Ta_mean','Avg. Ta_mean','Tot. FDD','Tot. TDD','Tot. TDD','Avg. SLP','Tot. snowfall','Avg. cloud cover','Avg. level Ottawa River','AO','NAO','NAO','PDO','PDO','Avg. windspeed','Tot. snowfall','Avg. Ta_mean','Tot. TDD','Tot. FDD','Avg. discharge St-L. River','Avg. level Ottawa River','NAO','AO'  ]
m_list = [11,            5,             11,         5,        11,         11,        11,             9,                 10,                       11,   9,    11,   1,    2,   10,              12,             12,            12,        12,        12,                          12,                       12,   12  ]
month_str = ['Jan. ', 'Feb. ', 'Mar. ', 'Apr. ', 'May ', 'Jun. ','Jul. ', 'Aug. ', 'Sep. ', 'Oct. ', 'Nov. ','Dec.']

pred_arr = np.zeros((monthly_pred_data.shape[1],len(p_list)))*np.nan
col = []
for i in range(len(p_list)):
    ipred = np.where(pred_names == p_list[i])[0][0]
    pred_arr[:,i] = monthly_pred_data[ipred,:,m_list[i]-1]
    col.append(month_str[m_list[i]-1]+p_list[i])
pred_df =  pd.DataFrame(pred_arr,columns=col)
#%%

c = []
for i in range(10000):
    rng = np.random.default_rng()
    arr = np.arange(len(avg_freezeup_doy))
    rng.shuffle(arr)
    y = avg_freezeup_doy
    # x = pred_df_clean['Oct. Avg. level Ottawa River']
    x = pred_df_clean['Sep. Avg. cloud cover']
    # fig,ax = plt.subplots(nrows=2)
    # ax[0].plot(x)
    # ax[1].plot(y)
    m = linear_fit(x[arr],y)
    rsqr = m[1]
    c.append(np.sqrt(rsqr))

plt.figure()
plt.hist(c,bins=20)
print(np.sum(np.array(c)>np.sqrt(linear_fit(x,y)[1]))/10000)