#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:25:00 2022

@author: Amelie
"""

#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'
#%%
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path+'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import copy
import time
import os

import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
import dateutil
import calendar

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

# import tensorflow as tf

from functions import rolling_climo

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")


#%%
# Load Data
fpath = local_path+'slice/data/ML_timeseries/'
fname = '/ML_dataset_with_cansips.npz'

with np.load(fpath+fname, allow_pickle='TRUE') as data:
    ds = data['data']
    # date_ref = data['date_ref']
    date_ref = dt.date(1900,1,1)
    region_ERA5 = data['region_ERA5']
    region_cansips = data['region_cansips']
    loc_Twater = data['Twater_loc_list']
    loc_discharge = data['loc_discharge']
    loc_level = data['loc_level']
    loc_levelO = data['loc_levelO']
    labels = [k.decode('UTF-8') for k in data['labels']]

#%%
# Select variables from data set and convert to DataFrame
# available:  'Twater','Ta_mean', 'Ta_min', 'Ta_max',
#             'SLP','runoff','snowfall','precip','cloud','windspeed',
#             'wind direction', 'FDD', 'TDD', 'SW', 'LH', 'SH',
#             'discharge','level St-L. River','level Ottawa River',
#             'NAO','PDO','ONI','AO','PNA','WP','TNH','SCAND','PT','POLEUR','EPNP','EA','Nino',
#             'Monthly forecast PRATE_SFC_0 - Start Sep. 1st','Monthly forecast TMP_TGL_2m - Start Sep. 1st','Monthly forecast PRMSL_MSL_0 - Start Sep. 1st',
#             'Monthly forecast PRATE_SFC_0 - Start Oct. 1st','Monthly forecast TMP_TGL_2m - Start Oct. 1st','Monthly forecast PRMSL_MSL_0 - Start Oct. 1st',
#             'Monthly forecast PRATE_SFC_0 - Start Nov. 1st','Monthly forecast TMP_TGL_2m - Start Nov. 1st','Monthly forecast PRMSL_MSL_0 - Start Nov. 1st',
#             'Monthly forecast PRATE_SFC_0 - Start Dec. 1st','Monthly forecast TMP_TGL_2m - Start Dec. 1st','Monthly forecast PRMSL_MSL_0 - Start Dec. 1st',
#             'Seasonal forecast PRATE_SFC_0 - Start Sep. 1st','Seasonal forecast TMP_TGL_2m - Start Sep. 1st','Seasonal forecast PRMSL_MSL_0 - Start Sep. 1st',
#             'Seasonal forecast PRATE_SFC_0 - Start Oct. 1st','Seasonal forecast TMP_TGL_2m - Start Oct. 1st','Seasonal forecast PRMSL_MSL_0 - Start Oct. 1st',
#             'Seasonal forecast PRATE_SFC_0 - Start Nov. 1st','Seasonal forecast TMP_TGL_2m - Start Nov. 1st','Seasonal forecast PRMSL_MSL_0 - Start Nov. 1st',
#             'Seasonal forecast PRATE_SFC_0 - Start Dec. 1st','Seasonal forecast TMP_TGL_2m - Start Dec. 1st','Seasonal forecast PRMSL_MSL_0 - Start Dec. 1st']

vars_out = []
# vars_out = ['Twater','Ta_mean','snowfall']

if len(vars_out) > 1:
    # Initialize output array and list of selected variables
    ds_out = np.zeros((ds.shape[0],len(vars_out)+1))
    var_list_out = []

    # First column is always time
    ds_out[:,0] = ds[:,0]
    var_list_out.append(labels[0])

    # Fill other columns with selected variables
    for k in range(len(vars_out)):
        idx = [i for i,v in enumerate(np.array(labels)) if vars_out[k] in v]

        if ('FDD' in vars_out[k])|('TDD' in vars_out[k]):
            ds_out[:,k+1] = np.squeeze(ds[:,idx[0]])
            ds_out[:,k+1][np.isnan(ds_out[:,k+1])] = 0
        else:
            ds_out[:,k+1] = np.squeeze(ds[:,idx[0]])

        var_list_out.append(labels[idx[0]])
else:
    ds_out = ds
    var_list_out = labels


year_start = np.where(ds_out[:,0] == (dt.date(1992,1,1)-date_ref).days)[0][0]
year_end = np.where(ds_out[:,0] == (dt.date(2021,12,31)-date_ref).days)[0][0]+1

df = pd.DataFrame(ds_out[year_start:year_end],columns=var_list_out)

#%%
# Check if Tw time serie is stationnary:
# A-D Fuller test:
from statsmodels.tsa.stattools import adfuller
from numpy import log
# result = adfuller(df['Avg. Twater'].dropna(how='all'))
result = adfuller(df['Avg. Twater'].fillna(0))
print('p-value: %9.8f' % result[1])

# KPSS test:
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
kpss_test(df['Avg. Twater'].fillna(0))

#Autocorrelation function:
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
plot_acf(df['Avg. Twater'],missing='drop',lags=365)
tacf = acf(df['Avg. Twater'],missing='drop')

fig,ax = plt.subplots()
y = df['Avg. Twater'].values
ax.hist(y)

#%%
# Try differencing:
Tdiff = df['Avg. Twater'][1:].fillna(0).values-df['Avg. Twater'][0:-1].fillna(0).values
result = adfuller(Tdiff)
print('p-value: %9.8f' % result[1])
kpss_test(Tdiff)
# plot_acf(Tdiff,missing='drop')
# tacf = acf(Tdiff,missing='drop')
# plt.figure()
# plt.plot(Tdiff)
fig,ax = plt.subplots()
# y = Tdiff
y = df['Avg. Twater'][1:].values-df['Avg. Twater'][0:-1].values
ax.hist(y,bins = np.arange(-3,3,0.2))

#%%
# Try second order differencing:
Tdiff2 = Tdiff[1:] - Tdiff[0:-1]
# plot_acf(Tdiff2,missing='drop')
# tacf = acf(Tdiff2,missing='drop')
# plt.figure();plt.plot(Tdiff2)
result = adfuller(Tdiff2)
print('p-value: %9.8f' % result[1])
kpss_test(Tdiff2)

fig,ax = plt.subplots()
# y_tmp = Tdiff
y_tmp = df['Avg. Twater'][1:].values-df['Avg. Twater'][0:-1].values
y = y_tmp[1:]-y_tmp[0:-1]
ax.hist(y,bins = np.arange(-3,3,0.2))

#%%
# Try seasonal differencing:
Tseasonaldiff = df['Avg. Twater'][366:].fillna(0).values-df['Avg. Twater'][0:-366].fillna(0).values
plot_acf(Tseasonaldiff,missing='drop')
tacf = acf(Tseasonaldiff,missing='drop')
plt.figure();plt.plot(Tseasonaldiff)
result = adfuller(Tseasonaldiff)
print('p-value: %9.8f' % result[1])
kpss_test(Tseasonaldiff)
fig,ax = plt.subplots()
# y = Tseasonaldiff
y = df['Avg. Twater'][366:].values-df['Avg. Twater'][0:-366].values
ax.hist(y,bins = np.arange(-3,3,0.2))

Tseasonaldiff2 = Tseasonaldiff[1:] - Tseasonaldiff[0:-1]
plot_acf(Tseasonaldiff2,missing='drop')
tacf = acf(Tseasonaldiff2,missing='drop')
plt.figure();plt.plot(Tseasonaldiff2)
result = adfuller(Tseasonaldiff2)
print('p-value: %9.8f' % result[1])
kpss_test(Tseasonaldiff2)
fig,ax = plt.subplots()
y_tmp = Tseasonaldiff
# y_tmp = df['Avg. Twater'][366:].values-df['Avg. Twater'][0:-366].values
y = y_tmp[1:]-y_tmp[0:-1]
ax.hist(y,bins = np.arange(-3,3,0.2))

#%%
# Try removing rolling climatology
#   *** NOTE: THIS IS EQUIVALENT TO (iv) Seasonal differencing ABOVE
nw = 1
years = np.arange(1992,2019)
Tw_clim_mean, Tw_clim_std, _ = rolling_climo(nw,df['Avg. Twater'].values,'all_time',df[df.columns[0]].values,years)
                             # rolling_climo(Nwindow,ts_in,array_type,time,years,date_ref = dt.date(1900,1,1)):
from functions import deseasonalize_ts
df_deseason = deseasonalize_ts(nw,df.values,df.columns,'all_time',df[df.columns[0]].values,years)
df_deseason = pd.DataFrame(df_deseason,columns=labels)

Tw_deseason = df_deseason['Avg. Twater'].values
#1
plot_acf(Tw_deseason,missing='drop')
tacf = acf(Tw_deseason,missing='drop')
#2
plt.figure()
plt.plot(Tw_deseason)
#3
fig,ax = plt.subplots()
# y = Tw_deseason.fillna(0)
y = df_deseason['Avg. Twater'].values
ax.hist(y,bins = np.arange(-3,3,0.2))

#%%
bias=0
for iyr in range(len(years)):
    # plot_acf(Tw_deseason[365*iyr+bias:365*(iyr+1)+bias],missing='drop',lags=180)
    plot_acf(Tw_deseason[365*(iyr)+bias:365*(iyr+1)+bias],missing='drop',lags=60)


decorrelation_time = np.array([21,14,9,15,11,6,14,16,20,13,7,13,16,16,9,8,12,13,13,15,12,11,15,15,18,14,20])

#%%
Tw_deseasondiff = Tw_deseason[1:]-Tw_deseason[0:-1]
plot_acf(Tw_deseasondiff,missing='drop',lags=600)
tacf = acf(Tw_deseasondiff,missing='drop')
plt.figure()
plt.plot(Tw_deseasondiff)
fig,ax = plt.subplots()
y = Tw_deseasondiff
ax.hist(y,bins = np.arange(-3,3,0.2))

Tw_deseasondiff2 = Tw_deseasondiff[1:] - Tw_deseasondiff[0:-1]
plot_acf(Tw_deseasondiff2,missing='drop',lags=600)
tacf = acf(Tw_deseasondiff2,missing='drop')
plt.figure()
plt.plot(Tw_deseasondiff2)
result = adfuller(Tw_deseasondiff2)
print('p-value: %9.8f' % result[1])
kpss_test(Tw_deseasondiff2)
fig,ax = plt.subplots()
y_tmp = Tw_deseasondiff
y = y_tmp[1:]-y_tmp[0:-1]
ax.hist(y,bins = np.arange(-3,3,0.2))

