#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:24:02 2022

@author: Amelie
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import datetime as dt
from functions import detect_FUD_from_Tw

#%%

fdir_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CMC_GHRSST/'
fdir_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CMC_GHRSST/'

verbose = False
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

ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")


train_yr_start = 1992 # [1992 - 2007] = 16 years
valid_yr_start = 2008 # [2008 - 2013] = 6 years
test_yr_start = 2014  # [2014 - 2019] = 6 years
nsplits = 2019-valid_yr_start+1

# istart_show = [2]
istart_show = [2,3]
max_pred = 4
# valid_metric = 'RMSE'
valid_metric = 'MAE'
# valid_metric = 'R2adj'
# valid_metric = 'Acc'

istart_labels = ['Sept. 1st', 'Oct. 1st','Nov. 1st','Dec. 1st']

#%%
# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
freezeup_doy[np.where(years == 2020)] = np.nan

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)

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
fpath_mp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

# Replace zeros with nan for snowfall, FDD, TDD:
# monthly_pred_data[5,:,:][monthly_pred_data[5,:,:] == 0] = np.nan
# monthly_pred_data[10,:,:][monthly_pred_data[10,:,:] == 0] = np.nan
# monthly_pred_data[11,:,:][monthly_pred_data[11,:,:] == 0] = np.nan

# Remove data for December:
monthly_pred_data[:,:,11] = np.nan

# Select specific predictors and make dataframe:
month_str = ['Jan. ', 'Feb. ', 'Mar. ', 'Apr. ', 'May ', 'Jun. ','Jul. ', 'Aug. ', 'Sep. ', 'Oct. ', 'Nov. ']

# PREDICTORS AVAILABLE SEPT 1st: FUD + Tw
# p_sep1_list = ['Avg. Ta_mean','Tot. TDD','Tot. TDD','Tot. snowfall','Avg. cloud cover','PDO','PDO','Avg. Twater']
# m_sep1_list = [5,             1,          5,         4,              4,                 1,    2,    5           ]
# PREDICTORS AVAILABLE SEPT 1st: FUD ONLY
p_sep1_list = ['Avg. Ta_mean','Tot. TDD','PDO','PDO']
m_sep1_list = [5,             5,          1,    2]
pred_sep1_arr = np.zeros((monthly_pred_data.shape[1],len(p_sep1_list)))*np.nan
col_sep1 = []
for i in range(len(p_sep1_list)):
    ipred = np.where(pred_names == p_sep1_list[i])[0][0]
    pred_sep1_arr[:,i] = monthly_pred_data[ipred,:,m_sep1_list[i]-1]
    col_sep1.append(month_str[m_sep1_list[i]-1]+p_sep1_list[i])
pred_sep1_df =  pd.DataFrame(pred_sep1_arr,columns=col_sep1)

# PREDICTORS AVAILABLE OCT 1st: FUD + Tw
# p_oct1_list = ['Avg. Ta_mean','Tot. TDD','Tot. TDD','Tot. snowfall','Avg. cloud cover','Avg. cloud cover','NAO','PDO','PDO','Avg. Twater']
# m_oct1_list = [5,             1,          5,         4,              4,                9,                  9,    1,    2,    5           ]
# PREDICTORS AVAILABLE OCT 1st: FUD ONLY
p_oct1_list = ['Avg. Ta_mean','Tot. TDD','Avg. cloud cover','NAO','PDO','PDO']
m_oct1_list = [5,             5,          9,                 9,    1,    2]
pred_oct1_arr = np.zeros((monthly_pred_data.shape[1],len(p_oct1_list)))*np.nan
col_oct1 = []
for i in range(len(p_oct1_list)):
    ipred = np.where(pred_names == p_oct1_list[i])[0][0]
    pred_oct1_arr[:,i] = monthly_pred_data[ipred,:,m_oct1_list[i]-1]
    col_oct1.append(month_str[m_oct1_list[i]-1]+p_oct1_list[i])
pred_oct1_df =  pd.DataFrame(pred_oct1_arr,columns=col_oct1)

# PREDICTORS AVAILABLE NOV 1st: FUD + Tw
# p_nov1_list = ['Avg. Ta_mean','Tot. TDD','Tot. TDD','Tot. snowfall','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','NAO','PDO','PDO','Avg. Twater','Avg. windspeed']
# m_nov1_list = [5,             1,          5,        4,              4,                 9,                 10,                        9,    1,    2,    5,         10              ]
# PREDICTORS AVAILABLE NOV 1st: FUD ONLY
p_nov1_list = ['Avg. Ta_mean','Tot. TDD','Avg. cloud cover','Avg. level Ottawa River','NAO','PDO','PDO','Avg. windspeed'  ]
m_nov1_list = [5,             5,          9,                 10,                       9,    1,    2,   10               ]
pred_nov1_arr = np.zeros((monthly_pred_data.shape[1],len(p_nov1_list)))*np.nan
col_nov1 = []
for i in range(len(p_nov1_list)):
    ipred = np.where(pred_names == p_nov1_list[i])[0][0]
    pred_nov1_arr[:,i] = monthly_pred_data[ipred,:,m_nov1_list[i]-1]
    col_nov1.append(month_str[m_nov1_list[i]-1]+p_nov1_list[i])
pred_nov1_df =  pd.DataFrame(pred_nov1_arr,columns=col_nov1)

# PREDICTORS AVAILABLE DEC 1st: FUD + Tw
# p_dec1_list = ['Avg. Ta_mean','Avg. Ta_mean','Tot. FDD','Tot. TDD','Tot. TDD','Tot. precip.','Avg. SLP','Tot. snowfall','Tot. snowfall','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','AO','NAO','NAO','PDO','PDO','Avg. Twater','Avg. Twater','Avg. windspeed'  ]
# m_dec1_list = [11,            5,             11,        1,          5,        11,            11,         4,              11,             4,                 9,                 11,                10,                       11,   9,    11,   1,    2,    5,            11         ,10                ]
# PREDICTORS AVAILABLE DEC 1st: FUD ONLY
p_dec1_list = ['Avg. Ta_mean','Avg. Ta_mean','Tot. FDD','Tot. TDD','Tot. TDD','Avg. SLP','Tot. snowfall','Avg. cloud cover','Avg. level Ottawa River','AO','NAO','NAO','PDO','PDO','Avg. windspeed'  ]
m_dec1_list = [11,            5,             11,         5,        11,         11,        11,             9,                 10,                       11,   9,    11,   1,    2,   10               ]
pred_dec1_arr = np.zeros((monthly_pred_data.shape[1],len(p_dec1_list)))*np.nan
col_dec1 = []
for i in range(len(p_dec1_list)):
    ipred = np.where(pred_names == p_dec1_list[i])[0][0]
    pred_dec1_arr[:,i] = monthly_pred_data[ipred,:,m_dec1_list[i]-1]
    col_dec1.append(month_str[m_dec1_list[i]-1]+p_dec1_list[i])
pred_dec1_df =  pd.DataFrame(pred_dec1_arr,columns=col_dec1)


#%%
# Quick test only for Dec. 1st Forecast
# valid_scheme == 'standard':
col = col_dec1
pred_df = pred_dec1_df
# col = col_nov1
# pred_df = pred_nov1_df
# col = col_oct1
# pred_df = pred_oct1_df
# col = col_sep1
# pred_df = pred_sep1_df

it_train_start = np.where(years == train_yr_start)[0][0]
it_train_end = np.where(years == valid_yr_start)[0][0]

it_valid_start = np.where(years == valid_yr_start)[0][0]
it_valid_end = np.where(years == test_yr_start)[0][0]

it_test_start = np.where(years == test_yr_start)[0][0]
it_test_end = np.where(years == 2020)[0][0]

df_train = pred_df[it_train_start:it_train_end].copy()
df_valid = pred_df[it_valid_start:it_valid_end].copy()
df_test = pred_df[it_test_start:it_test_end].copy()
train_years = years[it_train_start:it_train_end]
valid_years = years[it_valid_start:it_valid_end]
test_years = years[it_test_start:it_test_end]

target_train = avg_freezeup_doy[it_train_start:it_train_end].copy()
target_valid = avg_freezeup_doy[it_valid_start:it_valid_end].copy()
target_test = avg_freezeup_doy[it_test_start:it_test_end].copy()

X = df_train.to_numpy()
X_valid = df_valid.to_numpy()
X_test = df_test.to_numpy()
y =  target_train

svr_rbf = SVR(kernel="rbf", C=100, gamma="scale", epsilon=4)
svr_lin = SVR(kernel="linear", C=100, gamma="scale",epsilon=6)
svr_poly = SVR(kernel="poly", C=100, gamma="scale", degree=3, epsilon=1, coef0=10)

lw = 2
svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ["RBF", "Linear", "Polynomial"]
model_color = ["m", "c", "g"]


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7, 5), sharey=True)
for ix, svr in enumerate(svrs):
    # Plot Training
    axes[0,ix].plot(
        train_years,
        target_train,
        'o',
        color='k',
        label="FUD target",
    )
    axes[0,ix].plot(
        train_years,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    axes[0,ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

    # Plot validation forecasts
    axes[1,ix].plot(
        valid_years,
        target_valid,
        'o',
        color='k',
        label="FUD target",
    )
    axes[1,ix].plot(
        valid_years,
        svr.predict(X_valid),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    axes[1,ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

    # # # Plot test forecasts
    # axes[2,ix].plot(
    #     test_years,
    #     target_test,
    #     'o',
    #     color='k',
    #     label="FUD target",
    # )
    # axes[2,ix].plot(
    #     test_years,
    #     svr.predict(X_test),
    #     color=model_color[ix],
    #     lw=lw,
    #     label="{} model".format(kernel_label[ix]),
    # )
    # axes[2,ix].legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.1),
    #     ncol=1,
    #     fancybox=True,
    #     shadow=True,
    # )
fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
# fig.suptitle("Support Vector Regression - TRAINING ", fontsize=14)





