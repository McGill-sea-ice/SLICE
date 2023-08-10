#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:59:57 2022

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
from sklearn.linear_model import Lasso

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

ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")


# valid_scheme = 'standard'
# valid_scheme = 'rolling'
valid_scheme = 'LOOk'
# valid_scheme = 'standardk'
# valid_scheme = 'rollingk'

start_yr = 1992
end_yr = 2019

train_yr_start = start_yr # [1992 - 2007] = 16 years
valid_yr_start = 2008 # [2008 - 2013] = 6 years
test_yr_start = 2014  # [2014 - 2019] = 6 years
nsplits = end_yr-valid_yr_start+1
nfolds = 5

freezeup_opt = 1


istart_show = [4]
max_pred = 3
# valid_metric = 'RMSE'
valid_metric = 'MAE'
# valid_metric = 'R2adj'
# valid_metric = 'Acc'

istart_labels = ['Sept. 1st', 'Oct. 1st','Nov. 1st','Dec. 1st','Jan. 1st']

#%%

# Load Twater and FUD data
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

# Load monthly predictor data
fpath_mp = local_path+'slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars'+'_'+'Longueuil'+'.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

# Replace zeros with nan for snowfall, FDD, TDD:
# monthly_pred_data[5,:,:][monthly_pred_data[5,:,:] == 0] = np.nan
# monthly_pred_data[10,:,:][monthly_pred_data[10,:,:] == 0] = np.nan
# monthly_pred_data[11,:,:][monthly_pred_data[11,:,:] == 0] = np.nan

# Remove data for December:
# monthly_pred_data[:,:,11] = np.nan

# Select specific predictors and make dataframe:
month_str = ['Jan. ', 'Feb. ', 'Mar. ', 'Apr. ', 'May ', 'Jun. ','Jul. ', 'Aug. ', 'Sep. ', 'Oct. ', 'Nov. ','Dec. ']

# Keep only 1991 to 2020 (inclusive)
it_start = np.where(years == start_yr)[0][0]
it_end = np.where(years == end_yr)[0][0]

years = years[it_start:it_end+1]
avg_freezeup_doy = avg_freezeup_doy[it_start:it_end+1]
monthly_pred_data = monthly_pred_data[:,it_start:it_end+1,:]

p_sep1_list = ['Avg. cloud cover','Avg. level Ottawa River','NAO','PDO','Avg. discharge St-L. River','Avg. Ta_mean', 'Avg. windspeed','Avg. SW down (sfc)', 'Avg. LH (sfc)','Avg. SH (sfc)','Avg. Twater', ]
m_sep1_list = [8,                 8,                        8,    8,    8,                           8,              8,               8,                   8,               8,              8,             ]

p_oct1_list = ['Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','Avg. level Ottawa River','NAO','NAO','PDO','PDO','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. Ta_mean','Avg. Ta_mean', 'Avg. windspeed', 'Avg. windspeed','Avg. SW down (sfc)','Avg. SW down (sfc)', 'Avg. LH (sfc)', 'Avg. LH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)', 'Avg. Twater', 'Avg. Twater'   ]
m_oct1_list = [8,                  9,                8,                        9,                        8,     9,    8,    9,    8,                           9,                           8,              9,           8,                9,               8,                   9 ,                  8,                9,              8,              9,              8,             9,            ]

p_nov1_list = ['Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','NAO','NAO','NAO','PDO','PDO','PDO','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Tot. snowfall', 'Avg. windspeed', 'Avg. windspeed','Avg. windspeed','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)', 'Avg. LH (sfc)', 'Avg. LH (sfc)','Avg. LH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)', ]
m_nov1_list = [8,                  9,                 10,                8,                        9,                        10,                      8,     9,    10,   8,    9,   10 ,   8,                           9,                           10,                         8,              9,            10,            10,             8,                9,                10,             8,                   9 ,                  10,                  8,                9,              10,             8,              9,              10,               ]

p_dec1_list = ['Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','NAO','NAO','NAO','NAO','PDO','PDO','PDO','PDO','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Tot. snowfall','Tot. snowfall','Avg. windspeed', 'Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)', 'Avg. LH (sfc)', 'Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)',  ]
m_dec1_list = [11,                 10,               9,                8,                   11,                       10,                      9,                        8,                        11,    10,     9,  8,    11,   10,    9, 8,     11,                           10,                         9,                           8,                            11,            10,           9,              8,            11,              10,            8,                9,                10,             11,              8,                   9 ,                  10,                  11,                    8,                9,              10,           11,              8,              9,              10,            11,                            ]

p_jan1_list = ['Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','NAO','NAO','NAO','NAO','NAO','PDO','PDO','PDO','PDO','PDO','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Tot. snowfall','Tot. snowfall','Tot. snowfall', 'Avg. windspeed', 'Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)', 'Avg. LH (sfc)', 'Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)' ]
m_jan1_list = [12,                11,                 10,               9,                8,                   12,                       11,                       10,                      9,                        8,                        12,   11,    10,     9,  8,    12,    11,   10,    9, 8,     12,                         11,                           10,                         9,                           8,                            12,           11,            10,           9,              8,            12,               11,              10,             8,                9,                10,             11,             12,              8,                   9 ,                  10,                  11,                  12,                   8,                9,              10,           11,              12,              8,              9,              10,            11,             12,             ]



pred_sep1_arr = np.zeros((monthly_pred_data.shape[1],len(p_sep1_list)))*np.nan
col_sep1 = []
for i in range(len(p_sep1_list)):
    ipred = np.where(pred_names == p_sep1_list[i])[0][0]
    pred_sep1_arr[:,i] = monthly_pred_data[ipred,:,m_sep1_list[i]-1]
    col_sep1.append(month_str[m_sep1_list[i]-1]+p_sep1_list[i])
pred_sep1_df =  pd.DataFrame(pred_sep1_arr,columns=col_sep1)

pred_oct1_arr = np.zeros((monthly_pred_data.shape[1],len(p_oct1_list)))*np.nan
col_oct1 = []
for i in range(len(p_oct1_list)):
    ipred = np.where(pred_names == p_oct1_list[i])[0][0]
    pred_oct1_arr[:,i] = monthly_pred_data[ipred,:,m_oct1_list[i]-1]
    col_oct1.append(month_str[m_oct1_list[i]-1]+p_oct1_list[i])
pred_oct1_df =  pd.DataFrame(pred_oct1_arr,columns=col_oct1)

pred_nov1_arr = np.zeros((monthly_pred_data.shape[1],len(p_nov1_list)))*np.nan
col_nov1 = []
for i in range(len(p_nov1_list)):
    ipred = np.where(pred_names == p_nov1_list[i])[0][0]
    pred_nov1_arr[:,i] = monthly_pred_data[ipred,:,m_nov1_list[i]-1]
    col_nov1.append(month_str[m_nov1_list[i]-1]+p_nov1_list[i])
pred_nov1_df =  pd.DataFrame(pred_nov1_arr,columns=col_nov1)

pred_dec1_arr = np.zeros((monthly_pred_data.shape[1],len(p_dec1_list)))*np.nan
col_dec1 = []
for i in range(len(p_dec1_list)):
    ipred = np.where(pred_names == p_dec1_list[i])[0][0]
    pred_dec1_arr[:,i] = monthly_pred_data[ipred,:,m_dec1_list[i]-1]
    col_dec1.append(month_str[m_dec1_list[i]-1]+p_dec1_list[i])
pred_dec1_df =  pd.DataFrame(pred_dec1_arr,columns=col_dec1)

pred_jan1_arr = np.zeros((monthly_pred_data.shape[1],len(p_jan1_list)))*np.nan
col_jan1 = []
for i in range(len(p_jan1_list)):
    ipred = np.where(pred_names == p_jan1_list[i])[0][0]
    pred_jan1_arr[:,i] = monthly_pred_data[ipred,:,m_jan1_list[i]-1]
    col_jan1.append(month_str[m_jan1_list[i]-1]+p_jan1_list[i])
pred_jan1_df =  pd.DataFrame(pred_jan1_arr,columns=col_jan1)

#%%

for ind,istart in enumerate(istart_show):
    # istart = 0 # forecast starting on 'Sept 1st'
    # istart = 1 # forecast starting on 'Oct. 1st'
    # istart = 2 # forecast starting on 'Nov. 1st'
    # istart = 3 # forecast starting on 'Dec. 1st'

    if istart == 0:
        col = col_sep1; pred_df = pred_sep1_df
    if istart == 1:
        col = col_oct1; pred_df = pred_oct1_df
    if istart == 2:
        col = col_nov1; pred_df = pred_nov1_df
    if istart == 3:
        col = col_dec1; pred_df = pred_dec1_df
    if istart == 4:
        col = col_jan1; pred_df = pred_jan1_df


#%%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# pred_df_clean, dropped_cols = remove_collinear_features(pred_df, avg_freezeup_doy, threshold=0.8, target_in_df = False, verbose = True)
# print("dropped columns: ")
# print(list(dropped_cols))
pred_df_clean = pred_df

model = Lasso(alpha=1.0)
model_type = 'lasso'
X, y = pred_df_clean, pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])

model = LinearRegression()
model_type = 'mlr'
pred = ['Oct. Avg. cloud cover','Sep. NAO','Sep. Avg. SH (sfc)']
# pred = ['Oct. Avg. cloud cover','Sep. Avg. SW down (sfc)','Dec. Avg. Ta_mean']
# pred = ['Oct. Avg. cloud cover','Sep. NAO','Sep. Avg. SH (sfc)','Nov. Avg. Ta_mean','Dec. Avg. Ta_mean']
pred = ['Oct. Avg. cloud cover','Sep. NAO','Sep. Avg. SH (sfc)','Nov. Avg. Ta_mean','Dec. Avg. Ta_mean','Nov. Tot. snowfall','Dec. Tot. snowfall']
pred = ['Dec. Avg. Ta_mean','Nov. Tot. snowfall','Dec. Tot. snowfall']
# pred = ['Dec. Avg. Ta_mean']
# pred = ['Oct. Avg. cloud cover','Sep. NAO','Sep. Avg. SH (sfc)','Dec. Avg. Ta_mean','Dec. Tot. snowfall']
# pred = ['Oct. Avg. cloud cover','Sep. NAO','Sep. Avg. SH (sfc)','Dec. Tot. snowfall']
# pred = ['Oct. Avg. cloud cover','Sep. NAO','Sep. Avg. SH (sfc)','Dec. Avg. Ta_mean']
# pred = ['Oct. Avg. cloud cover','Sep. NAO','Sep. Avg. SH (sfc)','Nov. Avg. Ta_mean','Nov. Tot. snowfall']
X, y = pred_df[pred], pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])

loo = LeaveOneOut()
kf = KFold()

y_pred_test = np.zeros(y.shape)
y_plot = np.zeros(y.shape)


fig_c,ax_c = plt.subplots(nrows = int(len(years)/4), ncols = 4,sharex=True,sharey=True)
list_coeff = []
list_coeff_topN = []
N = 6
beta = 0
for i,[train_index, test_index] in enumerate(loo.split(X)):

    print('------------')
    print('TEST YEAR = ', years[i])
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Normalize predictors:
    # Xscaler = MinMaxScaler()
    Xscaler = StandardScaler()
    Xscaler = Xscaler.fit(X_train)
    X_train_scaled = Xscaler.transform(X_train)
    X_test_scaled = Xscaler.transform(X_test)

    # yscaler = MinMaxScaler()
    yscaler = StandardScaler()
    yscaler = yscaler.fit(y_train)
    y_train_scaled = yscaler.transform(y_train)
    y_test_scaled = yscaler.transform(y_test)

    # cv = KFold()
    # scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # scores = np.absolute(scores)
    # print(np.nanmean(scores), np.nanmin(scores))
    # print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    # print(scores)

    # Get Prediction:
    # y_pred_test[i] = model.fit(X_train,y_train).predict(X_test)
    # y_plot[i] = y_test
    # y_pred_test[i] = yscaler.inverse_transform(model.fit(X_train_scaled,y_train_scaled).predict(X_test_scaled).reshape(1,-1))
    # y_plot[i] = yscaler.inverse_transform(y_test_scaled)
    y_pred_test[i] = model.fit(X_train_scaled,y_train).predict(X_test_scaled)
    y_plot[i] = y_test

    # test_yr_coef = {}
    # for ic,c in enumerate(model.coef_):
    #     if c != 0:
    #         test_yr_coef[col[ic]] = c

    if not np.all(model.coef_ == 0):
        print('Reg. coeff.: ', model.coef_[model.coef_ != 0])
        print('Stand. coeff.: ', model.coef_[model.coef_ != 0]*(np.nanstd(X_train_scaled,axis=0)/np.nanstd(y_train)))
        beta += model.coef_[model.coef_ != 0]*(np.nanstd(X_train_scaled,axis=0)/np.nanstd(y_train))
        if model_type == 'lasso': 
            x = np.where(model.coef_)[0]
        else:
            x = np.arange(len(pred))
            
        if i/len(years) < 0.25:
            ax_c[i,0].stem(
            x,
            model.coef_[model.coef_ != 0],
            markerfmt="x")
        elif i/len(years) < 0.5:
            ax_c[i-int(len(years)/4),1].stem(
            x,
            model.coef_[model.coef_ != 0],
            markerfmt="x")
        elif i/len(years) < 0.75:
            ax_c[i-2*int(len(years)/4),2].stem(
            x,
            model.coef_[model.coef_ != 0],
            markerfmt="x")
        else:
            ax_c[i-3*int(len(years)/4),3].stem(
            x,
            model.coef_[model.coef_ != 0],
            markerfmt="x")

    list_coeff+=(np.array(np.where(model.coef_)[0][:],dtype='int')[:].tolist())
    if model_type == 'lasso': list_coeff_topN += [np.where(np.abs(model.coef_) == np.sort(np.abs(model.coef_[model.coef_ != 0]))[i])[0][0] for i in range(-N,0)]
    # plt.figure()
    # plt.title(years[i])
    # years_tmp = years.copy()
    # # plt.plot(np.delete(years_tmp,i),model.predict(X_train),'o-')
    # # plt.plot(np.delete(years_tmp,i),y_train,'o-',color='k')
    # # plt.plot(np.delete(years_tmp,i),yscaler.inverse_transform(model.predict(X_train).reshape(-1,1)),'o-')
    # # plt.plot(np.delete(years_tmp,i),yscaler.inverse_transform(y_train_scaled),'o-',color='k')
    # plt.plot(np.delete(years_tmp,i),model.predict(X_train_scaled),'o-')
    # plt.plot(np.delete(years_tmp,i),y_train,'o-',color='k')
    if model_type == 'lasso': print(pred_df_clean.columns[[np.where(np.abs(model.coef_) == np.sort(np.abs(model.coef_[model.coef_ != 0]))[i])[0][0] for i in range(-N,0)]])


print(beta/len(y_plot))
print(pred_df_clean.columns[[np.unique(np.array(list_coeff))]])

if model_type == 'lasso': print(pred_df_clean.columns[[np.unique(np.array(list_coeff_topN))]])


plt.figure()
plt.plot(years,y_plot,'-o',color='k')
plt.plot(years,y_pred_test,'-o',color=plt.get_cmap('tab20c')(0))
print(np.nanmean(np.abs(y[:]-y_pred_test[:])))
print(np.sum(np.abs(y[:]-y_pred_test[:]) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy))))


# GET COEFFICIENTS P-VALUES
from sklearn.model_selection import KFold
targets = y.values
test_years = years

kf = KFold(n_splits=nfolds)

# 1.
# For all validation and test periods,
# and for all possible combinations of predictors:
# Get FUD predictions, target FUD category, model f-pvalue and model's coefficients pvalues
FUD_cat_valid = np.zeros((len(years)))*np.nan
clim_test = np.zeros((len(years)))*np.nan
pred_test = np.zeros((1,len(years)))*np.nan
f_pvalue_train = np.zeros((1,len(years)))*np.nan
coeff_pvalues_train = np.zeros((1,len(years),len(pred)))*np.nan

for iyr_test,yr_test in enumerate(years):

    mask_test = np.ones(pred_df_clean.shape[0], dtype=bool)
    mask_test[iyr_test] = False

    # First, get forecast using all years other than test year for training
    target_train = targets[mask_test]
    target_test = targets[~mask_test]
    df_train = pred_df_clean[mask_test]
    df_test = pred_df_clean[~mask_test]

    clim_test[iyr_test] = np.nanmean(target_train)

    i = 0
    x_model = pred

    pred_train_select = df_train[x_model]
    pred_test_select = df_test[x_model]

    mlr_model_train = sm.OLS(target_train, sm.add_constant(pred_train_select,has_constant='skip'), missing='drop').fit()

    train_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_train_select,has_constant='skip'))
    test_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_test_select,has_constant='add'))
    pred_test[i,iyr_test] = (test_predictions_FUD)
    f_pvalue_train[i,iyr_test]= (mlr_model_train.f_pvalue)
    coeff_pvalues_train[i,iyr_test,0:len(x_model)]= (mlr_model_train.pvalues[1:].values)

print(np.all(coeff_pvalues_train < 0.05))

#%%

tmp = sm.OLS(X_train_scaled[:,0], sm.add_constant(X_train_scaled[:,2],has_constant='skip'), missing='drop').fit()
# tmp = sm.OLS(y_train, sm.add_constant(X_train_scaled[:,0],has_constant='skip'), missing='drop').fit()
print(tmp.rsquared)


#%%
# import numpy as np
# from sklearn.model_selection import LeaveOneOut
# from sklearn.model_selection import KFold
# from sklearn.linear_model import Lasso

# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]])
# y = np.array([1, 2,3,4,5,6,7,8,9,10,11])

# loo = LeaveOneOut()
# loo.get_n_splits(X)

# kf = KFold()

# for train_index, test_index in loo.split(X):
#      print("TRAIN:", train_index, "TEST:", test_index)
#      X_train, X_test = X[train_index], X[test_index]
#      y_train, y_test = y[train_index], y[test_index]
#      print(X_train, X_test, y_train, y_test)

#      for k_train_index, k_valid_index in kf.split(X_train):
#           print("TRAIN - FOLD:", k_train_index, "VALID - FOLD:", k_valid_index)
#           X_train_fold, X_valid_fold = X_train[k_train_index], X_train[k_valid_index]
#           y_train_fold, y_valid_fold = y_train[k_train_index], y_train[k_valid_index]
#           print(X_train_fold, X_valid_fold, y_train_fold, y_valid_fold)

#      print('=======================================')

#%%
fig,ax = plt.subplots(nrows = 2, sharex=True)
pred_name_list = ['Nov. Avg. Ta_mean',
                  'Nov. Tot. snowfall'
                  ]
years_shade_later = [1994,2001,2014,2015]
years_shade_earlier = [2002,2003,2005,2007]

y=pred_df['Nov. Avg. Ta_mean']+pred_df['Dec. Avg. Ta_mean']
ax[0].plot(years,np.ones(len(years))*np.nanmean(y),'-', color=plt.get_cmap('tab20c')(1),linewidth=0.5)
ax[0].fill_between(years,np.ones(len(years))*(np.nanmean(y)-np.nanstd(y)),np.ones(len(years))*(np.nanmean(y)+np.nanstd(y)), color=plt.get_cmap('tab20c')(1),alpha=0.3)

ax[0].plot(years,y,'.-',label='Nov. Avg. Ta_mean'+' + ' 'Dec. Avg. Ta_mean')
ax[0].legend()
ax[0].grid(True)
# ax[i].set_ylabel(pred_name)
for ys in years_shade_later:
    ymin = np.ones(2)*y.min()
    ymax = np.ones(2)*y.max()
    ax[0].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='pink',alpha=0.2)

for ys in years_shade_earlier:
    ymin = np.ones(2)*y.min()
    ymax = np.ones(2)*y.max()
    ax[0].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='cyan',alpha=0.2)


y=pred_df['Nov. Tot. snowfall']+pred_df['Dec. Tot. snowfall']
ax[1].plot(years,np.ones(len(years))*np.nanmean(y),'-', color=plt.get_cmap('tab20c')(1),linewidth=0.5)
ax[1].fill_between(years,np.ones(len(years))*(np.nanmean(y)-np.nanstd(y)),np.ones(len(years))*(np.nanmean(y)+np.nanstd(y)), color=plt.get_cmap('tab20c')(1),alpha=0.3)

ax[1].plot(years,y,'.-',label='Nov. Tot. snowfall'+' + ' 'Dec. Tot. snowfall')
ax[1].legend()
ax[1].grid(True)
for ys in years_shade_later:
    ymin = np.ones(2)*y.min()
    ymax = np.ones(2)*y.max()
    ax[1].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='pink',alpha=0.2)

for ys in years_shade_earlier:
    ymin = np.ones(2)*y.min()
    ymax = np.ones(2)*y.max()
    ax[1].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='cyan',alpha=0.2)


#%%


pred_name_list = [
                   'Sep. Avg. level Ottawa River',
                   'Oct. Avg. level Ottawa River',
                  ]
fig,ax = plt.subplots(nrows = len(pred_name_list), sharex=True)
years_shade_later = [1994,2001,2014,2015]
years_shade_earlier = [2002,2003,2005,2007]
for i, pred_name in enumerate(pred_name_list):
    ax[i].plot(years,np.ones(len(years))*np.nanmean(pred_df[pred_name]),'-', color=plt.get_cmap('tab20c')(1),linewidth=0.5)
    ax[i].fill_between(years,np.ones(len(years))*(np.nanmean(pred_df[pred_name])-np.nanstd(pred_df[pred_name])),np.ones(len(years))*(np.nanmean(pred_df[pred_name])+np.nanstd(pred_df[pred_name])), color=plt.get_cmap('tab20c')(1),alpha=0.3)

    ax[i].plot(years,pred_df[pred_name],'.-',label=pred_name)
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].set_ylabel(pred_name)
    for ys in years_shade_later:
        ymin = np.ones(2)*pred_df[pred_name].min()
        ymax = np.ones(2)*pred_df[pred_name].max()
        ax[i].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='pink',alpha=0.2)

    for ys in years_shade_earlier:
        ymin = np.ones(2)*pred_df[pred_name].min()
        ymax = np.ones(2)*pred_df[pred_name].max()
        ax[i].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='cyan',alpha=0.2)


for ys in years_shade_later:
    ymin = np.ones(2)*pred_df[pred_name].min()
    ymax = np.ones(2)*pred_df[pred_name].max()
    ax[-1].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='pink',alpha=0.2)

for ys in years_shade_earlier:
    ymin = np.ones(2)*pred_df[pred_name].min()
    ymax = np.ones(2)*pred_df[pred_name].max()
    ax[-1].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='cyan',alpha=0.2)



#%%

pred_name_list = [
                    # 'Sep. Avg. Ta_mean',
                    'Nov. NAO',
                    'Dec. NAO',
                    # 'Dec. Avg. Ta_mean',
                    'Sep. Avg. SW down (sfc)',
                   
                   
                  # 'Sep. Avg. discharge St-L. River',
                  # 'Oct. Avg. discharge St-L. River',
                  # 'Nov. Avg. discharge St-L. River',
                  # 'Sep. Avg. LH (sfc)',
                  # 'Oct. Avg. LH (sfc)',
                  # 'Oct. Avg. windspeed',
                  # 'Oct. Tot. snowfall',
                  # 'Sep. Avg. Twater',
                  # 'Oct. Avg. Twater',
                  # 'Dec. Avg. Twater' 
                  # 'Dec. Avg. SW down (sfc)',
                   # 'Dec. Avg. SH (sfc)',
                   # 'Dec. Avg. windspeed',
                  
                  ]
fig,ax = plt.subplots(nrows = len(pred_name_list), sharex=True)
years_shade_later = [1994,2001,2014,2015]
years_shade_earlier = [2002,2003,2005,2007]
for i, pred_name in enumerate(pred_name_list):
    ax[i].plot(years,np.ones(len(years))*np.nanmean(pred_df[pred_name]),'-', color=plt.get_cmap('tab20c')(1),linewidth=0.5)
    ax[i].fill_between(years,np.ones(len(years))*(np.nanmean(pred_df[pred_name])-np.nanstd(pred_df[pred_name])),np.ones(len(years))*(np.nanmean(pred_df[pred_name])+np.nanstd(pred_df[pred_name])), color=plt.get_cmap('tab20c')(1),alpha=0.3)

    ax[i].plot(years,pred_df[pred_name],'.-',label=pred_name)
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].set_ylabel(pred_name)
    for ys in years_shade_later:
        ymin = np.ones(2)*pred_df[pred_name].min()
        ymax = np.ones(2)*pred_df[pred_name].max()
        ax[i].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='pink',alpha=0.2)

    for ys in years_shade_earlier:
        ymin = np.ones(2)*pred_df[pred_name].min()
        ymax = np.ones(2)*pred_df[pred_name].max()
        ax[i].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='cyan',alpha=0.2)


for ys in years_shade_later:
    ymin = np.ones(2)*pred_df[pred_name].min()
    ymax = np.ones(2)*pred_df[pred_name].max()
    ax[-1].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='pink',alpha=0.2)

for ys in years_shade_earlier:
    ymin = np.ones(2)*pred_df[pred_name].min()
    ymax = np.ones(2)*pred_df[pred_name].max()
    ax[-1].fill_between([ys-0.4,ys+0.4],ymin,ymax,color='cyan',alpha=0.2)

