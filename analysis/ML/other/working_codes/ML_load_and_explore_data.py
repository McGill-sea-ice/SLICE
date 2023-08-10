#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:25:00 2022

@author: Amelie
"""

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

import tensorflow as tf
import sklearn.metrics as metrics

from functions import rolling_climo
from functions_ML import regression_metrics

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

#%%


#%%
plot = False

#%%
# Load Data
fpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/ML_timeseries/'
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

# vars_out = ['Twater','Ta_mean','snowfall','discharge','NAO']
vars_out = ['Twater','Ta_mean']

if len(vars_out) > 0:
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
# Data preparation: differentiate Twater
Tdiff = df['Avg. Twater'][1:].values-df['Avg. Twater'][0:-1].values

# Check that distribution is more or less Gaussian:
if plot:
    fig,ax = plt.subplots()
    ax.hist(Tdiff,bins = np.arange(-3,3,0.2))

# Add Twater differences to dataset:
df.insert(1, 'Twdiff', np.resize(Tdiff, Tdiff.shape[0]+1))

#%%
# Add DOY feature with a sin+cos
year = 365.2425
doysin = np.zeros((len(df)))*np.nan
doycos = np.zeros((len(df)))*np.nan
for it,t in enumerate(df[df.columns[0]]):
    date = date_ref+dt.timedelta(days=t)
    doy = (dt.date(date.year,date.month,date.day)-dt.date(date.year,1,1)).days +1
    doysin[it] = np.sin(doy * (2*np.pi/year))
    doycos[it] = np.cos(doy * (2*np.pi/year))

df.insert(2, 'sin(DOY)', doysin)
df.insert(3, 'cos(DOY)', doycos)

#%%
# Check how the distribution is similar between training, valid, and test sets:
split_valid = 0.6
split_test = 0.8
split_valid_it = int(np.round(split_valid*df.shape[0]))
split_test_it = int(np.round(split_test*df.shape[0]))

train_dataset = df[:split_valid_it]
valid_dataset = df[split_valid_it:split_test_it]
test_dataset = df[split_test_it:]

if plot:
    fig,ax = plt.subplots()
    ax.hist(train_dataset['Twdiff'],bins = np.arange(-3,3,0.2),density=True,alpha = 0.3)
    ax.hist(valid_dataset['Twdiff'],bins = np.arange(-3,3,0.2),density=True,alpha = 0.3)
    ax.hist(test_dataset['Twdiff'],bins = np.arange(-3,3,0.2),density=True,alpha = 0.3)

#%%
def backwardshift(table, max_lag, min_lag=0, separator='_'):
    """ Given a dataframe, return a dataframe with different lags of all its columns """
    values=[]
    for i in range(min_lag, max_lag + 1):
        values.append(table.shift(i).copy())
        values[-1].columns = [c + separator + str(i) for c in table.columns]
    return pd.concat(values, axis=1)

def forwardshift(table, max_lag, min_lag=0, separator='_'):
    """ Given a dataframe, return a dataframe with different increments of all its columns """
    values=[]
    for i in range(min_lag, max_lag + 1):
        values.append(table.shift(-i).copy())
        values[-1].columns = [c + separator + str(i) for c in table.columns]
    return pd.concat(values, axis=1)

input_width = 75
pred_width = 75

# Make windows of past data with predictors
# df_Xtrain = backwardshift(train_dataset.drop(['Avg. Twater',df.columns[0]],1),input_width,0)
# df_Xvalid = backwardshift(valid_dataset.drop(['Avg. Twater',df.columns[0]],1),input_width,0)
# df_Xtest = backwardshift(test_dataset.drop(['Avg. Twater',df.columns[0]],1),input_width,0)
df_Xtrain = backwardshift(train_dataset.drop([df.columns[0]],1),input_width,0)
df_Xvalid = backwardshift(valid_dataset.drop([df.columns[0]],1),input_width,0)
df_Xtest = backwardshift(test_dataset.drop([df.columns[0]],1),input_width,0)


# Make window of future data for Twater diff.
# df_Ytrain = forwardshift(pd.DataFrame(train_dataset['Twdiff']),pred_width,1)
# df_Yvalid = forwardshift(pd.DataFrame(valid_dataset['Twdiff']),pred_width,1)
# df_Ytest = forwardshift(pd.DataFrame(test_dataset['Twdiff']),pred_width,1)
df_Ytrain = forwardshift(pd.DataFrame(train_dataset['Avg. Twater']),pred_width,1)
df_Yvalid = forwardshift(pd.DataFrame(valid_dataset['Avg. Twater']),pred_width,1)
df_Ytest = forwardshift(pd.DataFrame(test_dataset['Avg. Twater']),pred_width,1)

Tw_train = forwardshift(pd.DataFrame(train_dataset['Avg. Twater']),pred_width,1)
Tw_valid = forwardshift(pd.DataFrame(valid_dataset['Avg. Twater']),pred_width,1)
Tw_test = forwardshift(pd.DataFrame(test_dataset['Avg. Twater']),pred_width,1)


# Remove samples with Nans
df_Xtrain.dropna(inplace=True);df_Ytrain.dropna(inplace=True);
id_train = df_Xtrain.index.intersection(df_Ytrain.index)
df_Xtrain = df_Xtrain.loc[id_train];df_Ytrain = df_Ytrain.loc[id_train]
Tw_train = Tw_train.loc[id_train]

df_Xvalid.dropna(inplace=True);df_Yvalid.dropna(inplace=True);
id_valid = df_Xvalid.index.intersection(df_Yvalid.index)
df_Xvalid = df_Xvalid.loc[id_valid];df_Yvalid = df_Yvalid.loc[id_valid]
Tw_valid = Tw_valid.loc[id_valid]

df_Xtest.dropna(inplace=True);df_Ytest.dropna(inplace=True);
id_test = df_Xtest.index.intersection(df_Ytest.index)
df_Xtest = df_Xtest.loc[id_test];df_Ytest = df_Ytest.loc[id_test]
Tw_test = Tw_test.loc[id_test]

#%%
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestRegressor
# rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=1), 4)
# fit = rfe.fit(finaldf_train_x, finaldf_train_y)
# y_pred = fit.predict(finaldf_test_x)


#%%
##apply random forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=9,
                            min_samples_split=0.0005, min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0, max_features='sqrt',
                            max_leaf_nodes=None, min_impurity_decrease=0.0,
                            min_impurity_split=None, bootstrap=True,
                            oob_score=False, n_jobs= -1, random_state=1,
                            verbose=0, warm_start=False, ccp_alpha=0.0,
                            max_samples=None)

# rf = RandomForestRegressor(n_estimators=10)
rf.fit(df_Xtrain, df_Ytrain)
yhat = rf.predict(df_Xtest)
print('random forest train score:', rf.score(df_Xtrain, df_Ytrain))
print('random forest test score:', rf.score(df_Xtest, df_Ytest))

import sklearn.metrics as skm

# print('random forest train score:', skm.r2_score(df_Xtrain, df_Ytrain))
print('random forest test Rsquared:', skm.r2_score(df_Ytest, yhat))
print('random forest test MAE (raw):', skm.mean_absolute_error(df_Ytest, yhat))

#%%
Tw_recons = np.zeros((Tw_test.shape[0],pred_width))*np.nan
for it in range(Tw_test.shape[0]):
    for tt in range(pred_width):
        if tt > 0 :
            Tw_recons[it,tt] = Tw_recons[it,tt-1] + yhat[it,tt]
            # Tw_recons[tt] = Tw_test.iloc[it,tt-1] + yhat[it,tt]
        else:
            Tw_recons[it,tt] =  Tw_test.iloc[it-1].values[0] + yhat[it,tt]

# print('random forest test Rsquared(recons):', skm.r2_score(Tw_test, Tw_recons, multioutput='raw_values'))
# print('random forest test MAE (recons):', skm.mean_absolute_error(Tw_test, Tw_recons, multioutput='raw_values'))

print('random forest test Rsquared(recons):', skm.r2_score(Tw_test, Tw_recons))
print('random forest test MAE (recons):', skm.mean_absolute_error(Tw_test, Tw_recons))

#%%
test_rsquare_recons=0
for it in range(Tw_test.shape[0]):
    test_rsquare_recons = test_rsquare_recons + skm.r2_score(Tw_test.iloc[it], Tw_recons[it])
    print(it,test_rsquare_recons)
    # if test_rsquare_recons < 0:
    #     print('PROB!!', it)
test_rsquare_recons /= Tw_test.shape[0]
print('random forest test Rsquared(recons):', test_rsquare_recons )

# random forest train score: 0.1489965032847034
# random forest test score: 0.1237499630685260

#%%
plt.figure()
for it in range(360):

    plt.plot(it,df_Ytest.iloc[it,5],'o')
    plt.plot(it,yhat[it,5],'s')


#%%
it = 890
plt.figure()
plt.plot(df_Ytest.iloc[it].values)
plt.plot(yhat[it])

# plt.figure()
# plt.plot(Tw_test.iloc[it].values)
# plt.plot(Tw_recons.iloc[it].values)

#%%

# from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor(solver = 'adam',
#                    learning_rate='constant',
#                    learning_rate_init=1e-6,
#                    hidden_layer_sizes=(10,10),
#                    activation='relu',
#                    batch_size = 1,
#                    max_iter=250,
#                    shuffle=False
#                    )

# mlp.fit(df_Xtrain, df_Ytrain)
# train_loss = mlp.loss_curve_
# yhat = mlp.predict(df_Xtest)
# print('MLP train score:', mlp.score(df_Xtrain, df_Ytrain))
# print('MLP test score:', mlp.score(df_Xtest, df_Ytest))

# regression_metrics(df_Ytest, yhat)


#%%
# import sklearn.metrics as metrics
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR

# models = []
# models.append(('LR', LinearRegression()))
# models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
# models.append(('KNN', KNeighborsRegressor()))
# models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
# models.append(('SVR', SVR(gamma='auto'))) # kernel = linear
# # Evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     # TimeSeries Cross validation
#     # tscv = TimeSeriesSplit(n_splits=10)

#     # cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
#     # print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#     model.fit(df_Xtrain, df_Ytrain)
#     yhat = model.predict(df_Xtest)
#     print(name+' train score:', model.score(df_Xtrain, df_Ytrain))
#     print(name+' test score:', model.score(df_Xtest, df_Ytest))
#     results.append([model.score(df_Xtrain, df_Ytrain),model.score(df_Xtest, df_Ytest)])
#     names.append(name)


# # Compare Algorithms
# for im in range(len(results)):
#     plt.plot(im,results[im][1],'o', label=names[im])
# plt.title('Algorithm Comparison')
# plt.show()


#%%
##calculate feature importances

# imp = pd.DataFrame(rf.feature_importances_, index=df_Xtest.columns)#, index=PC
# impy = pd.Series(rf.feature_importances_,
#                   index=df_Xtest.columns).sort_values(ascending=False)#, index=PC
# # impy = impy[:20]
# print(impy)
from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(
    rf, df_Xtest, df_Ytest, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=df_Xtest.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean Rsquare decrease")
fig.tight_layout()
plt.show()


#%%

# import matplotlib.pyplot as plt
# import seaborn as sns
# ZONE_DETECTION_MAIN_ZONE_TYPE = 'Sailing'

# ##barplot of feature importances
# sns.barplot(x=impy, y=impy.index)
# plt.xlabel('Relative Importance')
# plt.ylabel('Feature')
# plt.yticks(rotation=45)
# plt.title(ZONE_DETECTION_MAIN_ZONE_TYPE)# + ' ' + SEGNAME)
# plt.show()

# u = yhat
# v = y_test
# w = range(200, 1600, 10)

# ##scatter plot of regression model accuracy
# plt.scatter(u, v)
# plt.plot(w, w, '-k')
# plt.title('RFR_LAKE_SUPERIOR_DULUTH_UPBOUND')
# plt.ylabel('Predicted fuel consumption rate [kg/hr]')#'ME_SFOC [g/[kW*hr]]')
# plt.xlabel('Actual fuel consuption rate [kg/hr]')#'ER_AMBIENT_TEMP [degC]')#'ME_POWER [kW]')#'SPEED_THROUGH_WATER [knots]')
# plt.show()

# ##predict fuel consumption
# estimate_HOUSE_LOAD = 445.680403#mean: 462.413451[kW] voyage 17031: 478.638681 voyage 17036: 469.041215 voyage 18009: 463.588928 voyage 19001: 445.680403
# estimate_PROPELLER_SHAFT = 550.784552#mean: 609.946372[deg⁄s] voyage 17031: 638.281825 voyage 17036: 598.603450 voyage 18009: 650.867014 voyage 19001: 550.784552
# estimate_SPEED_THROUGH_WATER = 8.694824#mean: 10.913201[kn] voyage 17031: 11.875710 voyage 17036: 9.866438 voyage 18009: 13.072429 voyage 19001: 8.694824
# estimate_TRIM_ANGLE = 0.436278#mean: 0.709661[deg] voyage 17031: 0.399991 voyage 17036: 0.387139 voyage 18009: 1.360823 voyage 19001: 0.436278
# estimate_WATER_DEPTH = 0.008838#mean: 0.016744[nM] voyage 17031: 0.068478 voyage 17036: 0.000343 voyage 18009: 0.007774 voyage 19001: 0.008838
# estimate_X = [[estimate_HOUSE_LOAD, estimate_PROPELLER_SHAFT, estimate_SPEED_THROUGH_WATER, estimate_TRIM_ANGLE, estimate_WATER_DEPTH]]#
# #mean: 833.536287 voyage 17031: 1001.047273 voyage 17036: 705.705032 voyage 18009: 1078.106640 voyage 19001: 544.655281
# print('RFR_estimate_ME_FUEL_CONSUMPTION_RATE:', rf.predict(estimate_X))
# #mean: 1005.71056069 voyage 17031: 1059.86800319 voyage 17036: 797.37413009 voyage 18009: 1074.08066403 voyage 19001: 565.46969425
