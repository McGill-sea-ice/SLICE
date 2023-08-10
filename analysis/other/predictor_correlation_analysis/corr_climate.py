#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 08:33:54 2022

@author: Amelie
"""
import numpy as np
import datetime as dt
import calendar
import matplotlib.pyplot as plt
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir='/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/temp_files/') #python

import statsmodels.api as sm
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import pandas as pd
import seaborn as sns

from functions_MLR import datecheck_var_npz,update_water_level,update_monthly_NAO_index
from functions_MLR import update_ERA5_var,load_weather_vars_ERA5
from functions_MLR import update_daily_NAO_index,update_water_discharge
from functions_MLR import get_monthly_vars_from_daily, get_3month_vars_from_daily, get_rollingwindow_vars_from_daily

from functions import detect_FUD_from_Tw,detrend_ts,bootstrap
from functions import r_confidence_interval

#%%
def annual_average_from_daily_ts(var_in,years,time,date_ref = dt.date(1900,1,1)):

    var_out = np.zeros((len(years)))*np.nan

    for iyr,year in enumerate(years):
        if ((dt.date(int(year),1,1)-date_ref).days in time) & ((dt.date(int(year+1),1,1)-date_ref).days in time):
            it_start = np.where(time == (dt.date(int(year),1,1)-date_ref).days )[0][0]
            # it_end = np.where(time == (dt.date(int(year+1),1,1)-date_ref).days )[0][0]
            it_end = np.where(time == (dt.date(int(year),12,1)-date_ref).days )[0][0]

            var_yr = var_in[it_start:it_end]
            var_out[iyr] = np.nanmean(var_yr)

    return var_out


def remove_collinear_features(df_model, target_var, threshold, verbose):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold and which have the least correlation with the
        target (dependent) variable. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        df_model: features dataframe
        target_var: target (dependent) variable
        threshold: features with correlations greater than this value are removed
        verbose: set to "True" for the log printing

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = df_model.drop(target_var, 1).corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    dropped_feature = ""

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                if verbose:
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                col_value_corr = np.abs(df_model[col.values[0]].corr(df_model[target_var]))
                row_value_corr = np.abs(df_model[row.values[0]].corr(df_model[target_var]))
                if verbose:
                    print("{}: {}".format(col.values[0], np.round(col_value_corr, 3)))
                    print("{}: {}".format(row.values[0], np.round(row_value_corr, 3)))
                if col_value_corr < row_value_corr:
                    drop_cols.append(col.values[0])
                    dropped_feature = "dropped: " + col.values[0]
                else:
                    drop_cols.append(row.values[0])
                    dropped_feature = "dropped: " + row.values[0]
                if verbose:
                    print(dropped_feature)
                    print("-----------------------------------------------------------------------------")

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    df_model = df_model.drop(columns=drops)

    if verbose:
        print("dropped columns: ")
        print(list(drops))
        print("-----------------------------------------------------------------------------")
        print("used columns: ")
        print(df_model.columns.tolist())

    return df_model


#%%%%%%% OPTIONS %%%%%%%%%

plot = False

ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")

#------------------------------
# Period definition
years = np.arange(1979,2022)
years_FUD = np.arange(1991,2022)

date_ref = dt.date(1900,1,1)
date_start = dt.date(1979,1,1)
date_start_FUD = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
time_FUD = np.arange((date_start_FUD-date_ref).days, (date_end-date_ref).days+1)

#------------------------------
# Path of raw data
fp_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/'
# Path of processed data
fp_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'

#------------------------------
# Start of forecasts options
start_doy_arr    = [300,         307,       314,         321,         328,         335]
start_doy_labels = ['Oct. 27th', 'Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st']

#------------------------------
# Correlation analysis
p_critical = 0.01

replace_with_nan = False

detrend_FUD = False
detrend = False
if detrend:
   anomaly = 'linear'


#%%%%%%% LOAD VARIABLES %%%%%%%%%

# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['Longueuil','Candiac','Atwater']
station_type = 'cities'
freezeup_opt = 2
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years_FUD,time_FUD,show=False)

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)
# avg_freezeup_doy = freezeup_doy[:,0]

# put FUD on same time axis as climate indices
FUD = np.zeros((len(years)))*np.nan
for iy,year in enumerate(years_FUD):
    it_n = np.where(years == years_FUD[iy])[0][0]
    FUD[it_n] = avg_freezeup_doy[iy]

# BEAUHARNOIS FREEZE-UP:
FUD_data = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/freezeup_dates_HQ/freezeup_HQ_BeauharnoisCanal.npz')
fi = FUD_data['freezeup_fi'][:]
si = FUD_data['freezeup_si'][:]

fi = fi[~np.isnan(fi)]
si = si[~np.isnan(si)]

# years_beauharnois = np.arange(1979,2020)
# doy_fi_beauharnois = np.zeros((len(fi)))*np.nan
# doy_si_beauharnois = np.zeros((len(si)))*np.nan
# for i in range(len(fi)):
#     date_FUD_fi = date_ref + dt.timedelta(days=int(fi[i]))
#     if date_FUD_fi.year == years_beauharnois[i]:
#         doy_FUD_fi = (date_FUD_fi-dt.date(years_beauharnois[i],1,1)).days + 1
#     else:
#         doy_FUD_fi = (365 + calendar.isleap(years_beauharnois[i]) +
#                       (date_FUD_fi-dt.date(years_beauharnois[i]+1,1,1)).days + 1)
#     doy_fi_beauharnois[i] = doy_FUD_fi

#     date_FUD_si = date_ref + dt.timedelta(days=int(si[i]))
#     if date_FUD_si.year == years_beauharnois[i]:
#         doy_FUD_si = (date_FUD_si-dt.date(years_beauharnois[i],1,1)).days + 1
#     else:
#         doy_FUD_si = (365 + calendar.isleap(years_beauharnois[i]) +
#                      (date_FUD_si-dt.date(years_beauharnois[i]+1,1,1)).days + 1)
#     doy_si_beauharnois[i] = doy_FUD_si

# # put FUD on same time axis as climate indices
# FUD_fi_b = np.zeros((len(years)))*np.nan
# FUD_si_b = np.zeros((len(years)))*np.nan
# for iy,year in enumerate(years_beauharnois):
#     it_n = np.where(years == years_beauharnois[iy])[0][0]
#     FUD_fi_b[it_n] = doy_fi_beauharnois[iy]
#     FUD_si_b[it_n] = doy_si_beauharnois[iy]

# Average Twater from all locations:
avg_Twater = np.nanmean(Twater,axis=1)
avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
# avg_Twater = Twater[:,0]
# avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
avg_Twater_varnames = ['Avg. Twater']

# put Twater on same time axis as climate indices
Tw = np.zeros((len(time),1))*np.nan
for it in range(len(time_FUD)):
    it_n = np.where(time == time_FUD[it])[0][0]
    Tw[it_n,0] = avg_Twater_vars[it,0]



index_list   = ['NAO',  'PDO',    'ONI',    'AO',   'PNA',  'WP',     'TNH',    'SCAND',  'PT',     'POLEUR', 'EPNP',   'EA']
timerep_list = ['daily','monthly','monthly','daily','daily','monthly','monthly','monthly','monthly','monthly','monthly','monthly']
ci_varnames = []
ci_vars = np.zeros((len(time),len(index_list)))*np.nan
for i,iname in enumerate(index_list):
    if iname == 'PDO':
        # vexp = 'ersstv3'
        vexp = 'ersstv5'
        # vexp = 'hadisst1'
        fn = iname+'_index_'+timerep_list[i]+'_'+vexp+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['PDO_data'])
    elif iname == 'ONI':
        fn = iname+'_index_'+timerep_list[i]+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['ONI_data'])
    else:
        fn = iname+'_index_'+timerep_list[i]+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['data'])
    ci_varnames += [iname]

# Load discharge and level data
loc_discharge = 'Lasalle'
discharge_data = np.load(fp_p+'water_levels_discharge_ECCC/uwater_levels_discharge_'+loc_discharge+'.npz',allow_pickle=True)
discharge_vars = discharge_data['discharge'][:,1]
discharge_vars = np.expand_dims(discharge_vars, axis=1)
discharge_varnames= ['Avg. discharge St-L. River']

loc_level = 'PointeClaire'
level_data = np.load(fp_p+'water_levels_discharge_ECCC/uwater_levels_discharge_'+loc_level+'.npz',allow_pickle=True)
level_vars = level_data['level'][:,1]
level_vars = np.expand_dims(level_vars, axis=1)
level_varnames= ['Avg. level St-L. River']

loc_levelO = 'SteAnnedeBellevue'
levelO_data = np.load(fp_p+'water_levels_discharge_ECCC/uwater_levels_discharge_'+loc_levelO+'.npz',allow_pickle=True)
levelO_vars = levelO_data['level'][:,1]
levelO_vars = np.expand_dims(levelO_vars, axis=1)
levelO_varnames= ['Avg. level Ottawa River']

#%%
Tw_annual = annual_average_from_daily_ts(Tw, years, time)
level_annual = annual_average_from_daily_ts(level_vars, years, time)
levelO_annual = annual_average_from_daily_ts(levelO_vars, years, time)
discharge_annual = annual_average_from_daily_ts(discharge_vars, years, time)

ci_annual = np.zeros((len(years),len(index_list)))*np.nan
for i in range(len(index_list)):
    ci_annual[:,i] =  annual_average_from_daily_ts(ci_vars[:,i], years, time)

plot = True
if plot:
    fig, ax = plt.subplots(figsize=(6,3))
    plt.title(index_list[i])
    # ax.plot(years,FUD,'.-', color= 'black')
    # ax.plot(years,FUD_fi_b,'.-', color= 'gray')
    # ax.set_ylabel('FUD', color= 'black')
    ax2=ax.twinx()
    ax2.plot(years,Tw_annual,'.-', color= plt.get_cmap('tab10')(0))
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Tw', color= plt.get_cmap('tab10')(0))
    ax2.grid()

    fig, ax = plt.subplots(figsize=(6,3))
    plt.title('Level - Lasalle')
    # ax.plot(years,FUD,'.-', color= 'black')
    # ax.plot(years,FUD_fi_b,'.-', color= 'gray')
    # ax.set_ylabel('FUD', color= 'black')
    ax2=ax.twinx()
    ax2.plot(years,level_annual,'.-', color= plt.get_cmap('tab10')(0))
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Level', color= plt.get_cmap('tab10')(0))
    ax2.grid()

    fig, ax = plt.subplots(figsize=(6,3))
    plt.title('Discharge - Pointe-Claire')
    # ax.plot(years,FUD,'.-', color= 'black')
    # ax.plot(years,FUD_fi_b,'.-', color= 'gray')
    # ax.set_ylabel('FUD', color= 'black')
    ax2=ax.twinx()
    ax2.plot(years,discharge_annual,'.-', color= plt.get_cmap('tab10')(0))
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Discharge', color= plt.get_cmap('tab10')(0))
    ax2.grid()

    fig, ax = plt.subplots(figsize=(6,3))
    plt.title('Level Ott. River')
    # ax.plot(years,FUD,'.-', color= 'black')
    # ax.plot(years,FUD_fi_b,'.-', color= 'gray')
    # ax.set_ylabel('FUD', color= 'black')
    ax2=ax.twinx()
    ax2.plot(years,levelO_annual,'.-', color= plt.get_cmap('tab10')(0))
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Level Ott. Riv.', color= plt.get_cmap('tab10')(0))
    ax2.grid()

    for i in range(len(index_list)):
        fig, ax = plt.subplots(figsize=(6,3))
        plt.title(index_list[i])
        # ax.plot(years,FUD,'.-', color= 'black')
        # ax.plot(years,FUD_fi_b,'.-', color= 'gray')
        # ax.set_ylabel('FUD', color= 'black')
        ax2=ax.twinx()
        ax2.plot(years,ci_annual[:,i],'.-', color= plt.get_cmap('tab10')(0))
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Index', color= plt.get_cmap('tab10')(0))
        ax2.grid()


#%%%%%%% GET MONTHLY VARIABLES %%%%%%%%%
# Average for Jan, Feb, Mar, April,...., Dec
Tw_monthly = get_monthly_vars_from_daily(Tw,['Avg.'],years,time,replace_with_nan)
discharge_monthly = get_monthly_vars_from_daily(discharge_vars,['Avg.'],years,time,replace_with_nan)
level_monthly = get_monthly_vars_from_daily(level_vars,['Avg.'],years,time,replace_with_nan)
levelO_monthly = get_monthly_vars_from_daily(levelO_vars,['Avg.'],years,time,replace_with_nan)
ci_monthly= get_monthly_vars_from_daily(ci_vars,['Avg.' + index_list[j] for j in range(len(index_list)) ],years,time,replace_with_nan)

#%%%%%%% GET 3-MONTH VARIABLES %%%%%%%%%
#average for # -, -, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND
Tw_threemonthly = get_3month_vars_from_daily(Tw,['Avg.'],years,time,replace_with_nan)
discharge_threemonthly = get_3month_vars_from_daily(discharge_vars,['Avg.'],years,time,replace_with_nan)
level_threemonthly = get_3month_vars_from_daily(level_vars,['Avg.'],years,time,replace_with_nan)
levelO_threemonthly = get_3month_vars_from_daily(levelO_vars,['Avg.'],years,time,replace_with_nan)
ci_threemonthly = get_3month_vars_from_daily(ci_vars,['Avg.' + index_list[j] for j in range(len(index_list)) ],years,time,replace_with_nan)

#%%%%%%%

# vars_annual = [Tw_annual,discharge_annual,level_annual,levelO_annual]
varnames = ['Tw','discharge','level','levelO']+ index_list

df = np.zeros((len(years),(1+12+10)*len(varnames)+1))*np.nan
FUD_tmp = FUD.copy()
# FUD_tmp = FUD_si_b.copy()
# FUD_tmp[37:] = np.nan # (use 1992-2015)
# FUD_tmp[:19] = np.nan # (use 1998-2021)
df[:,0] = FUD_tmp
df[:,1] = Tw_annual
df[:,2] = discharge_annual
df[:,3] = level_annual
df[:,4] = levelO_annual
df[:,5:5+len(index_list)] = ci_annual

columns = ['FUD','annual Tw','annual discharge','annual level','annual levelO'] + ['annual '+index_list[j] for j in range(len(index_list))]

month_str = ['Jan.','Feb.','Mar.','Apr.','May','June','Jul.','Aug.','Sep.','Oct.','Nov.','Dec.']
for i in range(12):
    df[:,5+len(index_list)+i] = Tw_monthly[0,:,i]
    columns += [month_str[i]+ ' Tw']

for i in range(12):
    df[:,5+len(index_list)+12+i] = discharge_monthly[0,:,i]
    columns += [month_str[i]+ ' discharge']

for i in range(12):
    df[:,5+len(index_list)+24+i] = level_monthly[0,:,i]
    columns += [month_str[i]+ ' level']

for i in range(12):
    df[:,5+len(index_list)+36+i] = levelO_monthly[0,:,i]
    columns += [month_str[i]+ ' level Ott.']

for ivar in range(len(index_list)):
    for i in range(12):
        df[:,5+len(index_list)+48+(12*ivar)+i] = ci_monthly[ivar,:,i]
        columns += [month_str[i]+' '+index_list[ivar]]

threemonth_str = ['JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']
for i in range(10):
    df[:,5+len(index_list)+48+(12*len(index_list))+i] = Tw_threemonthly[0,:,i]
    columns += [threemonth_str[i]+ ' Tw']

for i in range(10):
    df[:,5+len(index_list)+48+(12*len(index_list))+10+i] = discharge_threemonthly[0,:,i+2]
    columns += [threemonth_str[i]+ ' discharge']

for i in range(10):
    df[:,5+len(index_list)+48+(12*len(index_list))+20+i] = level_threemonthly[0,:,i+2]
    columns += [threemonth_str[i]+ ' level']

for i in range(10):
    df[:,5+len(index_list)+48+(10*len(index_list))+30+i] = levelO_threemonthly[0,:,i+2]
    columns += [threemonth_str[i]+ ' level Ott.']

for ivar in range(len(index_list)):
    for i in range(10):
        df[:,5+len(index_list)+48+(12*len(index_list))+40+(10*ivar)+i] = ci_threemonthly[ivar,:,i+2]
        columns += [threemonth_str[i]+' '+index_list[ivar]]

#%%
df = pd.DataFrame(df,columns=columns)
cor = df.corr(min_periods=len(years)*0.5)
cor_target = cor["FUD"]

#Selecting features significatively correlated with target
navailable = [np.sum(~np.isnan(df[c])) for c in df.columns]
threshold = [r_confidence_interval(0,p_critical,navailable[c],tailed='two')[1] for c in range(len(df.columns))]
relevant_features = cor_target[abs(cor_target)>=threshold]
df_sel = df[[relevant_features.index[i] for i in range(len(relevant_features))]]
print(relevant_features)

plot = False
if plot:
    plt.figure()
    plt.plot(np.arange(len(threshold)),threshold,':',color='gray')
    plt.plot(np.arange(len(cor_target)),cor_target,'.-')
#%%
# SECOND PRESELECTION STEP: REMOVE COLLINEAR FEATURES
cor_mc = np.abs(df_sel.corr())
upper_cor = cor_mc.where(np.triu(np.ones(cor_mc.shape),k=1).astype(np.bool))
# Plot correlation heat map between features
plot = False
if plot:
    plt.figure(figsize=(12,10))
    sns.heatmap(cor_mc, cmap=plt.cm.Reds)
    plt.show()

df_clean = remove_collinear_features(df_sel,'FUD', 0.7, verbose=True)

#%%

# x = years[:]
# y = FUD_si_b[:]
# model = sm.OLS(y,sm.add_constant(x,has_constant='skip'), missing='drop').fit()


# print(model.params, model.pvalues)

