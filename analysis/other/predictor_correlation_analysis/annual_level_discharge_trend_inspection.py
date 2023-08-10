#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:47:04 2021

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


def convert_train_valid_to_dataset(iset,istart,target_train,target_valid,features_train1,features_valid1,feature_varnames1,features_train2,features_valid2,feature_varnames2,features_train3,features_valid3,feature_varnames3,start_doy,ws,ns):
    ntraining = features_train1.shape[1]
    nvalid = features_valid1.shape[1]

    nvars_tot = features_train1.shape[0]*features_train1.shape[3]+features_train2.shape[0]*features_train2.shape[3]+features_train3.shape[0]*features_train3.shape[3]
    trainset = np.zeros((ntraining,nvars_tot+1))*np.nan
    validset = np.zeros((nvalid,nvars_tot+1))*np.nan

    # First, add FUD (target variable) to dataset:
    trainset[:,0] = target_train[:,iset]
    validset[:,0] = target_valid[:,iset]

    pred_list = []
    monthly_str = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May','June','July','Aug.','Sept.','Oct.','Nov.','Dec.']
    threemonthly_str = ['N_D_J', 'D_JF', 'JFM', 'FMA', 'MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']
    # window_str = ' weeks ago'

    nwindows_max = np.max([features_train1.shape[3],features_train2.shape[3],features_train3.shape[3]])

    window_to_datestr = np.zeros((features_train1.shape[2],nwindows_max),dtype='object')*np.nan
    for iw in range(nwindows_max):
        if np.abs(-ws-(iw*ns)) <= start_doy:
            doy_arr = np.arange(start_doy-1)+1
            istartw = -ws-(iw*ns)
            iendw = (iw>0)*(-(iw*ns)) + (start_doy-1)*(iw==0)
            str_start = doy_arr[istartw]
            str_end = doy_arr[iendw-1]

            date_start = (dt.date(1997,1,1)+dt.timedelta(days=int(str_start)-1))
            date_start_m = date_start.strftime('%b')
            date_start_d = date_start.strftime('%d')

            date_end = (dt.date(1997,1,1)+dt.timedelta(days=int(str_end)-1))
            date_end_m = date_end.strftime('%b')
            date_end_d = date_end.strftime('%d')

            window_to_datestr[istart,iw] = (date_start_m+' '+date_start_d+' - '+date_end_m+' '+date_end_d)
        else:
            window_to_datestr[istart,iw] = 'NaN Window ' +str(iw)
    window_str = window_to_datestr[istart,:]

    # Then add all possible features:
    v = 1
    for ivar in range(features_train1.shape[0]):
        for iw in range(features_train1.shape[3]):
            # trainset[:,v] =  np.zeros((features_train1[ivar,:,istart,iw,iset]).shape)*np.nan
            # validset[:,v] =  np.zeros((features_valid1[ivar,:,istart,iw,iset]).shape)*np.nan
            # trainset[:,v] =  target_train[:,iset]
            # validset[:,v] =  target_valid[:,iset]
            # pred_list += [monthly_str[iw] +' '+ 'TEST FUD']
            trainset[:,v] =  features_train1[ivar,:,istart,iw,iset]
            validset[:,v] =  features_valid1[ivar,:,istart,iw,iset]
            pred_list += [monthly_str[iw] +' '+ feature_varnames1[ivar]]
            v+=1
    for ivar in range(features_train2.shape[0]):
        for iw in range(features_train2.shape[3]):
            # trainset[:,v] =  np.zeros((features_train2[ivar,:,istart,iw,iset]).shape)*np.nan
            # validset[:,v] =  np.zeros((features_valid2[ivar,:,istart,iw,iset]).shape)*np.nan
            trainset[:,v] =  features_train2[ivar,:,istart,iw,iset]
            validset[:,v] =  features_valid2[ivar,:,istart,iw,iset]
            pred_list += [threemonthly_str[iw] +' '+ feature_varnames2[ivar]]
            v+=1
    for ivar in range(features_train3.shape[0]):
        for iw in range(features_train3.shape[3]):
            # trainset[:,v] =  np.zeros((features_train3[ivar,:,istart,iw,iset]).shape)*np.nan
            # validset[:,v] =  np.zeros((features_valid3[ivar,:,istart,iw,iset]).shape)*np.nan
            trainset[:,v] =  features_train3[ivar,:,istart,iw,iset]
            validset[:,v] =  features_valid3[ivar,:,istart,iw,iset]
            pred_list += [feature_varnames3[ivar] + ', '+ window_str[iw] ]

            v+=1


    df_t = pd.DataFrame(trainset,columns=['FUD']+pred_list)
    df_v = pd.DataFrame(validset,columns=['FUD']+pred_list)

    return df_t, df_v


def forwardSelection(x, Y, sl, columns, verbose):
    import statsmodels.api as sm
    initial_features = columns.tolist()
    best_features = []
    Radj = 0
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype='float64')
        new_Radj = pd.Series(index=remaining_features, dtype='float64')
        for new_column in remaining_features:
            model = sm.OLS(Y, sm.add_constant(x[best_features+[new_column]], has_constant='skip'), missing='drop').fit()
            new_pval[new_column] = model.pvalues[new_column]
            new_Radj[new_column] = model.rsquared_adj
        min_p_value = new_pval.min()
        min_Radj = new_Radj[np.argmin(new_pval)]

        if (min_p_value<sl) & (min_Radj > Radj):
            best_features.append(new_pval.idxmin())
            Radj = min_Radj
        else:
            break

    if verbose:
        model = sm.OLS(Y, sm.add_constant(x[best_features], has_constant='skip'), missing='drop').fit()
        print(model.summary())

    return best_features


def backwardSelection(x, Y, sl, columns, verbose):
    import statsmodels.api as sm
    x = x.values
    Y = Y.values
    numVars = len(x[0])
    for i in range(0, numVars):
        x = sm.add_constant(x, has_constant='skip')
        regressor_OLS = sm.OLS(Y, x, missing='drop').fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i+1):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    if (j> 0):
                        x = np.delete(x, j, 1)
                        columns = np.delete(columns, j-1)
                    else:
                        print('ERROR!!! CONSTANT IS NOT SIGNIFICANT....')

    if verbose:
        regressor_OLS = sm.OLS(Y, x, missing='drop').fit()
        print(regressor_OLS.summary())

    return columns.tolist()


def stepwiseSelection(X, y, columns,
                                      initial_list=[],
                                      threshold_in=0.05,
                                      threshold_out = 0.1,
                                      verbose = True,
                                      use_Radj = False):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    import statsmodels.api as sm
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        initial_features = columns.tolist()
        Radj = 0
        remaining_features = list(set(initial_features)-set(included))
        new_pval = pd.Series(index=remaining_features, dtype='float64')
        if use_Radj:
            new_Radj = pd.Series(index=remaining_features, dtype='float64')
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[included+[new_column]], has_constant='skip'), missing='drop').fit()
            new_pval[new_column] = model.pvalues[new_column]
            if use_Radj:
                new_Radj[new_column] = model.rsquared_adj
        min_p_value = new_pval.min()
        if use_Radj:
            min_Radj = new_Radj[np.argmin(new_pval)]
            if (min_p_value < threshold_in) & (min_Radj > Radj):
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                Radj = min_Radj
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, min_p_value))
        else:
            if (min_p_value < threshold_in):
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, min_p_value))

        # backward step
        model = sm.OLS(y, sm.add_constant(X[included], has_constant='skip'), missing='drop').fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break

        # nmax = 2
        # if len(included) == nmax:
        #     break

    if verbose:
        model = sm.OLS(y, sm.add_constant(X[included], has_constant='skip'), missing='drop').fit()
        print(model.summary())
    return included


#%%%%%%% OPTIONS %%%%%%%%%

plot = False

ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")

#------------------------------
# Period definition
years = np.array([1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021])

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#------------------------------
# Path of raw data
fp_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/'
# Path of processed data
fp_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'

#------------------------------
# Start of forecasts options
# start_doy_arr    = [305,         312,       319,         326,         333,         340]
# start_doy_labels = ['Nov. 1st', 'Nov. 8th', 'Nov. 15th', 'Nov. 22nd', 'Nov. 29th', 'Dec. 6th']
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



#%%%%%%% UPDATE VARIABLES %%%%%%%%%
update_ERA5_vars = False
update_level = False
update_discharge = False
update_daily_NAO = False
update_NAO_monthly = False

update_startdate = dt.date(2021,11,30)
# update_enddate = dt.date(2021,11,25)
update_enddate = dt.date.today()

if update_ERA5_vars:
    # UPDATE ERA5 VARIABLES
    region = 'D'
    fp_p_ERA5 = fp_p + 'ERA5_hourly/region'+region+'/'
    fp_r_ERA5 = fp_r + 'ERA5_hourly/region'+region+'/'
    var_list      = ['u10',                    'v10',                    't2m',           't2m',           't2m',           'd2m',                    'licd',          'lict',                'ltlt',                        'msl',                    'ro',    'siconc',       'sst',                    'sf',      'smlt',    'tcc',              'tp',                 'windspeed','RH',  'SH',  'FDD',  'TDD']
    savename_list = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_temperature','2m_temperature','2m_dewpoint_temperature','lake_ice_depth','lake_ice_temperature','lake_total_layer_temperature','mean_sea_level_pressure','runoff','sea_ice_cover','sea_surface_temperature','snowfall','snowmelt','total_cloud_cover','total_precipitation','windspeed','RH',  'SH',  'FDD',  'TDD']
    vartype_list  = ['mean',                   'mean',                   'mean',          'max',           'min',           'mean',                   'mean',          'mean',                'mean',                        'mean',                   'sum',   'mean',         'mean',                   'sum',     'sum',     'mean',             'sum',                'mean',     'mean','mean','mean','mean']

    for ivar, var in enumerate(var_list):

        var = var_list[ivar]
        processed_varname = savename_list[ivar]
        var_type = 'daily'+vartype_list[ivar]

        data_available, n = datecheck_var_npz('data',fp_p_ERA5+'ERA5_'+var_type+'_'+processed_varname,
                            date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print(var, data_available, n)

        if not data_available:
            print('Updating '+ var +' data... ')
            data = update_ERA5_var(var,var_type,processed_varname,region,update_startdate,update_enddate,fp_r_ERA5,fp_p_ERA5,save=True)
            data_available, n = datecheck_var_npz('data',fp_p_ERA5+'ERA5_'+var_type+'_'+processed_varname,
                                                  date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
            print('Done!', data_available, n)

if update_level:
    # UPDATE WATER LEVEL DATA
    # Note: level data from water.gc.ca cannot be downloaded directly from script.
    #       The csv file corresponding to the 'update_datestr' must be available in the
    #       'raw_fp' directory for this update to work.
    fp_r_QL = fp_r+'QL_ECCC/'
    fp_p_QL = fp_p+'water_levels_discharge_ECCC/'
    loc_name_list = ['PointeClaire','SteAnnedeBellevue']
    loc_nb_list = ['02OA039','02OA013']
    update_datestr = update_enddate.strftime("%b")+'-'+str(update_enddate.day)+'-'+str(update_enddate.year)

    for iloc in range(len(loc_name_list)):
        loc_name = loc_name_list[iloc]
        loc_nb = loc_nb_list[iloc]
        data_available, n = datecheck_var_npz('level',fp_p_QL+'water_levels_discharge_'+loc_name,
                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Water level - ' + loc_name , data_available, n)

        if not data_available:
            print('Updating water level data... ')
            level = update_water_level(update_datestr,loc_name,loc_nb,fp_r_QL,fp_p_QL,save=True)
            data_available, n = datecheck_var_npz('level',fp_p_QL+'water_levels_discharge_'+loc_name,
                              date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
            print('Done!', data_available, n)

if update_discharge:
    # UPDATE DISCHARGE DATA
    # Note: level data from water.gc.ca cannot be downloaded directly from script.
    #       The csv file corresponding to the 'update_datestr' must be available in the
    #       'raw_fp' directory for this update to work.
    fp_r_QL = fp_r+'QL_ECCC/'
    fp_p_QL = fp_p+'water_levels_discharge_ECCC/'
    loc_name = 'Lasalle'
    loc_nb = '02OA016'
    update_datestr = update_enddate.strftime("%b")+'-'+str(update_enddate.day)+'-'+str(update_enddate.year)

    data_available, n = datecheck_var_npz('discharge',fp_p_QL+'water_levels_discharge_'+loc_name,
                      date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
    print('Discharge', data_available, n)

    if not data_available:
        print('Updating discharge data... ')
        level = update_water_discharge(update_datestr,loc_name,loc_nb,fp_r_QL,fp_p_QL,save=True)
        data_available, n = datecheck_var_npz('discharge',fp_p_QL+'water_levels_discharge_'+loc_name,
                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Done!', data_available, n)

if update_daily_NAO:
    fp_r_NAOd = fp_r + 'NAO_daily/'
    fp_p_NAOd = fp_p + 'NAO_daily/'
    NAO_daily = update_daily_NAO_index(fp_r_NAOd,fp_p_NAOd,save=True)

    data_available, n = datecheck_var_npz('data',fp_p_NAOd+'NAO_daily',
                      date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
    print('NAO daily index', data_available, n)

    if not data_available:
        print('Updating NAO daily data... ')
        NAO_daily = update_daily_NAO_index(fp_r_NAOd,fp_p_NAOd,save=True)
        data_available, n = datecheck_var_npz('data',fp_p_NAOd+'NAO_daily',
                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Done!', data_available, n)

if update_NAO_monthly:
    # UPDATE MONTHLY NAO INDEX
    fp_p_NAOm = fp_p + 'NAO_index_NOAA/'
    data_available, n = datecheck_var_npz('data',fp_p_NAOm+'NAO_index_NOAA_monthly',
                                          date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
    print('NAO monthly index',data_available, n)

    if not data_available:
        print('Updating monthly NAO data... ')
        NAO_monthly = update_monthly_NAO_index(fp_p_NAOm, save = True)
        data_available, n = datecheck_var_npz('data',fp_p_NAOm+'NAO_index_NOAA_monthly',
                                              date_check = update_enddate,past_days=(update_enddate-update_startdate).days,n=0.75)
        print('Done!', data_available, n)

#%%%%%%% LOAD VARIABLES %%%%%%%%%

# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['Longueuil','Candiac','Atwater']
station_type = 'cities'
freezeup_opt = 2
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)
# avg_freezeup_doy = freezeup_doy[:,0]

# Average Twater from all locations:
avg_Twater = np.nanmean(Twater,axis=1)
avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
# avg_Twater = Twater[:,0]
# avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
avg_Twater_varnames = ['Avg. Twater']

# Load ERA5 weather variables:
region = 'D'
ERA5_varlist = [#'dailymean_10m_u_component_of_wind',
                #'dailymean_10m_v_component_of_wind',
                'dailymean_2m_temperature',
                'dailymin_2m_temperature',
                'dailymax_2m_temperature',
                #'dailymean_2m_dewpoint_temperature',
                'dailymean_mean_sea_level_pressure',
                'dailysum_runoff',
                'dailysum_snowfall',
                #'dailysum_snowmelt',
                'dailysum_total_precipitation',
                'dailymean_total_cloud_cover',
                'dailymean_windspeed',
                'daily_theta_wind',
                #'dailymean_RH',
                'dailymean_FDD',
                'dailymean_TDD'
                ]
fp_p_ERA5 = fp_p + 'ERA5_hourly/region'+region+'/'
weather_vars, weather_varnames = load_weather_vars_ERA5(fp_p_ERA5,ERA5_varlist,region,time)
# weather_varnames = ['Avg. Ta_max','Avg. Ta_min','Avg. Ta_mean','Tot. TDD','Tot. FDD','Tot. precip.','Avg. SLP','Avg. wind speed','Avg. u-wind','Avg. v-wind','Tot. snowfall','Avg. cloud cover','Avg. spec. hum.','Avg. rel. hum.']

# Load monthly NAO data
# NAO_monthly_data = np.load(fp_p+'NAO_index_NOAA/NAO_index_NOAA_monthly.npz',allow_pickle='TRUE')
# NAO_monthly_vars = NAO_monthly_data['data']
# NAO_monthly_varnames = ['Avg. monthly NAO']

# Load daily NAO data
NAO_daily_data = np.load(fp_p+'NAO_daily/NAO_daily.npz',allow_pickle='TRUE')
NAO_daily_vars = NAO_daily_data['data']
NAO_daily_varnames = ['Avg. daily NAO']

# Load monthly PDO data
# fn = 'PDO_index_NOAA_monthly_ersstv3.npz'
fn = 'PDO_index_NOAA_monthly_ersstv5.npz'
# fn = 'PDO_index_NOAA_monthly_hadisst1.npz'
PDO_data = np.load(fp_p+'PDO_index_NOAA/'+fn,allow_pickle='TRUE')
PDO_vars = PDO_data['PDO_data']
PDO_varnames = ['Avg. monthly PDO']

# Load monthly EL NINO data
fn = 'Nino34_index_NOAA_monthly.npz'
ENSO_data = np.load(fp_p+'Nino34_index_NOAA/'+fn,allow_pickle='TRUE')
ENSO_vars = ENSO_data['Nino34_data']
ENSO_varnames = ['Avg. montlhy Nino34']
# fn = 'ONI_index_NOAA_monthly.npz'
# ENSO_data = np.load(fp_p+'ONI_index_NOAA/'+fn,allow_pickle='TRUE')
# ENSO_vars = ENSO_data['ONI_data']
# ENSO_varnames = ['ONI']

# Load discharge and level data
loc_discharge = 'Lasalle'
discharge_data = np.load(fp_p+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_discharge+'.npz',allow_pickle=True)
discharge_vars = discharge_data['discharge'][:,1]
discharge_vars = np.expand_dims(discharge_vars, axis=1)
discharge_varnames= ['Avg. discharge St-L. River']

loc_level = 'PointeClaire'
level_data = np.load(fp_p+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_level+'.npz',allow_pickle=True)
level_vars = level_data['level'][:,1]
level_vars = np.expand_dims(level_vars, axis=1)
level_varnames= ['Avg. level St-L. River']

loc_levelO = 'SteAnnedeBellevue'
levelO_data = np.load(fp_p+'water_levels_discharge_ECCC/water_levels_discharge_'+loc_levelO+'.npz',allow_pickle=True)
levelO_vars = levelO_data['level'][:,1]
levelO_vars = np.expand_dims(levelO_vars, axis=1)
levelO_varnames= ['Avg. level Ottawa River']

Ta_mean = weather_vars[:,0]
SLP = weather_vars[:,3]
precip = weather_vars[:,6]
snowfall = weather_vars[:,5]
clouds = weather_vars[:,7]


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



Tw_annual = annual_average_from_daily_ts(avg_Twater_vars, years, time)
level_annual = annual_average_from_daily_ts(level_vars, years, time)
levelO_annual = annual_average_from_daily_ts(levelO_vars, years, time)
discharge_annual = annual_average_from_daily_ts(discharge_vars, years, time)

Ta_mean_annual = annual_average_from_daily_ts(Ta_mean, years, time)
SLP_annual = annual_average_from_daily_ts(SLP, years, time)
precip_annual = annual_average_from_daily_ts(precip, years, time)
snowfall_annual = annual_average_from_daily_ts(snowfall, years, time)
clouds_annual = annual_average_from_daily_ts(clouds, years, time)

ENSO_annual =  annual_average_from_daily_ts(ENSO_vars, years, time)
NAO_daily_annual =  annual_average_from_daily_ts(NAO_daily_vars, years, time)
PDO_annual =  annual_average_from_daily_ts(PDO_vars, years, time)

#%%

from functions import rolling_climo

Tw_climo,_,_ = rolling_climo(31,np.squeeze(avg_Twater_vars),'all_time',time,years[:-1])

plt.figure()
plt.plot(time,Tw_climo)
plt.plot(time,avg_Twater_vars,color='k')

plt.figure()
plt.plot(time,np.squeeze(avg_Twater_vars)-Tw_climo)

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(np.squeeze(avg_Twater_vars)-Tw_climo,lags=560,missing='drop')

#%%

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,Tw_annual,'o-')
ax.grid()
ax.set_xlabel('Year')
ax.set_ylabel('Annual water temp. ($^{\circ}C$)',color='black')

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,level_annual,'o-', label='Annual St-L. Level',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('Level ($m$)',color='black')
ax.grid()
ax2=ax.twinx()
ax2.plot(years,discharge_annual,'o--', color= plt.get_cmap('tab10')(0),label='Discharge')
ax2.set_ylabel('Discharge ($m^{3}/s$)', color= plt.get_cmap('tab10')(0))
ax2.legend()
ax.legend()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,SLP_annual,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('SLP ($Pa$)',color='black')
plt.title('SLP')
ax.grid()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,precip_annual,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('Precip ($m$)',color='black')
plt.title('Precip')
ax.grid()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,snowfall_annual,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('Snowfall ($m$ equivalent)',color='black')
plt.title('Snowfall')
ax.grid()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,clouds_annual,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('Clouds (percent)',color='black')
plt.title('Clouds')
ax.grid()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,Ta_mean_annual,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('Ta_mean ($^{\circ}C$)',color='black')
plt.title('Ta_mean')
ax.grid()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,ENSO_annual,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('ENSO ($^{\circ}C$)',color='black')
plt.title('ENSO')
ax.grid()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,NAO_daily_annual,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('NAO ($^{\circ}C$)',color='black')
plt.title('NAO')
ax.grid()
#%%
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,PDO_annual,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('PDO ($^{\circ}C$)',color='black')
plt.title('PDO')
ax.grid()

#%%

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(years,avg_freezeup_doy,'o-',color='black')
ax.set_xlabel('Year')
ax.set_ylabel('FUD',color='black')
ax.grid()

#%%
anomaly = 'linear'
detrend = False

if detrend:
    Tw_annual_all,[m,b] = detrend_ts(Tw_annual,years,anomaly)
    level_annual_all,[m,b] = detrend_ts(level_annual,years,anomaly)
    discharge_annual_all,[m,b] = detrend_ts(discharge_annual,years,anomaly)

    Tw_annual_1,[m,b] = detrend_ts(Tw_annual[0:20],years[0:20],anomaly)
    level_annual_1,[m,b] = detrend_ts(level_annual[0:20],years[0:20],anomaly)
    discharge_annual_1,[m,b] = detrend_ts(discharge_annual[0:20],years[0:20],anomaly)

    Tw_annual_2,[m,b] = detrend_ts(Tw_annual[20:],years[20:],anomaly)
    level_annual_2,[m,b] = detrend_ts(level_annual[20:],years[20:],anomaly)
    discharge_annual_2,[m,b] = detrend_ts(discharge_annual[20:],years[20:],anomaly)

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], discharge_annual_all[0:20],'o')
    plt.plot(avg_freezeup_doy[20:], discharge_annual_all[20:],'v')

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], level_annual_all[0:20],'o')
    plt.plot(avg_freezeup_doy[20:], level_annual_all[20:],'v')

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], Tw_annual_all[0:20],'o')
    plt.plot(avg_freezeup_doy[20:], Tw_annual_all[20:],'v')

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], discharge_annual_1,'o')
    plt.plot(avg_freezeup_doy[20:], discharge_annual_2,'v')

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], level_annual_1,'o')
    plt.plot(avg_freezeup_doy[20:], level_annual_2,'v')

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], Tw_annual_1,'o')
    plt.plot(avg_freezeup_doy[20:], Tw_annual_2,'v')


    fig, ax = plt.subplots(figsize=(11,3))
    ax.plot(years,Tw_annual_all,'o-',color='k')
    ax.plot(years[0:20],Tw_annual_1,'x:', color= plt.get_cmap('tab10')(2))
    ax.plot(years[20:],Tw_annual_2,'x:', color= plt.get_cmap('tab10')(2))
    ax.grid()
    ax.set_xlabel('Year')
    ax.set_ylabel('Detrended annual water temp. ($^{\circ}C$)',color='black')

    fig, ax = plt.subplots(figsize=(11,3))
    ax.plot(years,level_annual_all,'o-', label='Annual St-L. Level',color='black')
    ax.plot(years[0:20],level_annual_1,'x:', color= plt.get_cmap('tab10')(4))
    ax.plot(years[20:],level_annual_2,'x:', color= plt.get_cmap('tab10')(4))
    ax.set_xlabel('Year')
    ax.set_ylabel('Detrended Level ($m$)',color='black')
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(years,discharge_annual_all,'o--', color= plt.get_cmap('tab10')(0),label='Discharge')
    ax2.plot(years[0:20],discharge_annual_1,'x:', color= plt.get_cmap('tab10')(2))
    ax2.plot(years[20:],discharge_annual_2,'x:', color= plt.get_cmap('tab10')(2))
    ax2.set_ylabel('Detrended Discharge ($m^{3}/s$)', color= plt.get_cmap('tab10')(0))
    ax2.legend()
    ax.legend()

else:

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], discharge_annual[0:20],'o')
    plt.plot(avg_freezeup_doy[20:], discharge_annual[20:],'v')

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], level_annual[0:20],'o')
    plt.plot(avg_freezeup_doy[20:], level_annual[20:],'v')

    plt.figure()
    plt.plot(avg_freezeup_doy[0:20], Tw_annual[0:20],'o')
    plt.plot(avg_freezeup_doy[20:], Tw_annual[20:],'v')






