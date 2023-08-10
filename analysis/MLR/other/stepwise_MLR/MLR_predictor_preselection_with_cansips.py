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
from functions_MLR import get_daily_var_from_monthly_cansips_forecasts,get_daily_var_from_seasonal_cansips_forecasts

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

    # print(features_train1.shape[0]*features_train1.shape[3],len(feature_varnames1))

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
save_ML = False

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
# Twater_loc_list = ['Longueuil_updated','Candiac','Atwater']
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 2
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)

# Average (and round) FUD from all locations:
# avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
# avg_freezeup_doy = np.round(avg_freezeup_doy)
avg_freezeup_doy = freezeup_doy[:,0]

# Average Twater from all locations:
# avg_Twater = np.nanmean(Twater,axis=1)
# avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
avg_Twater = Twater[:,0]
avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
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
                'dailymean_TDD',
                'dailymean_surface_solar_radiation_downwards',
                'dailymean_surface_latent_heat_flux',
                'dailymean_surface_sensible_heat_flux'
                ]
fp_p_ERA5 = fp_p + 'ERA5_hourly/region'+region+'/'
weather_vars, weather_varnames = load_weather_vars_ERA5(fp_p_ERA5,ERA5_varlist,region,time)
# weather_varnames = ['Avg. Ta_max','Avg. Ta_min','Avg. Ta_mean','Tot. TDD','Tot. FDD','Tot. precip.','Avg. SLP','Avg. wind speed','Avg. u-wind','Avg. v-wind','Tot. snowfall','Avg. cloud cover','Avg. spec. hum.','Avg. rel. hum.']

# # Load daily NAO data
# NAO_daily_data = np.load(fp_p+'NAO_daily/NAO_daily.npz',allow_pickle='TRUE')
# NAO_daily_vars = NAO_daily_data['data']
# NAO_daily_varnames = ['Avg. daily NAO']

# # Load monthly PDO data
# # fn = 'PDO_index_NOAA_monthly_ersstv3.npz'
# # fn = 'PDO_index_NOAA_monthly_ersstv5.npz'
# fn = 'PDO_index_NOAA_monthly_hadisst1.npz'
# PDO_data = np.load(fp_p+'PDO_index_NOAA/'+fn,allow_pickle='TRUE')
# PDO_vars = PDO_data['PDO_data']
# PDO_varnames = ['Avg. monthly PDO']

# # Load monthly EL NINO data
# fn = 'Nino34_index_NOAA_monthly.npz'
# ENSO_data = np.load(fp_p+'Nino34_index_NOAA/'+fn,allow_pickle='TRUE')
# ENSO_vars = ENSO_data['Nino34_data']
# ENSO_varnames = ['Avg. montlhy Nino34']
# # fn = 'ONI_index_NOAA_monthly.npz'
# # ENSO_data = np.load(fp_p+'ONI_index_NOAA/'+fn,allow_pickle='TRUE')
# # ENSO_vars = ENSO_data['ONI_data']
# # ENSO_varnames = ['ONI']

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


index_list   = ['AMO',    'SOI',    'NAO',  'PDO',    'ONI',    'AO',   'PNA',  'WP',     'TNH',    'SCAND',  'PT',     'POLEUR', 'EPNP',   'EA']
timerep_list = ['monthly','monthly','daily','monthly','monthly','daily','daily','monthly','monthly','monthly','monthly','monthly','monthly','monthly']
ci_varnames = []
ci_vars = np.zeros((len(time),len(index_list)))*np.nan
for i,iname in enumerate(index_list):
    if iname == 'PDO':
        # vexp = 'ersstv3'
        vexp = 'ersstv5'
        # vexp = 'hadisst1'
        fn = iname+'_index_'+timerep_list[i]+'_'+vexp+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['PDO_data'][365:])
    elif iname == 'ONI':
        fn = iname+'_index_'+timerep_list[i]+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['ONI_data'][365:])
    else:
        fn = iname+'_index_'+timerep_list[i]+'.npz'
        data = np.load(fp_p+'climate_indices_NOAA/'+fn,allow_pickle='TRUE')
        ci_vars[:,i] = np.squeeze(data['data'][365:])
    ci_varnames += [iname]

#%%%%%%%%%%% Load CanSIPS seasonal forecasts %%%%%%%%%%%
feature_list = [#'WTMP_SFC_0',
                'PRATE_SFC_0',
                'TMP_TGL_2m',
                'PRMSL_MSL_0'
                ]
region_cansips = 'D'
# region_cansips = 'YUL'
anomaly_cansips = True
sort_type = 'startmonth'
lag=1

cansips_seasonal_vars_startO = np.zeros((len(time),len(feature_list)*7))*np.nan
cansips_seasonal_vars_startN = np.zeros((len(time),len(feature_list)*9))*np.nan
cansips_seasonal_vars_startD = np.zeros((len(time),len(feature_list)*10))*np.nan
cansips_seasonal_startO_varnames = []
cansips_seasonal_startN_varnames = []
cansips_seasonal_startD_varnames = []
for f,feature in enumerate(feature_list):
    varname = feature
    cansips_seasonal_vars_startO[:,(7*f):7*(f+1)], varnames_startO = get_daily_var_from_seasonal_cansips_forecasts(sort_type,1,varname,anomaly_cansips,region_cansips,time,lag=lag)
    cansips_seasonal_vars_startN[:,(9*f):9*(f+1)], varnames_startN = get_daily_var_from_seasonal_cansips_forecasts(sort_type,2,varname,anomaly_cansips,region_cansips,time,lag=lag)
    cansips_seasonal_vars_startD[:,(10*f):10*(f+1)], varnames_startD = get_daily_var_from_seasonal_cansips_forecasts(sort_type,3,varname,anomaly_cansips,region_cansips,time,lag=lag)
    cansips_seasonal_startO_varnames += [varnames_startO[i] for i in range(len(varnames_startO))]
    cansips_seasonal_startN_varnames += [varnames_startN[i] for i in range(len(varnames_startN))]
    cansips_seasonal_startD_varnames += [varnames_startD[i] for i in range(len(varnames_startD))]


# Convert variables to one time series per forecast start month
cansips_seasonal_vars_concat_sep1st = np.zeros((len(time),len(feature_list)))
cansips_seasonal_vars_concat_oct1st = np.zeros((len(time),len(feature_list)))
cansips_seasonal_vars_concat_nov1st = np.zeros((len(time),len(feature_list)))
cansips_seasonal_vars_concat_dec1st = np.zeros((len(time),len(feature_list)))

for f,feature in enumerate(feature_list):

    lf = int(cansips_seasonal_vars_startD.shape[1]/len(feature_list))

    sep_allmonths_seasonal = cansips_seasonal_vars_startD[:,(f*lf)      :(f*lf)+4    ]
    oct_allmonths_seasonal = cansips_seasonal_vars_startD[:,(f*lf)+4    :(f*lf)+4+3  ]
    nov_allmonths_seasonal = cansips_seasonal_vars_startD[:,(f*lf)+4+3  :(f*lf)+4+3+2]
    dec_allmonths_seasonal = cansips_seasonal_vars_startD[:,(f*lf)+4+3+2:(f*lf)+4+3+2+1]

    for year in range(date_start.year,date_end.year+1):

        it_sep1st = np.where(time == (dt.date(year, 9,1)-date_ref).days)[0][0]
        it_oct1st = np.where(time == (dt.date(year,10,1)-date_ref).days)[0][0]
        it_nov1st = np.where(time == (dt.date(year,11,1)-date_ref).days)[0][0]
        it_dec1st = np.where(time == (dt.date(year,12,1)-date_ref).days)[0][0]
        if year < years[-1]:
            it_jan1st = np.where(time == (dt.date(year+1,1,1)-date_ref).days)[0][0]
        else:
            it_jan1st = len(time)

        cansips_seasonal_vars_concat_sep1st[it_sep1st:it_oct1st,f] = sep_allmonths_seasonal[it_sep1st,0]
        cansips_seasonal_vars_concat_sep1st[it_oct1st:it_nov1st,f] = sep_allmonths_seasonal[it_sep1st,1]
        cansips_seasonal_vars_concat_sep1st[it_nov1st:it_dec1st,f] = sep_allmonths_seasonal[it_sep1st,2]
        cansips_seasonal_vars_concat_sep1st[it_dec1st:it_jan1st,f] = sep_allmonths_seasonal[it_sep1st,3]

        cansips_seasonal_vars_concat_oct1st[it_oct1st:it_nov1st,f] = oct_allmonths_seasonal[it_oct1st,0]
        cansips_seasonal_vars_concat_oct1st[it_nov1st:it_dec1st,f] = oct_allmonths_seasonal[it_oct1st,1]
        cansips_seasonal_vars_concat_oct1st[it_dec1st:it_jan1st,f] = oct_allmonths_seasonal[it_oct1st,2]

        cansips_seasonal_vars_concat_nov1st[it_nov1st:it_dec1st,f] = nov_allmonths_seasonal[it_nov1st,0]
        cansips_seasonal_vars_concat_nov1st[it_dec1st:it_jan1st,f] = nov_allmonths_seasonal[it_nov1st,1]

        cansips_seasonal_vars_concat_dec1st[it_dec1st:it_jan1st,f] = dec_allmonths_seasonal[it_dec1st,0]

cansips_seasonal_varnames_concat_sep1st = []
cansips_seasonal_varnames_concat_oct1st = []
cansips_seasonal_varnames_concat_nov1st = []
cansips_seasonal_varnames_concat_dec1st = []
for f,feature in enumerate(feature_list):
    cansips_seasonal_varnames_concat_sep1st += ['Seasonal forecast '+ feature + ' - Start Sep. 1st']
    cansips_seasonal_varnames_concat_oct1st += ['Seasonal forecast '+ feature + ' - Start Oct. 1st']
    cansips_seasonal_varnames_concat_nov1st += ['Seasonal forecast '+ feature + ' - Start Nov. 1st']
    cansips_seasonal_varnames_concat_dec1st += ['Seasonal forecast '+ feature + ' - Start Dec. 1st']


#%%%%%%%%%%% Load CanSIPS monthly forecasts %%%%%%%%%%%

feature_list = [#'WTMP_SFC_0',
                'PRATE_SFC_0',
                'TMP_TGL_2m',
                'PRMSL_MSL_0'
                ]
region_cansips = 'D'
# region_cansips = 'YUL'
anomaly_cansips = True
sort_type = 'startmonth'
lag=1

cansips_monthly_vars_startO = np.zeros((len(time),len(feature_list)*9))*np.nan
cansips_monthly_vars_startN = np.zeros((len(time),len(feature_list)*12))*np.nan
cansips_monthly_vars_startD = np.zeros((len(time),len(feature_list)*14))*np.nan
cansips_monthly_startO_varnames = []
cansips_monthly_startN_varnames = []
cansips_monthly_startD_varnames = []
for f,feature in enumerate(feature_list):
    varname = feature
    cansips_monthly_vars_startO[:,(9*f):9*(f+1)], varnames_startO = get_daily_var_from_monthly_cansips_forecasts(sort_type,1,varname,anomaly_cansips,region_cansips,time,lag=lag)
    cansips_monthly_vars_startN[:,(12*f):12*(f+1)], varnames_startN = get_daily_var_from_monthly_cansips_forecasts(sort_type,2,varname,anomaly_cansips,region_cansips,time,lag=lag)
    cansips_monthly_vars_startD[:,(14*f):14*(f+1)], varnames_startD = get_daily_var_from_monthly_cansips_forecasts(sort_type,3,varname,anomaly_cansips,region_cansips,time,lag=lag)
    cansips_monthly_startO_varnames += [varnames_startO[i] for i in range(len(varnames_startO))]
    cansips_monthly_startN_varnames += [varnames_startN[i] for i in range(len(varnames_startN))]
    cansips_monthly_startD_varnames += [varnames_startD[i] for i in range(len(varnames_startD))]


# Convert variables to one time series per forecast start month
cansips_monthly_vars_concat_sep1st = np.zeros((len(time),len(feature_list)))
cansips_monthly_vars_concat_oct1st = np.zeros((len(time),len(feature_list)))
cansips_monthly_vars_concat_nov1st = np.zeros((len(time),len(feature_list)))
cansips_monthly_vars_concat_dec1st = np.zeros((len(time),len(feature_list)))

for f,feature in enumerate(feature_list):

    lf = int(cansips_monthly_vars_startD.shape[1]/len(feature_list))

    sep_allmonths = cansips_monthly_vars_startD[:,(f*lf)      :(f*lf)+5    ]
    oct_allmonths = cansips_monthly_vars_startD[:,(f*lf)+5    :(f*lf)+5+4  ]
    nov_allmonths = cansips_monthly_vars_startD[:,(f*lf)+5+4  :(f*lf)+5+4+3]
    dec_allmonths = cansips_monthly_vars_startD[:,(f*lf)+5+4+3:(f*lf)+5+4+3+2]

    for year in range(date_start.year,date_end.year+1):

        it_sep1st = np.where(time == (dt.date(year, 9,1)-date_ref).days)[0][0]
        it_oct1st = np.where(time == (dt.date(year,10,1)-date_ref).days)[0][0]
        it_nov1st = np.where(time == (dt.date(year,11,1)-date_ref).days)[0][0]
        it_dec1st = np.where(time == (dt.date(year,12,1)-date_ref).days)[0][0]
        if year < years[-1]:
            it_jan1st = np.where(time == (dt.date(year+1,1,1)-date_ref).days)[0][0]
            it_feb1st = np.where(time == (dt.date(year+1,2,1)-date_ref).days)[0][0]
        else:
            it_jan1st = len(time)
            it_feb1st = len(time)

        cansips_monthly_vars_concat_sep1st[it_sep1st:it_oct1st,f] = sep_allmonths[it_sep1st,0]
        cansips_monthly_vars_concat_sep1st[it_oct1st:it_nov1st,f] = sep_allmonths[it_sep1st,1]
        cansips_monthly_vars_concat_sep1st[it_nov1st:it_dec1st,f] = sep_allmonths[it_sep1st,2]
        cansips_monthly_vars_concat_sep1st[it_dec1st:it_jan1st,f] = sep_allmonths[it_sep1st,3]
        cansips_monthly_vars_concat_sep1st[it_jan1st:it_feb1st,f] = sep_allmonths[it_sep1st,4]

        cansips_monthly_vars_concat_oct1st[it_oct1st:it_nov1st,f] = oct_allmonths[it_oct1st,0]
        cansips_monthly_vars_concat_oct1st[it_nov1st:it_dec1st,f] = oct_allmonths[it_oct1st,1]
        cansips_monthly_vars_concat_oct1st[it_dec1st:it_jan1st,f] = oct_allmonths[it_oct1st,2]
        cansips_monthly_vars_concat_oct1st[it_jan1st:it_feb1st,f] = oct_allmonths[it_oct1st,3]

        cansips_monthly_vars_concat_nov1st[it_nov1st:it_dec1st,f] = nov_allmonths[it_nov1st,0]
        cansips_monthly_vars_concat_nov1st[it_dec1st:it_jan1st,f] = nov_allmonths[it_nov1st,1]
        cansips_monthly_vars_concat_nov1st[it_jan1st:it_feb1st,f] = nov_allmonths[it_nov1st,2]

        cansips_monthly_vars_concat_dec1st[it_dec1st:it_jan1st,f] = dec_allmonths[it_dec1st,0]
        cansips_monthly_vars_concat_dec1st[it_jan1st:it_feb1st,f] = dec_allmonths[it_dec1st,1]


cansips_monthly_varnames_concat_sep1st = []
cansips_monthly_varnames_concat_oct1st = []
cansips_monthly_varnames_concat_nov1st = []
cansips_monthly_varnames_concat_dec1st = []
for f,feature in enumerate(feature_list):
    cansips_monthly_varnames_concat_sep1st += ['Monthly forecast '+ feature + ' - Start Sep. 1st']
    cansips_monthly_varnames_concat_oct1st += ['Monthly forecast '+ feature + ' - Start Oct. 1st']
    cansips_monthly_varnames_concat_nov1st += ['Monthly forecast '+ feature + ' - Start Nov. 1st']
    cansips_monthly_varnames_concat_dec1st += ['Monthly forecast '+ feature + ' - Start Dec. 1st']

#%%%%%%% SAVE DATSET FOR ML MODELS %%%%%%%%%
if save_ML:
    ntot = (avg_Twater_vars.shape[1]+
           weather_vars.shape[1]+
           # NAO_daily_vars.shape[1]+
           # PDO_vars.shape[1]+
           # ENSO_vars.shape[1]+
           ci_vars.shape[1]+
           discharge_vars.shape[1]+
           level_vars.shape[1]+
           levelO_vars.shape[1]+
           cansips_monthly_vars_concat_sep1st.shape[1]+
           cansips_monthly_vars_concat_oct1st.shape[1]+
           cansips_monthly_vars_concat_nov1st.shape[1]+
           cansips_monthly_vars_concat_dec1st.shape[1]+
           cansips_seasonal_vars_concat_sep1st.shape[1]+
           cansips_seasonal_vars_concat_oct1st.shape[1]+
           cansips_seasonal_vars_concat_nov1st.shape[1]+
           cansips_seasonal_vars_concat_dec1st.shape[1]
           )

    ds = np.zeros((len(time),ntot+1))
    labels = np.chararray(ntot+1, itemsize=100)
    save_name = 'ML_dataset_with_cansips'

    # Add time as first column:
    ds[:,0] = time
    labels[0] = 'Days since '+str(date_ref)

    # Add water temperature time series:
    ds[:,1] = avg_Twater_vars[:,0]
    labels[1] = avg_Twater_varnames[0]

    # Add weather data:
    for i in range(weather_vars.shape[1]):
        ds[:,2+i] = weather_vars[:,i]
        labels[2+i] = weather_varnames[i]

    # Add discharge and water level data:
    ds[:,2+weather_vars.shape[1]] = discharge_vars[:,0]
    labels[2+weather_vars.shape[1]] = discharge_varnames[0]

    ds[:,3+weather_vars.shape[1]] = level_vars[:,0]
    labels[3+weather_vars.shape[1]] = level_varnames[0]

    ds[:,4+weather_vars.shape[1]] = levelO_vars[:,0]
    labels[4+weather_vars.shape[1]] = levelO_varnames[0]

    # #Add NAO index:
    # ds[:,5+weather_vars.shape[1]] = NAO_daily_vars[:,0]
    # labels[5+weather_vars.shape[1]] = NAO_daily_varnames[0]

    # #Add PDO index:
    # ds[:,6+weather_vars.shape[1]] = PDO_vars[:,0]
    # labels[6+weather_vars.shape[1]] = PDO_varnames[0]

    # #Add ENSO index:
    # ds[:,7+weather_vars.shape[1]] = ENSO_vars[:,0]
    # labels[7+weather_vars.shape[1]] = ENSO_varnames[0]

    # Add Climate Indices
    for i in range(ci_vars.shape[1]):
        ds[:,5+weather_vars.shape[1]+i] = ci_vars[:,i]
        labels[5+weather_vars.shape[1]+i] = ci_varnames[i]

    # Add CanSIPS Monthly Forecasts
    for i in range(cansips_monthly_vars_concat_sep1st.shape[1]):
        n=5+weather_vars.shape[1]+ci_vars.shape[1]
        ds[:,n+i] = cansips_monthly_vars_concat_sep1st[:,i]
        labels[n+i] = cansips_monthly_varnames_concat_sep1st[i]

    for i in range(cansips_monthly_vars_concat_oct1st.shape[1]):
        n=5+weather_vars.shape[1]+ci_vars.shape[1]+cansips_monthly_vars_concat_sep1st.shape[1]
        ds[:,n+i] = cansips_monthly_vars_concat_oct1st[:,i]
        labels[n+i] = cansips_monthly_varnames_concat_oct1st[i]

    for i in range(cansips_monthly_vars_concat_nov1st.shape[1]):
        n=5+weather_vars.shape[1]+ci_vars.shape[1]+cansips_monthly_vars_concat_sep1st.shape[1]+cansips_monthly_vars_concat_oct1st.shape[1]
        ds[:,n+i] = cansips_monthly_vars_concat_nov1st[:,i]
        labels[n+i] = cansips_monthly_varnames_concat_nov1st[i]

    for i in range(cansips_monthly_vars_concat_dec1st.shape[1]):
        n=5+weather_vars.shape[1]+ci_vars.shape[1]+cansips_monthly_vars_concat_sep1st.shape[1]+cansips_monthly_vars_concat_oct1st.shape[1]+cansips_monthly_vars_concat_nov1st.shape[1]
        ds[:,n+i] = cansips_monthly_vars_concat_dec1st[:,i]
        labels[n+i] = cansips_monthly_varnames_concat_dec1st[i]


    # Add CanSIPS Seasonal Forecasts
    for i in range(cansips_seasonal_vars_concat_sep1st.shape[1]):
        n=5+weather_vars.shape[1]+ci_vars.shape[1]+cansips_monthly_vars_concat_sep1st.shape[1]+cansips_monthly_vars_concat_oct1st.shape[1]+cansips_monthly_vars_concat_nov1st.shape[1]+cansips_monthly_vars_concat_dec1st.shape[1]
        ds[:,n+i] = cansips_seasonal_vars_concat_sep1st[:,i]
        labels[n+i] = cansips_seasonal_varnames_concat_sep1st[i]

    for i in range(cansips_seasonal_vars_concat_oct1st.shape[1]):
        n=5+weather_vars.shape[1]+ci_vars.shape[1]+cansips_monthly_vars_concat_sep1st.shape[1]+cansips_monthly_vars_concat_oct1st.shape[1]+cansips_monthly_vars_concat_nov1st.shape[1]+cansips_monthly_vars_concat_dec1st.shape[1]+cansips_seasonal_vars_concat_sep1st.shape[1]
        ds[:,n+i] = cansips_seasonal_vars_concat_oct1st[:,i]
        labels[n+i] = cansips_seasonal_varnames_concat_oct1st[i]

    for i in range(cansips_seasonal_vars_concat_nov1st.shape[1]):
        n=5+weather_vars.shape[1]+ci_vars.shape[1]+cansips_monthly_vars_concat_sep1st.shape[1]+cansips_monthly_vars_concat_oct1st.shape[1]+cansips_monthly_vars_concat_nov1st.shape[1]+cansips_monthly_vars_concat_dec1st.shape[1]+cansips_seasonal_vars_concat_sep1st.shape[1]+cansips_seasonal_vars_concat_oct1st.shape[1]
        ds[:,n+i] = cansips_seasonal_vars_concat_nov1st[:,i]
        labels[n+i] = cansips_seasonal_varnames_concat_nov1st[i]

    for i in range(cansips_seasonal_vars_concat_dec1st.shape[1]):
        n=5+weather_vars.shape[1]+ci_vars.shape[1]+cansips_monthly_vars_concat_sep1st.shape[1]+cansips_monthly_vars_concat_oct1st.shape[1]+cansips_monthly_vars_concat_nov1st.shape[1]+cansips_monthly_vars_concat_dec1st.shape[1]+cansips_seasonal_vars_concat_sep1st.shape[1]+cansips_seasonal_vars_concat_oct1st.shape[1]+cansips_seasonal_vars_concat_nov1st.shape[1]
        ds[:,n+i] = cansips_seasonal_vars_concat_dec1st[:,i]
        labels[n+i] = cansips_seasonal_varnames_concat_dec1st[i]

    # SAVE:
    np.savez('../../../data/ML_timeseries/'+save_name,
              data=ds,
              labels=labels,
              region_ERA5 = region,
              region_cansips = region_cansips,
              Twater_loc_list = Twater_loc_list,
              loc_discharge = loc_discharge,
              loc_level = loc_level,
              loc_levelO = loc_levelO,
              date_ref=date_ref)



#%%%%%%% GET MONTHLY VARIABLES %%%%%%%%%
# Average for Jan, Feb, Mar, April,...., Dec
monthly_vars_tmp = np.zeros((100,len(years),12))*np.nan
monthly_vars_in =['weather_vars',
                  'avg_Twater_vars',
                  # 'NAO_daily_vars',
                  # 'PDO_vars',
                  # 'ENSO_vars',
                  'ci_vars',
                  'discharge_vars',
                  'level_vars',
                  'levelO_vars'
                  ]

nvars_monthly = 0
varnames_monthly_i = []
for i,var in enumerate(monthly_vars_in):
    if var == 'weather_vars':
        monthly_weather_vars = get_monthly_vars_from_daily(weather_vars,weather_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_weather_vars.shape[0],:,:] = monthly_weather_vars
        nvars_monthly += monthly_weather_vars.shape[0]
        varnames_monthly_i += [weather_varnames[i] for i in range(len(weather_varnames))]
    if var == 'avg_Twater_vars':
        monthly_avg_Twater_vars = get_monthly_vars_from_daily(avg_Twater_vars,avg_Twater_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_avg_Twater_vars.shape[0],:,:] = monthly_avg_Twater_vars
        nvars_monthly += monthly_avg_Twater_vars.shape[0]
        varnames_monthly_i += avg_Twater_varnames
    # if var == 'NAO_daily_vars':
    #     monthly_NAO_daily_vars = get_monthly_vars_from_daily(NAO_daily_vars,NAO_daily_varnames,years,time,replace_with_nan)
    #     monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_NAO_daily_vars.shape[0],:,:] = monthly_NAO_daily_vars
    #     nvars_monthly += monthly_NAO_daily_vars.shape[0]
    #     varnames_monthly_i += NAO_daily_varnames
    # if var == 'PDO_vars':
    #     monthly_PDO_vars = get_monthly_vars_from_daily(PDO_vars,PDO_varnames,years,time,replace_with_nan)
    #     monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_PDO_vars.shape[0],:,:] = monthly_PDO_vars
    #     nvars_monthly += monthly_PDO_vars.shape[0]
    #     varnames_monthly_i += PDO_varnames
    # if var == 'ENSO_vars':
    #     monthly_ENSO_vars = get_monthly_vars_from_daily(ENSO_vars,ENSO_varnames,years,time,replace_with_nan)
    #     monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_ENSO_vars.shape[0],:,:] = monthly_ENSO_vars
    #     nvars_monthly += monthly_ENSO_vars.shape[0]
    #     varnames_monthly_i += ENSO_varnames
    if var == 'ci_vars':
        monthly_ci_vars = get_monthly_vars_from_daily(ci_vars,['Avg.' + index_list[j] for j in range(len(index_list)) ],years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_ci_vars.shape[0],:,:] = monthly_ci_vars
        nvars_monthly += monthly_ci_vars.shape[0]
        varnames_monthly_i += ci_varnames
    if var == 'discharge_vars':
        monthly_discharge_vars = get_monthly_vars_from_daily(discharge_vars,discharge_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_discharge_vars.shape[0],:,:] = monthly_discharge_vars
        nvars_monthly += monthly_discharge_vars.shape[0]
        varnames_monthly_i += discharge_varnames
    if var == 'level_vars':
        monthly_level_vars = get_monthly_vars_from_daily(level_vars,level_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_level_vars.shape[0],:,:] = monthly_level_vars
        nvars_monthly += monthly_level_vars.shape[0]
        varnames_monthly_i += level_varnames
    if var == 'levelO_vars':
        monthly_levelO_vars = get_monthly_vars_from_daily(levelO_vars,levelO_varnames,years,time,replace_with_nan)
        monthly_vars_tmp[nvars_monthly:nvars_monthly+monthly_levelO_vars.shape[0],:,:] = monthly_levelO_vars
        nvars_monthly += monthly_levelO_vars.shape[0]
        varnames_monthly_i += levelO_varnames


monthly_vars_tmp = monthly_vars_tmp[0:nvars_monthly,:,:]

save_monthly_vars = True
if save_monthly_vars:
    save_name = 'monthly_vars'
    np.savez('../../../data/monthly_predictors/'+save_name,
              data=monthly_vars_tmp,
              labels=varnames_monthly_i,
              region_ERA5 = region,
              region_cansips = region_cansips,
              Twater_loc_list = Twater_loc_list,
              loc_discharge = loc_discharge,
              loc_level = loc_level,
              loc_levelO = loc_levelO,
              date_ref=date_ref)

# Repeat arr for all start dates and remove the months that are incomplete, or after start date:
monthly_vars_i = np.repeat(monthly_vars_tmp[:, :, np.newaxis,:], len(start_doy_arr), axis=2)

for istart,start_doy in enumerate(start_doy_arr):
    month_max = (dt.timedelta(days=int(start_doy-1)) + dt.date(1991,1,1)).month
    imonth_max = month_max-1
    monthly_vars_i[:,:,istart,imonth_max:] = np.nan

#%%
# Add CanSIPS variables to the appropriate start dates
monthly_vars = np.zeros((nvars_monthly+56+40,monthly_vars_i.shape[1],monthly_vars_i.shape[2],monthly_vars_i.shape[3]))*np.nan
varnames_monthly = np.zeros((len(start_doy_arr),nvars_monthly+56+40),dtype='object')
for istart,start_doy in enumerate(start_doy_arr):
    month_max = (dt.timedelta(days=int(start_doy-1)) + dt.date(1991,1,1)).month
    imonth_max = month_max-1
    monthly_vars[0:nvars_monthly,:,istart,:] = monthly_vars_i[:,:,istart,:]
    varnames_monthly[istart,0:nvars_monthly] = np.array(varnames_monthly_i)
    if month_max == 10: # add all October variables
        dm = cansips_monthly_vars_startO
        vn_monthly = ['Avg. ' + cansips_monthly_startO_varnames[j] for j in range(dm.shape[1])]
        monthly_tmp = get_monthly_vars_from_daily(dm,vn_monthly,years,time,replace_with_nan)
        nvars_monthly_tmp = dm.shape[1]
        monthly_vars[nvars_monthly:nvars_monthly+nvars_monthly_tmp,:,istart,:] = monthly_tmp
        varnames_monthly_tmp = [vn_monthly[t] for t in range(nvars_monthly_tmp)]
        varnames_monthly[istart,nvars_monthly:nvars_monthly+nvars_monthly_tmp] = np.array(varnames_monthly_tmp)

        ds = cansips_seasonal_vars_startO
        vn_seasonal = ['Avg. ' + cansips_seasonal_startO_varnames[j] for j in range(ds.shape[1])]
        seasonal_tmp = get_monthly_vars_from_daily(ds,vn_seasonal,years,time,replace_with_nan)
        nvars_seasonal_tmp = ds.shape[1]
        monthly_vars[nvars_monthly+nvars_monthly_tmp:nvars_monthly+nvars_monthly_tmp+nvars_seasonal_tmp,:,istart,:] = seasonal_tmp
        varnames_seasonal_tmp = [vn_seasonal[t] for t in range(nvars_seasonal_tmp)]
        varnames_monthly[istart,nvars_monthly+nvars_monthly_tmp:nvars_monthly+nvars_monthly_tmp+nvars_seasonal_tmp] = np.array(varnames_seasonal_tmp)

    if month_max == 11: # add all November variables
        dm = cansips_monthly_vars_startN
        vn_monthly = ['Avg. ' + cansips_monthly_startN_varnames[j] for j in range(dm.shape[1])]
        monthly_tmp = get_monthly_vars_from_daily(dm,vn_monthly,years,time,replace_with_nan)
        nvars_monthly_tmp = dm.shape[1]
        monthly_vars[nvars_monthly:nvars_monthly+nvars_monthly_tmp,:,istart,:] = monthly_tmp
        varnames_monthly_tmp = [vn_monthly[t] for t in range(nvars_monthly_tmp)]
        varnames_monthly[istart,nvars_monthly:nvars_monthly+nvars_monthly_tmp] = np.array(varnames_monthly_tmp)

        ds = cansips_seasonal_vars_startN
        vn_seasonal = ['Avg. ' + cansips_seasonal_startN_varnames[j] for j in range(ds.shape[1])]
        seasonal_tmp = get_monthly_vars_from_daily(ds,vn_seasonal,years,time,replace_with_nan)
        nvars_seasonal_tmp = ds.shape[1]
        monthly_vars[nvars_monthly+nvars_monthly_tmp:nvars_monthly+nvars_monthly_tmp+nvars_seasonal_tmp,:,istart,:] = seasonal_tmp
        varnames_seasonal_tmp = [vn_seasonal[t] for t in range(nvars_seasonal_tmp)]
        varnames_monthly[istart,nvars_monthly+nvars_monthly_tmp:nvars_monthly+nvars_monthly_tmp+nvars_seasonal_tmp] = np.array(varnames_seasonal_tmp)

    if month_max == 12: # add all December variables
        dm = cansips_monthly_vars_startD
        vn_monthly = ['Avg. ' + cansips_monthly_startD_varnames[j] for j in range(dm.shape[1])]
        monthly_tmp = get_monthly_vars_from_daily(dm,vn_monthly,years,time,replace_with_nan)
        nvars_monthly_tmp = dm.shape[1]
        monthly_vars[nvars_monthly:nvars_monthly+nvars_monthly_tmp,:,istart,:] = monthly_tmp
        varnames_monthly_tmp = [vn_monthly[t] for t in range(nvars_monthly_tmp)]
        varnames_monthly[istart,nvars_monthly:nvars_monthly+nvars_monthly_tmp] = np.array(varnames_monthly_tmp)

        ds = cansips_seasonal_vars_startD
        vn_seasonal = ['Avg. ' + cansips_seasonal_startD_varnames[j] for j in range(ds.shape[1])]
        seasonal_tmp = get_monthly_vars_from_daily(ds,vn_seasonal,years,time,replace_with_nan)
        nvars_seasonal_tmp = ds.shape[1]
        monthly_vars[nvars_monthly+nvars_monthly_tmp:nvars_monthly+nvars_monthly_tmp+nvars_seasonal_tmp,:,istart,:] = seasonal_tmp
        varnames_seasonal_tmp = [vn_seasonal[t] for t in range(nvars_seasonal_tmp)]
        varnames_monthly[istart,nvars_monthly+nvars_monthly_tmp:nvars_monthly+nvars_monthly_tmp+nvars_seasonal_tmp] = np.array(varnames_seasonal_tmp)

nvars_monthly_n = monthly_vars.shape[0]

#%%%%%%% GET 3-MONTH VARIABLES %%%%%%%%%
#average for # -, -, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND
threemonthly_vars_tmp = np.zeros((100,len(years),12))*np.nan
threemonthly_vars_in =['weather_vars',
                       'avg_Twater_vars',
                       # 'NAO_daily_vars',
                       # 'PDO_vars',
                       # 'ENSO_vars',
                       'ci_vars',
                       'discharge_vars',
                       'level_vars',
                       'levelO_vars']

nvars_threemonthly = 0
varnames_threemonthly = []
for i,var in enumerate(threemonthly_vars_in):
    if var == 'weather_vars':
        threemonthly_weather_vars = get_3month_vars_from_daily(weather_vars,weather_varnames,years,time,replace_with_nan)
        threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_weather_vars.shape[0],:,:] = threemonthly_weather_vars
        nvars_threemonthly += threemonthly_weather_vars.shape[0]
        varnames_threemonthly += [weather_varnames[i] for i in range(len(weather_varnames))]
    if var == 'avg_Twater_vars':
        threemonthly_avg_Twater_vars = get_3month_vars_from_daily(avg_Twater_vars,avg_Twater_varnames,years,time,replace_with_nan)
        threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_avg_Twater_vars.shape[0],:,:] = threemonthly_avg_Twater_vars
        nvars_threemonthly += threemonthly_avg_Twater_vars.shape[0]
        varnames_threemonthly += avg_Twater_varnames
    # if var == 'NAO_daily_vars':
    #     threemonthly_NAO_daily_vars = get_3month_vars_from_daily(NAO_daily_vars,NAO_daily_varnames,years,time,replace_with_nan)
    #     threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_NAO_daily_vars.shape[0],:,:] = threemonthly_NAO_daily_vars
    #     nvars_threemonthly += threemonthly_NAO_daily_vars.shape[0]
    #     varnames_threemonthly += NAO_daily_varnames
    # if var == 'PDO_vars':
    #     threemonthly_PDO_vars = get_3month_vars_from_daily(PDO_vars,PDO_varnames,years,time,replace_with_nan)
    #     threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_PDO_vars.shape[0],:,:] = threemonthly_PDO_vars
    #     nvars_threemonthly += threemonthly_PDO_vars.shape[0]
    #     varnames_threemonthly += PDO_varnames
    # if var == 'ENSO_vars':
    #     threemonthly_ENSO_vars = get_3month_vars_from_daily(ENSO_vars,ENSO_varnames,years,time,replace_with_nan)
    #     threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_ENSO_vars.shape[0],:,:] = threemonthly_ENSO_vars
    #     nvars_threemonthly += threemonthly_ENSO_vars.shape[0]
    #     varnames_threemonthly += ENSO_varnames
    if var == 'ci_vars':
        threemonthly_ci_vars = get_3month_vars_from_daily(ci_vars,['Avg.' + index_list[j] for j in range(len(index_list)) ],years,time,replace_with_nan)
        threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_ci_vars.shape[0],:,:] = threemonthly_ci_vars
        nvars_threemonthly += threemonthly_ci_vars.shape[0]
        varnames_threemonthly += ci_varnames
    if var == 'discharge_vars':
        threemonthly_discharge_vars = get_3month_vars_from_daily(discharge_vars,discharge_varnames,years,time,replace_with_nan)
        threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_discharge_vars.shape[0],:,:] = threemonthly_discharge_vars
        nvars_threemonthly += threemonthly_discharge_vars.shape[0]
        varnames_threemonthly += discharge_varnames
    if var == 'level_vars':
        threemonthly_level_vars = get_3month_vars_from_daily(level_vars,level_varnames,years,time,replace_with_nan)
        threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_level_vars.shape[0],:,:] = threemonthly_level_vars
        nvars_threemonthly += threemonthly_level_vars.shape[0]
        varnames_threemonthly += level_varnames
    if var == 'levelO_vars':
        threemonthly_levelO_vars = get_3month_vars_from_daily(levelO_vars,levelO_varnames,years,time,replace_with_nan)
        threemonthly_vars_tmp[nvars_threemonthly:nvars_threemonthly+threemonthly_levelO_vars.shape[0],:,:] = threemonthly_levelO_vars
        nvars_threemonthly += threemonthly_levelO_vars.shape[0]
        varnames_threemonthly += levelO_varnames

threemonthly_vars_tmp = threemonthly_vars_tmp[0:nvars_threemonthly,:,:]

# Repeat arr for all start dates and remove the months that are incomplete, or after start date:
threemonthly_vars = np.repeat(threemonthly_vars_tmp[:, :, np.newaxis,:], len(start_doy_arr), axis=2)

for istart,start_doy in enumerate(start_doy_arr):
    month_max = (dt.timedelta(days=int(start_doy-1)) + dt.date(1991,1,1)).month
    imonth_max = month_max-1
    threemonthly_vars[:,:,istart,imonth_max:] = np.nan



#%%%%%%% GET 30-DAY ROLLING WINDOW VARIABLES %%%%%%%%%
# Average variables in 30-day windows that move backwards every 7 days from
# forecast start_date (i.e. end_dates_arr)
ws = 30
ns = 7

nwindows_max = []
for i in range(len(start_doy_arr)):
    nwindows_max += [np.max([int(np.floor(((start_doy_arr[i]-1)-ws)/ns))+1, int(np.floor((start_doy_arr[i]-ws)/ns))+1])]
nwindows_max = np.max(nwindows_max)

window_vars = np.zeros((23,len(years),len(start_doy_arr),nwindows_max))*np.nan
window_vars_in =['weather_vars',
                 'avg_Twater_vars',
                 # 'NAO_daily_vars',
                 'discharge_vars',
                 'level_vars',
                 'levelO_vars']

nvars_window = 0
varnames_window = []
for i,var in enumerate(window_vars_in):
    if var == 'weather_vars':
        window_weather_vars = np.zeros((weather_vars.shape[1],len(years),len(start_doy_arr),nwindows_max))*np.nan
        for istart,start_doy in enumerate(start_doy_arr):
            nw = np.max([int(np.floor(((start_doy_arr[istart]-1)-ws)/ns))+1, int(np.floor((start_doy_arr[istart]-ws)/ns))+1])
            window_weather_vars[:,:,istart,0:nw] = get_rollingwindow_vars_from_daily(weather_vars,weather_varnames,start_doy,ws,ns,years,time,replace_with_nan)

        window_vars[nvars_window:nvars_window+window_weather_vars.shape[0],:,:,:] = window_weather_vars
        nvars_window += window_weather_vars.shape[0]
        varnames_window += [weather_varnames[i] for i in range(len(weather_varnames))]

    if var == 'avg_Twater_vars':
        window_avg_Twater_vars = np.zeros((avg_Twater_vars.shape[1],len(years),len(start_doy_arr),nwindows_max))*np.nan
        for istart,start_doy in enumerate(start_doy_arr):
            nw = np.max([int(np.floor(((start_doy_arr[istart]-1)-ws)/ns))+1, int(np.floor((start_doy_arr[istart]-ws)/ns))+1])
            window_avg_Twater_vars[:,:,istart,0:nw] = get_rollingwindow_vars_from_daily(avg_Twater_vars,avg_Twater_varnames,start_doy,ws,ns,years,time,replace_with_nan)

        window_vars[nvars_window:nvars_window+window_avg_Twater_vars.shape[0],:,:] = window_avg_Twater_vars
        nvars_window += window_avg_Twater_vars.shape[0]
        varnames_window += avg_Twater_varnames
    # if var == 'NAO_daily_vars':
    #     window_NAO_daily_vars = np.zeros((NAO_daily_vars.shape[1],len(years),len(start_doy_arr),nwindows_max))*np.nan
    #     for istart,start_doy in enumerate(start_doy_arr):
    #         nw = np.max([int(np.floor(((start_doy_arr[istart]-1)-ws)/ns))+1, int(np.floor((start_doy_arr[istart]-ws)/ns))+1])
    #         window_NAO_daily_vars[:,:,istart,0:nw] = get_rollingwindow_vars_from_daily(NAO_daily_vars,NAO_daily_varnames,start_doy,ws,ns,years,time,replace_with_nan)

    #     window_vars[nvars_window:nvars_window+window_NAO_daily_vars.shape[0],:,:] = window_NAO_daily_vars
    #     nvars_window += window_NAO_daily_vars.shape[0]
    #     varnames_window += NAO_daily_varnames
    if var == 'discharge_vars':
        window_discharge_vars = np.zeros((discharge_vars.shape[1],len(years),len(start_doy_arr),nwindows_max))*np.nan
        for istart,start_doy in enumerate(start_doy_arr):
            nw = np.max([int(np.floor(((start_doy_arr[istart]-1)-ws)/ns))+1, int(np.floor((start_doy_arr[istart]-ws)/ns))+1])
            window_discharge_vars[:,:,istart,0:nw] = get_rollingwindow_vars_from_daily(discharge_vars,discharge_varnames,start_doy,ws,ns,years,time,replace_with_nan)

        window_vars[nvars_window:nvars_window+window_discharge_vars.shape[0],:,:] = window_discharge_vars
        nvars_window += window_discharge_vars.shape[0]
        varnames_window += discharge_varnames
    if var == 'level_vars':
        window_level_vars = np.zeros((level_vars.shape[1],len(years),len(start_doy_arr),nwindows_max))*np.nan
        for istart,start_doy in enumerate(start_doy_arr):
            nw = np.max([int(np.floor(((start_doy_arr[istart]-1)-ws)/ns))+1, int(np.floor((start_doy_arr[istart]-ws)/ns))+1])
            window_level_vars[:,:,istart,0:nw] = get_rollingwindow_vars_from_daily(level_vars,level_varnames,start_doy,ws,ns,years,time,replace_with_nan)

        window_vars[nvars_window:nvars_window+window_level_vars.shape[0],:,:] = window_level_vars
        nvars_window += window_level_vars.shape[0]
        varnames_window += level_varnames
    if var == 'levelO_vars':
        window_levelO_vars = np.zeros((levelO_vars.shape[1],len(years),len(start_doy_arr),nwindows_max))*np.nan
        for istart,start_doy in enumerate(start_doy_arr):
            nw = np.max([int(np.floor(((start_doy_arr[istart]-1)-ws)/ns))+1, int(np.floor((start_doy_arr[istart]-ws)/ns))+1])
            window_levelO_vars[:,:,istart,0:nw] = get_rollingwindow_vars_from_daily(levelO_vars,levelO_varnames,start_doy,ws,ns,years,time,replace_with_nan)

        window_vars[nvars_window:nvars_window+window_levelO_vars.shape[0],:,:] = window_levelO_vars
        nvars_window += window_levelO_vars.shape[0]
        varnames_window += levelO_varnames

window_vars = window_vars[0:nvars_window,:,:]


#%%%%%%% KEEP PREDICTORS ONLY IF AVAILABLE EVERY YEAR %%%%%%%%%
# *** NOTE: THIS IS NOW DONE WHEN DOING THE DATA SET CORRELATION WITH FUD
# AND USING THE 'min_periods' KEYWORD TO SET A MINIMUM NUMBER OF
# AVAILABLE DATA TO CONSIDER THE CORRELATION

# monthly_vars_select = monthly_vars.copy()

# tol = 0.30 # If percentage of years that have nan values or larger than 'tol', then that variable is removed from the analysis

# # Check monthly_vars
# for ivar in range(nvars_monthly_n):
#     for iw in range(12):
#         for istart in range(len(start_doy_arr)):
#             if np.sum(np.isnan(monthly_vars_select[ivar,:,istart,iw]))/len(avg_freezeup_doy[~np.isnan(avg_freezeup_doy)]) > tol:
#                 monthly_vars_select[ivar,:,istart,iw] = np.nan
#                 if iw < 10: print(varnames_monthly[ivar] + '('+str(istart)+', '+str(iw)+') - Removed!')

#%%%%%%% SPLIT VARIABLES INTO TRAIN/VALID SETS %%%%%%%%%
# HERE THERE ARE THREE VALIDATION SCHEME OPTIONS:
# OPTION 0: USE NTRAINING FOR TRAINING, THEN VALID ON THE NEXT NVALID YEARS
#            TTTTTTTTTTTTTTTTTT-VVVVVVVVVV
# OPTION 1: START WITH NTRAINING, VALID ONE YEAR; ADD 1 YR TO NTRAINING, VALID NEXT YEARS, ETC.
#            TTTTTTTTTTTTTTTTTT-V
#            TTTTTTTTTTTTTTTTTTT-V
#            TTTTTTTTTTTTTTTTTTTT-V
#                                 ...
# OPTION 2: USE NTRAINING FOR TRAINING, VALID ONE YEAR; OFFSET BY ONE YEAR AND USE NEXT NTRAINING FOR TRAINING, VALID NEXT YEAR; ETC.
#            TTTTTTTTTTTTTTTTTTT-V
#             TTTTTTTTTTTTTTTTTTT-V
#              TTTTTTTTTTTTTTTTTTT-V
#                                  ...



# ntraining_list = [12,14,16,18,20,np.where(~np.isnan(avg_freezeup_doy[1:]))[0][-1]-1]
ntraining_list = [18]
# ntraining_list = [12,13,14,15,16,17,18,19,20,np.where(~np.isnan(avg_freezeup_doy[1:]))[0][-1]-1]

MAE_valid_end = np.zeros((len(ntraining_list),3,6))*np.nan
RMSE_valid_end = np.zeros((len(ntraining_list),3,6))*np.nan
week_accuracy_valid_end  = np.zeros((len(ntraining_list),3,6))*np.nan
rsqr_valid_end = np.zeros((len(ntraining_list),3,6))*np.nan
pval_valid_end = np.zeros((len(ntraining_list),3,6))*np.nan
nFUD = np.where(~np.isnan(avg_freezeup_doy[1:]))[0][-1] +1

for it, ntraining in enumerate(ntraining_list):

    for valid_opt in range(1):
        if valid_opt == 0:

            nvalid = nFUD-ntraining
            ntraining_tot = ntraining
        if valid_opt == 1:
            nvalid = 1
            nsets = int(np.floor((nFUD-ntraining)/nvalid))
            ntraining_tot = ntraining+nsets-1
        if valid_opt == 2:
            nvalid = 1
            ntraining_tot = ntraining

        nsets = int(np.floor((nFUD-ntraining)/nvalid))

        years_train = np.zeros((ntraining_tot,nsets))
        avg_freezeup_doy_train = np.zeros((ntraining_tot,nsets))
        monthly_vars_train = np.zeros((nvars_monthly_n, ntraining_tot, len(start_doy_arr), monthly_vars.shape[3], nsets ))*np.nan
        threemonthly_vars_train = np.zeros((nvars_threemonthly, ntraining_tot, len(start_doy_arr), threemonthly_vars.shape[3], nsets ))*np.nan
        window_vars_train = np.zeros((nvars_window, ntraining_tot, len(start_doy_arr), window_vars.shape[3], nsets ))*np.nan

        years_valid = np.zeros((nvalid,nsets))
        avg_freezeup_doy_valid = np.zeros((nvalid,nsets))
        monthly_vars_valid = np.zeros((nvars_monthly_n, nvalid, len(start_doy_arr), monthly_vars.shape[3], nsets ))*np.nan
        threemonthly_vars_valid = np.zeros((nvars_threemonthly, nvalid, len(start_doy_arr), threemonthly_vars.shape[3], nsets ))*np.nan
        window_vars_valid = np.zeros((nvars_window, nvalid, len(start_doy_arr), window_vars.shape[3], nsets ))*np.nan

        for s in range(nsets):

            if valid_opt == 1:
                it0 = 1 # +1 because 1991 is nan anyway so we start with 1992.
                it1 = s+1+ntraining
                years_train[0:it1-1,s] = years[it0:it1].copy()
                avg_freezeup_doy_train[0:it1-1,s] = avg_freezeup_doy[it0:it1].copy()
                monthly_vars_train[:,0:it1-1,:,:,s] = monthly_vars[:,it0:it1,:,:].copy()
                threemonthly_vars_train[:,0:it1-1,:,:,s] = threemonthly_vars[:,it0:it1,:,:].copy()
                window_vars_train[:,0:it1-1,:,:,s] = window_vars[:,it0:it1,:,:].copy()
            else:
                it0 = s+1 # +1 because 1991 is nan anyway so we start with 1992.
                it1 = it0+ntraining
                years_train[:,s] = years[it0:it1].copy()
                avg_freezeup_doy_train[:,s] = avg_freezeup_doy[it0:it1].copy()
                monthly_vars_train[:,:,:,:,s] = monthly_vars[:,it0:it1,:,:].copy()
                threemonthly_vars_train[:,:,:,:,s] = threemonthly_vars[:,it0:it1,:,:].copy()
                window_vars_train[:,:,:,:,s] = window_vars[:,it0:it1,:,:].copy()

            iv0 = it1
            iv1 = iv0 + nvalid
            years_valid[:,s] = years[iv0:iv1].copy()
            avg_freezeup_doy_valid[:,s] = avg_freezeup_doy[iv0:iv1].copy()
            monthly_vars_valid[:,:,:,:,s] = monthly_vars[:,iv0:iv1,:,:].copy()
            threemonthly_vars_valid[:,:,:,:,s] = threemonthly_vars[:,iv0:iv1,:,:].copy()
            window_vars_valid[:,:,:,:,s] = window_vars[:,iv0:iv1,:,:].copy()


        ##%%%%%%% DETREND VARIABLES %%%%%%%%%
        # USING TRAINING PERIODS TO COMPUTE TREND FOR EACH SET

        for s in range(nsets):
            # Detrend FUD
            if detrend_FUD:
                if anomaly == 'linear':
                    avg_freezeup_doy_train[:,s], [m,b] = detrend_ts(avg_freezeup_doy_train[:,s],years_train[:,s],anomaly)
                    avg_freezeup_doy_valid[:,s] = avg_freezeup_doy_valid[:,s] - (m*years_valid[:,s]+b)

                if anomaly == 'mean':
                    avg_freezeup_doy_train[:,s], mean = detrend_ts(avg_freezeup_doy_train[:,s],years_train[:,s],anomaly)
                    avg_freezeup_doy_valid[:,s] = avg_freezeup_doy_valid[:,s] - (mean)

            if detrend:
                # Detrend monthly_vars
                for ivar in range(nvars_monthly_n):
                    for iw in range(monthly_vars.shape[3]):
                        for istart in range(len(start_doy_arr)):
                            if anomaly == 'linear':
                                monthly_vars_train[ivar,:,istart,iw,s], [m,b] = detrend_ts(monthly_vars_train[ivar,:,istart,iw,s],years_train[:,s],anomaly)
                                monthly_vars_valid[ivar,:,istart,iw,s] = monthly_vars_valid[ivar,:,istart,iw,s] - (m*years_valid[:,s]+b)

                            if anomaly == 'mean':
                                monthly_vars_train[ivar,:,istart,iw,s], mean = detrend_ts(monthly_vars_train[ivar,:,istart,iw,s],years_train[:,s],anomaly)
                                monthly_vars_valid[ivar,:,istart,iw,s] = monthly_vars_valid[ivar,:,istart,iw,s] - (mean)

                # Detrend threemonthly_vars
                for ivar in range(nvars_threemonthly):
                    for iw in range(threemonthly_vars.shape[3]):
                        for istart in range(len(start_doy_arr)):
                            if anomaly == 'linear':
                                threemonthly_vars_train[ivar,:,istart,iw,s], [m,b] = detrend_ts(threemonthly_vars_train[ivar,:,istart,iw,s],years_train[:,s],anomaly)
                                threemonthly_vars_valid[ivar,:,istart,iw,s] = threemonthly_vars_valid[ivar,:,istart,iw,s] - (m*years_valid[:,s]+b)

                            if anomaly == 'mean':
                                threemonthly_vars_train[ivar,:,istart,iw,s], mean = detrend_ts(threemonthly_vars_train[ivar,:,istart,iw,s],years_train[:,s],anomaly)
                                threemonthly_vars_valid[ivar,:,istart,iw,s] = threemonthly_vars_valid[ivar,:,istart,iw,s] - (mean)

                # Detrend window_vars
                for ivar in range(nvars_window):
                    for iw in range(window_vars.shape[3]):
                        for istart in range(len(start_doy_arr)):
                            if anomaly == 'linear':
                                window_vars_train[ivar,:,istart,iw,s], [m,b] = detrend_ts(window_vars_train[ivar,:,istart,iw,s],years_train[:,s],anomaly)
                                window_vars_valid[ivar,:,istart,iw,s] = window_vars_valid[ivar,:,istart,iw,s] - (m*years_valid[:,s]+b)

                            if anomaly == 'mean':
                                window_vars_train[ivar,:,istart,iw,s], mean = detrend_ts(window_vars_train[ivar,:,istart,iw,s],years_train[:,s],anomaly)
                                window_vars_valid[ivar,:,istart,iw,s] = window_vars_valid[ivar,:,istart,iw,s] - (mean)


        ##%%%%%%%FEATURE SELECTION %%%%%%%%%%
        # istart = 0 # forecast starting on 'Oct. 27th'
        # istart = 1 # forecast starting on 'Nov. 3rd'
        # istart = 2 # forecast starting on 'Nov. 10th'
        # istart = 3 # forecast starting on 'Nov. 17th'
        # istart = 4 # forecast starting on 'Nov. 24th'
        # istart = 5 # forecast starting on 'Dec. 1st'
        # for istart in range(6):
        for istart in range(5,6):
        # for istart in range(1):
            FUD_valid_pred = np.zeros((nsets,nvalid))*np.nan
            FUD_valid_obs = np.zeros((nsets,nvalid))*np.nan
            print('==============================')
            print(start_doy_labels[istart])

            for iset in range(nsets):
                #-----------------
                # SELECT FORECAST START + TRAIN/VALID PERIOD AND CONVERT TO DATASET
                monthly_varnames_in = []
                for n in range(varnames_monthly[istart,:].shape[0]):
                    if (varnames_monthly[istart,n] != 0):
                        monthly_varnames_in += [varnames_monthly[istart,n]]
                    # else:
                    #     monthly_varnames_in += ['']
                nvars_in = len(monthly_varnames_in)

                df_trainset, df_validset = convert_train_valid_to_dataset(iset,istart,avg_freezeup_doy_train,avg_freezeup_doy_valid,monthly_vars_train[0:nvars_in],monthly_vars_valid[0:nvars_in],monthly_varnames_in,threemonthly_vars_train,threemonthly_vars_valid,varnames_threemonthly,window_vars_train,window_vars_valid,varnames_window,start_doy_arr[istart],ws,ns)
                # !!!!!!!!! TEST WITH ONLY CANSIPS VARS ALLOWED:
                # df_trainset, df_validset = convert_train_valid_to_dataset(iset,istart,avg_freezeup_doy_train,avg_freezeup_doy_valid,monthly_vars_train[19:nvars_in],monthly_vars_valid[19:nvars_in],monthly_varnames_in[19:],threemonthly_vars_train,threemonthly_vars_valid,varnames_threemonthly,window_vars_train,window_vars_valid,varnames_window,start_doy_arr[istart],ws,ns)
                pred_list = df_trainset.columns[1:]

                #-----------------
                # FIRST PRESELECTION STEP: KEEP ONLY VARS WITH SIGNIFICANT
                # CORRELATION WITH FUD DURING TRAINING
                # Correlation with target variable
                if valid_opt == 1:
                    cor = df_trainset.corr(min_periods=int(np.ceil(0.75*(ntraining+iset))))
                else:
                    cor = df_trainset.corr(min_periods=int(np.ceil(0.75*ntraining)))

                cor_target = abs(cor["FUD"])
                # cor_target = (cor["FUD"])

                #Selecting features significatively correlated with target
                navailable = [np.sum(~np.isnan(df_trainset[c])) for c in df_trainset.columns]
                threshold = [r_confidence_interval(0,p_critical,navailable[c],tailed='two')[1] for c in range(len(df_trainset.columns))]
                relevant_features = cor_target[cor_target>=threshold]
                df_trainset_sel1 = df_trainset[[relevant_features.index[i] for i in range(len(relevant_features))]]
                # df_trainset_sel1 = df_trainset

                #-----------------
                # SECOND PRESELECTION STEP: REMOVE COLLINEAR FEATURES
                cor_mc = np.abs(df_trainset_sel1.corr())
                upper_cor = cor_mc.where(np.triu(np.ones(cor_mc.shape),k=1).astype(np.bool))
                # Plot correlation heat map between features
                if plot:
                    plt.figure(figsize=(12,10))
                    sns.heatmap(cor_mc, cmap=plt.cm.Reds)
                    plt.show()

                df_clean = remove_collinear_features(df_trainset_sel1,'FUD', 0.8, verbose=False)
                # df_clean = df_trainset_sel1 # This bypasses the preselction step #2

                #-----------------
                # STEPWISE FEATURE SELECTION
                x = df_clean.drop('FUD',1)
                Y = df_clean['FUD']

                # By forward elimination
                # cf = forwardSelection(x,Y,p_critical,df_clean.columns[1:].values,verbose=False)
                # By backward elimination
                # cb = backwardSelection(x,Y,p_critical,df_clean.columns[1:].values,verbose=False)
                # By stepwise forward-backward elimination
                # cfb = stepwiseSelection(x,Y,df_clean.columns[1:].values,verbose=False)
                cfb = stepwiseSelection(x,Y,df_clean.columns[1:].values,threshold_in=0.01,threshold_out = 0.05,verbose=False)


                print('')
                print('====================================')
                print('Set ' + str(iset))
                for i,co in enumerate(df_trainset_sel1.drop('FUD',1).columns):
                    print('  - '+co)
                # if ntraining == 20:
                    # print('')
                    # print('====================================')
                    # print('Set ' + str(iset))
                    # print('Started with ' + str(df_trainset.shape[1]-1) + ' possible predictors')
                    # print('Got down to ' + str(df_trainset_sel1.shape[1]-1) + ' predictors after Step #1')
                    # print('Then down to ' + str(df_clean.shape[1]-1) + ' predictors after Step #2:')
                    # for i,co in enumerate(df_clean.drop('FUD',1).columns):
                    #     print('  - '+co)
                    # print('')
                    # print('And finally to ' + str(len(cfb)) + ' predictors after stepwise selection:')
                    # for i in range(len(cfb)):
                    #     print('  - '+cfb[i])

                #-----------------
                # KEEP ONLY RETAINED FEATURES AND MAKE LINEAR MODEL FOR TRAINING:
                df_train_mlr = df_clean[['FUD']+cfb]
                df_valid_mlr = df_validset[['FUD']+cfb]
                mlr_model = sm.OLS(df_train_mlr['FUD'], sm.add_constant(df_train_mlr[cfb],has_constant='skip'), missing='drop').fit()
                FUD_train_pred = mlr_model.predict(sm.add_constant(df_train_mlr[cfb],has_constant='skip'))
                # print(mlr_model.params,mlr_model.rsquared,mlr_model.f_pvalue)
                # if ntraining == 18:
                #     print(mlr_model.summary())

                #-----------------
                # COMPUTE VALIDATION METRICS:
                FUD_valid_pred[iset,:] = mlr_model.predict(sm.add_constant(df_valid_mlr[cfb],has_constant='add')).values
                FUD_valid_obs[iset,:]  = df_valid_mlr['FUD'].values

                # if ntraining == 20:
                #     plt.figure()
                #     FUD_train_obs = df_train_mlr['FUD'].values
                #     lb = ''
                #     for i in range(len(cfb)):
                #         lb+= cfb[i]+',\n'
                #     if valid_opt == 1:
                #         plt.plot(np.squeeze(years_train[0:ntraining+iset,iset]),np.squeeze(FUD_train_obs[0:ntraining+iset]),'*-',color='black')
                #         plt.plot(np.squeeze(years_train[0:ntraining+iset,iset]),np.squeeze(FUD_train_pred[0:ntraining+iset].values),'o-', label=lb)
                #     else:
                #         plt.plot(np.squeeze(years_train[:,iset]),np.squeeze(FUD_train_obs),'o-',color='black')
                #         plt.plot(np.squeeze(years_train[:,iset]),np.squeeze(FUD_train_pred.values),'o-', label=lb)

                #     plt.plot(np.squeeze(years_valid[:,iset]), FUD_valid_obs[iset,:],'o:',color='black')
                #     plt.plot(np.squeeze(years_valid[:,iset]), FUD_valid_pred[iset,:],'o:',color=plt.get_cmap('tab20')(0))
                #     plt.legend()

                # if it == (len(ntraining_list)-1):
                #     # plt.figure()
                #     # plt.plot(years_train,df_trainset['Avg. daily NAO, Sep 13 - Oct 12'],'o-',label='Avg. daily NAO, Sep 13 - Oct 12')
                #     # plt.legend()
                #     # plt.figure()
                #     # plt.plot(years_train,df_trainset['Avg. wind direction, Feb 22 - Mar 23'],'o-', label='Avg. wind direction, Feb 22 - Mar 23')
                #     # plt.legend()
                #     # plt.figure()
                #     # plt.plot(years_train,df_trainset['Avg. cloud cover, Sep 13 - Oct 12'],'o-',label='Avg. cloud cover, Sep 13 - Oct 12')
                #     # plt.legend()
                #     plt.figure()
                #     plt.plot(years_train,df_trainset['Tot. snowfall, Nov 01 - Nov 30'],'o-',label='Tot. snowfall, Nov 01 - Nov 30')
                #     plt.legend()


            # if ntraining  == 20:
            # if nvalid > 1:
            #     plt.figure()
            #     plt.plot(np.squeeze(years_valid),FUD_valid_obs[0,:],'o-',color='black')
            #     plt.plot(np.squeeze(years_valid),FUD_valid_pred[0,:],'o-')
            #     plt.title(start_doy_labels[istart]+' - valid_opt = ' +str(valid_opt)+' - ntraining = ' +str(ntraining))
            # else:
            #     plt.figure()
            #     plt.plot(np.squeeze(years_valid),np.squeeze(FUD_valid_obs),'o-',color='black')
            #     plt.plot(np.squeeze(years),avg_freezeup_doy,'o-',color='black')
            #     plt.plot(np.squeeze(years_valid),np.squeeze(FUD_valid_pred),'o:')
            #     plt.title(start_doy_labels[istart]+' - valid_opt = ' +str(valid_opt)+' - ntraining = ' +str(ntraining))

            MAE_valid_end[it,valid_opt,istart] = np.nanmean(np.abs(FUD_valid_pred-FUD_valid_obs))
            RMSE_valid_end[it,valid_opt,istart] = np.sqrt(np.nanmean((FUD_valid_pred-FUD_valid_obs)**2.))
            week_accuracy_valid_end[it,valid_opt,istart] = np.sum(np.abs(FUD_valid_pred-FUD_valid_obs) <= 7)
            valid_model = sm.OLS(np.squeeze(FUD_valid_obs), sm.add_constant(np.squeeze(FUD_valid_pred),has_constant='skip'), missing='drop').fit()
            rsqr_valid_end[it,valid_opt,istart] = valid_model.rsquared
            pval_valid_end[it,valid_opt,istart] = valid_model.f_pvalue
            # print('')
            # print('====================================')
            # print('Validation MAE: ' + "{:05.2f}".format(MAE_valid_end[it,valid_opt,istart]))
            # print('Validation RMSE: ' + "{:05.2f}".format(RMSE_valid_end[it,valid_opt,istart]))
            # print('Validation Week_Acc.: ' + "{:05.2f}".format(week_accuracy_valid_end[it,valid_opt,istart]))



#%%

# # print(mlr_model.summary())



# ols_resid = mlr_model.resid
# y = df_train_mlr['FUD'].values

# w = 1./ ((y - np.mean(y))**2.)

# test_mlr_model = sm.RLM(df_train_mlr['FUD'], sm.add_constant(df_train_mlr[cfb],has_constant='skip'), missing='drop').fit()
# test_FUD_train_pred = test_mlr_model.predict(sm.add_constant(df_train_mlr[cfb],has_constant='skip'))

# plt.figure()
# plt.plot(y, '*-')
# plt.plot(FUD_train_pred,'o-')
# plt.plot(test_FUD_train_pred,'o-')



#%%
fig,ax = plt.subplots(nrows=4,ncols=len(ntraining_list),figsize=(10,10), sharex=True, sharey='row')


for it in range(len(ntraining_list)):
    for valid_opt in range(3):
        ax[0,it].plot(np.arange(6),RMSE_valid_end[it,valid_opt,:],'o-', color = plt.get_cmap('tab20c')(4*valid_opt+1))
ax[0,0].set_ylabel('RMSE (days)')

for it in range(len(ntraining_list)):
    for valid_opt in range(3):
        ax[1,it].plot(np.arange(6),MAE_valid_end[it,valid_opt,:],'o-', color = plt.get_cmap('tab20c')(4*valid_opt+1))
ax[1,0].set_ylabel('MAE (days)')

for it in range(len(ntraining_list)):
    for valid_opt in range(3):
        ax[2,it].plot(np.arange(6),100*(week_accuracy_valid_end[it,valid_opt,:]/(nFUD-ntraining_list[it])),'o-', color = plt.get_cmap('tab20c')(4*valid_opt+1))
ax[2,0].set_ylabel('Accuracy (%)')


for it in range(len(ntraining_list)):
    for valid_opt in range(3):
        ax[3,it].plot(np.arange(6),rsqr_valid_end[it,valid_opt,:],'o-', color = plt.get_cmap('tab20c')(4*valid_opt+1))
ax[3,0].set_ylabel('Rsqr')

for it in range(len(ntraining_list)):
    ax[0,it].set_title('n$_{train}$ ='+ str(ntraining_list[it]))
    plt.sca(ax[3,it])
    plt.xticks(np.arange(6), start_doy_labels, rotation=90)
    ax[3,it].set_xlabel('Forecast start')



#%%

# fig,ax = plt.subplots(nrows=len(ntraining_list),ncols=4,figsize=(10,10), sharex=True, sharey='col')


# for it in range(len(ntraining_list)):
#     for valid_opt in range(3):
#         ax[it,0].plot(np.arange(6),RMSE_valid_end[it,valid_opt,:],'o-', color = plt.get_cmap('tab20c')(4*valid_opt+1))
# ax[0,0].set_title('RMSE (days)')
# ax[-1,0].set_xlabel('Forecast start')
# plt.sca(ax[-1,0])
# plt.xticks(np.arange(6), start_doy_labels, rotation=45)

# for it in range(len(ntraining_list)):
#     for valid_opt in range(3):
#         ax[it,1].plot(np.arange(6),MAE_valid_end[it,valid_opt,:],'o-', color = plt.get_cmap('tab20c')(4*valid_opt+1))
# ax[0,1].set_title('MAE (days)')
# ax[-1,1].set_xlabel('Forecast start')
# plt.sca(ax[-1,1])
# plt.xticks(np.arange(6), start_doy_labels, rotation=45)

# for it in range(len(ntraining_list)):
#     for valid_opt in range(3):
#         ax[it,2].plot(np.arange(6),100*(week_accuracy_valid_end[it,valid_opt,:]/(nFUD-ntraining_list[it])),'o-', color = plt.get_cmap('tab20c')(4*valid_opt+1))
# ax[0,2].set_title('Accuracy (%)')
# ax[-1,2].set_xlabel('Forecast start')
# plt.sca(ax[-1,2])
# plt.xticks(np.arange(6), start_doy_labels, rotation=50)


# for it in range(len(ntraining_list)):
#     for valid_opt in range(3):
#         ax[it,3].plot(np.arange(6),rsqr_valid_end[it,valid_opt,:],'o-', color = plt.get_cmap('tab20c')(4*valid_opt+1))
# ax[0,3].set_title('Rsqr')
# ax[-1,3].set_xlabel('Forecast start')
# plt.sca(ax[-1,3])
# plt.xticks(np.arange(6), start_doy_labels, rotation=50)

# #%%
# BELOW WAS FOR MEOPAR PRESENTATION PURPOSES:

# FUD_valid_pred_test = np.zeros((nsets,nvalid))*np.nan
# FUD_valid_obs_test = np.zeros((nsets,nvalid))*np.nan

# vars_test = ['Nov. Avg. daily NAO',
#              'Nov. Tot. precip.',
#              'Oct. Avg. windspeed',
#              'Apr. Tot. snowfall',
#              'Apr. Avg. daily NAO']

# vars_test = ['Nov. Avg. Ta_mean',
#              # 'Oct. Avg. Ta_mean',
#              # 'Nov. Avg. Twater',
#              ]


# # vars_test = ['Nov. Avg. Ta_mean',
# #               # 'Nov. Avg. daily NAO',
# #               #  'Nov. Tot. snowfall',
# #               # 'Nov. Avg. daily NAO',
# #               # 'Oct. Avg. Ta_mean',
# #               # 'Sept. Avg. Ta_mean',
# #               # 'Aug. Avg. Ta_mean'
# #               ]

# for iset in range(nsets):
# # for iset in range(1):
#     #-----------------
#     # SELECT FORECAST START + TRAIN/VALID PERIOD AND CONVERT TO DATASET
#     df_trainset, df_validset = convert_train_valid_to_dataset(iset,istart,avg_freezeup_doy_train,avg_freezeup_doy_valid,monthly_vars_train,monthly_vars_valid,varnames_monthly,threemonthly_vars_train,threemonthly_vars_valid,varnames_threemonthly,window_vars_train,window_vars_valid,varnames_window,start_doy_arr[istart],ws,ns)
#     pred_list = df_trainset.columns[1:]

#     test_df_trainset = df_trainset[['FUD']+vars_test]
#     test_df_validset = df_validset[['FUD']+vars_test]

#     model_test_train = sm.OLS(test_df_trainset['FUD'], sm.add_constant(test_df_trainset[vars_test],has_constant='skip'), missing='drop').fit()
#     print(model_test_train.summary())

#     FUD_valid_pred_test[iset,:] = model_test_train.predict(sm.add_constant(test_df_validset[vars_test],has_constant='add'))
#     FUD_valid_obs_test[iset,:] = test_df_validset['FUD'].values


# plt.figure();plt.plot(FUD_valid_pred_test,'o-');plt.plot(FUD_valid_obs_test,'*-')
# # plt.figure();plt.plot(FUD_valid_pred_test[0,:],'o-');plt.plot(FUD_valid_obs_test[0,:],'*-')

# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,3))
# ax.set_xlabel('Year')
# ax.set_ylabel('FUD day-of-year')

# plt.plot(years[:-1],np.ones(len(years[:-1]))*(np.nanmean(avg_freezeup_doy)),color=[0.7,0.7,0.7])
# plt.plot(years[:-1],avg_freezeup_doy[:-1],'o-',color='black')

# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,3))
# ax.set_xlabel('Year')
# ax.set_ylabel('FUD day-of-year')

# plt.plot(years[:-1],np.ones(len(years[:-1]))*(365),color=plt.get_cmap('tab20')(9))
# plt.plot(years[:-1],np.ones(len(years[:-1]))*(np.nanmean(avg_freezeup_doy)),color=[0.7,0.7,0.7])
# plt.plot(years[:-1],avg_freezeup_doy[:-1],'o-',color='black')


