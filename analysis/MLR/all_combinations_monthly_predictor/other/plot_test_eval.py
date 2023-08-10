#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:40:54 2023

@author: amelie
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import calendar
from netCDF4 import Dataset
import statsmodels.api as sm
#%%

years = np. array([1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
                   2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                   2014, 2015, 2016, 2017, 2018, 2019])

# OBSERVATIONS (LONGUEUIL 1992-2019)
avg_freezeup_doy = np.array([360., 358., 364., 342., 365., 350., 365., 360., 343., 367., 339.,
                               348., 354., 350., 381., 341., 347., 352., 357., 364., 358., 347.,
                               365., 371., 351., 348., 356., 354.])
# FUD_clim = 355.0
FUD_clim = np.nanmean(avg_freezeup_doy)
cat_obs = np.array([0., 0.,  1., -1.,  1., -1.,  1.,  0., -1.,  1., -1., -1.,  0., -1.,1.,
                   -1., -1.,  0.,  0.,  1.,  0., -1.,  1.,  1.,  0., -1.,  0., 0.,])
p66_obs = 360.0
p33_obs = 350.0

# CATEGORICAL BASELINE
cb_acc = 35.71


# CLIM FUD BASELINE
clim_pred = np.zeros(len(years))*np.nan
cat_clim = np.zeros(len(years))*np.nan
for iyr,year in enumerate(years[:]):
    if ~np.isnan(avg_freezeup_doy[iyr]):
        # REMOVE THE YEAR BEING FORECASTED FROM DATA FOR MAKING CLIMATOLOGY
        FUD_in = avg_freezeup_doy.copy()
        FUD_in[iyr] = np.nan
        # COMPUTE TW CLIMATOLOGY AND AVERAGE OBSERVED FUD FOR ALL OTHER YEARS
        mean_obs_FUD = (np.nanmean(FUD_in))
        std_obs_FUD = (np.nanstd(FUD_in))
        # clim_pred[iyr] = np.floor(mean_obs_FUD)
        clim_pred[iyr] = (mean_obs_FUD) # SAMES AS IN MLR METHOD. SLIGHTLY DIFFERENT THAN ABOVE BUT OVERALL EQUIVALENT...
        if clim_pred[iyr] <= p33_obs:
            cat_clim[iyr] = -1
        elif clim_pred[iyr] > p66_obs:
            cat_clim[iyr] = 1
        else:
            cat_clim[iyr] = 0
clim_mae = np.nanmean(np.abs(clim_pred-avg_freezeup_doy))
clim_rmse = np.sqrt(np.nanmean( (clim_pred-avg_freezeup_doy)**2.))
model = sm.OLS(avg_freezeup_doy, sm.add_constant(clim_pred,has_constant='skip'), missing='drop').fit()
clim_rsqr = model.rsquared
clim_acc7 = ((np.sum(np.abs(avg_freezeup_doy-clim_pred) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
clim_acc = (np.sum(cat_clim == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100



# MLR - PERFECT FORECAST
valid_scheme = 'LOOk'
save_folder = './output/all_coefficients_significant_05/'
file_name = save_folder +'MLR_monthly_pred_varslast6monthsp05only_Jan1st_maxpred4_valid_scheme_LOOk'

df_clim_test_all = pd.read_pickle(file_name+'_df_clim_test_all')
df_clim_valid_all = pd.read_pickle(file_name+'_df_clim_valid_all')
df_select_valid = pd.read_pickle(file_name+'_df_select_valid')
df_select_test = pd.read_pickle(file_name+'_df_select_test')
pred_df_clean = pd.read_pickle(file_name+'_pred_df_clean')

mlr_predictors = df_select_test['predictors']
mlr_pred = np.array(df_select_test['test_predictions'])
mlr_mae = df_select_test['test_MAE']
mlr_rmse = df_select_test['test_RMSE']
mlr_rsqr = df_select_test['test_R2']
mlr_rsqr_adj = df_select_test['test_R2adj']
mlr_acc7 = ((np.sum(np.abs(avg_freezeup_doy-mlr_pred) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_acc10 = ((np.sum(np.abs(avg_freezeup_doy-mlr_pred) <= 10)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_acc5 = ((np.sum(np.abs(avg_freezeup_doy-mlr_pred) <= 5)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_acc2 = ((np.sum(np.abs(avg_freezeup_doy-mlr_pred) <= 2)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_acc = df_select_test['test_Acc']*100
mlr_ss = 1-(mlr_mae/clim_mae)



# MLR - REAL-WORLD FORECAST - ENSEMBLE MEAN FORECASTS
mlr_rw_EM_pred_nov1st = np.array([np.nan, 349.08863958, 365.46479098, 343.12084977, 364.42560775,
       364.05898944, 350.44933272, 360.57673107, 350.49539438,
       350.60422056, 350.9271575 , 353.80872058, 348.72246418,
       358.00406604, 365.73426841, 348.00383761, 356.21691112,
       359.90672395, 360.10827662, 363.84013537, 352.75056742,
       355.94697869, 346.79142819, 359.6744208 , 352.21996029,
       348.68745208, 361.51100995, 350.28340285])
mlr_rw_EM_pred_dec1st = np.array([np.nan, 360.74445162, 359.03442245, 338.22189667, 360.42479061,
       358.71721676, 362.56537427, 366.37178395, 347.88139193,
       369.17916134, 346.63521875, 352.15441466, 351.97735595,
       344.36387918, 366.64108927, 340.30506913, 349.65729634,
       362.17561229, 366.32355327, 365.16513067, 359.72314979,
       347.28850197, 356.3459345 , 365.10805073, 350.0180384 ,
       345.39744417, 348.76749579, 354.14555796])
cat_mlr_rw_EM_nov1st = np.zeros(len(years))*np.nan
cat_mlr_rw_EM_dec1st = np.zeros(len(years))*np.nan
for iyr,year in enumerate(years[:]):
    if ~np.isnan(avg_freezeup_doy[iyr]):
        if mlr_rw_EM_pred_nov1st[iyr] <= p33_obs:
            cat_mlr_rw_EM_nov1st[iyr] = -1
        elif mlr_rw_EM_pred_nov1st[iyr] > p66_obs:
            cat_mlr_rw_EM_nov1st[iyr] = 1
        else:
            cat_mlr_rw_EM_nov1st[iyr] = 0

        if mlr_rw_EM_pred_dec1st[iyr] <= p33_obs:
            cat_mlr_rw_EM_dec1st[iyr] = -1
        elif mlr_rw_EM_pred_dec1st[iyr] > p66_obs:
            cat_mlr_rw_EM_dec1st[iyr] = 1
        else:
            cat_mlr_rw_EM_dec1st[iyr] = 0
mlr_rw_EM_mae_nov1st = 7.173828231113738
mlr_rw_EM_mae_dec1st = 4.666355561977447
mlr_rw_EM_rmse_nov1st = 8.899842288290715
mlr_rw_EM_rmse_dec1st = 5.799686650313942
mlr_rw_EM_rsqr_nov1st = 0.19257706855370416
mlr_rw_EM_rsqr_dec1st = 0.6571173439984634
mlr_rw_EM_acc7_nov1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_EM_pred_nov1st) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_EM_acc7_dec1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_EM_pred_dec1st) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_EM_acc10_nov1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_EM_pred_nov1st) <= 10)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_EM_acc10_dec1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_EM_pred_dec1st) <= 10)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_EM_acc5_nov1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_EM_pred_nov1st) <= 5)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_EM_acc5_dec1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_EM_pred_dec1st) <= 5)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_EM_acc2_nov1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_EM_pred_nov1st) <= 2)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_EM_acc2_dec1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_EM_pred_dec1st) <= 2)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_EM_acc_nov1st = (np.sum(cat_mlr_rw_EM_nov1st == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100
mlr_rw_EM_acc_dec1st = (np.sum(cat_mlr_rw_EM_dec1st == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100
mlr_rw_EM_ss_nov1st = 1-(mlr_rw_EM_mae_nov1st/clim_mae)
mlr_rw_EM_ss_dec1st = 1-(mlr_rw_EM_mae_dec1st/clim_mae)

mlr_rw_EM_mae = np.array([mlr_rw_EM_mae_nov1st,mlr_rw_EM_mae_nov1st,mlr_rw_EM_mae_nov1st,mlr_rw_EM_mae_nov1st,mlr_rw_EM_mae_dec1st])
mlr_rw_EM_rmse = np.array([mlr_rw_EM_rmse_nov1st,mlr_rw_EM_rmse_nov1st,mlr_rw_EM_rmse_nov1st,mlr_rw_EM_rmse_nov1st,mlr_rw_EM_rmse_dec1st])
mlr_rw_EM_rsqr = np.array([mlr_rw_EM_rsqr_nov1st,mlr_rw_EM_rsqr_nov1st,mlr_rw_EM_rsqr_nov1st,mlr_rw_EM_rsqr_nov1st,mlr_rw_EM_rsqr_dec1st])
mlr_rw_EM_acc7 = np.array([mlr_rw_EM_acc7_nov1st,mlr_rw_EM_acc7_nov1st,mlr_rw_EM_acc7_nov1st,mlr_rw_EM_acc7_nov1st,mlr_rw_EM_acc7_dec1st])
mlr_rw_EM_acc10 = np.array([mlr_rw_EM_acc10_nov1st,mlr_rw_EM_acc10_nov1st,mlr_rw_EM_acc10_nov1st,mlr_rw_EM_acc10_nov1st,mlr_rw_EM_acc10_dec1st])
mlr_rw_EM_acc5 = np.array([mlr_rw_EM_acc5_nov1st,mlr_rw_EM_acc5_nov1st,mlr_rw_EM_acc5_nov1st,mlr_rw_EM_acc5_nov1st,mlr_rw_EM_acc5_dec1st])
mlr_rw_EM_acc2 = np.array([mlr_rw_EM_acc2_nov1st,mlr_rw_EM_acc2_nov1st,mlr_rw_EM_acc2_nov1st,mlr_rw_EM_acc2_nov1st,mlr_rw_EM_acc2_dec1st])
mlr_rw_EM_acc = np.array([mlr_rw_EM_acc_nov1st,mlr_rw_EM_acc_nov1st,mlr_rw_EM_acc_nov1st,mlr_rw_EM_acc_nov1st,mlr_rw_EM_acc_dec1st])
mlr_rw_EM_ss = np.array([mlr_rw_EM_ss_nov1st,mlr_rw_EM_ss_nov1st,mlr_rw_EM_ss_nov1st,mlr_rw_EM_ss_nov1st,mlr_rw_EM_ss_dec1st])


# MLR - REAL-WORLD FORECAST - MEAN OF ALL MEMBERS FORECASTS
mlr_rw_MM_pred_nov1st = np.array([np.nan, 357.97764508, 359.73527413, 352.16672313, 361.15179515,
       362.7972494 , 351.34825538, 357.11217209, 355.97464643,
       352.22442375, 354.69321617, 355.77603183, 351.66922721,
       352.60506111, 360.63315909, 349.52928235, 356.58250198,
       352.2558313 , 362.61248615, 358.6225232 , 353.87775256,
       354.50767802, 353.63642225, 352.29305961, 349.97289969,
       348.55843656, 360.89707376, 354.60448766])
mlr_rw_MM_pred_dec1st = np.array([np.nan, 360.74872863, 359.92026795, 339.31756501, 360.00853521,
       357.98042982, 359.46088047, 364.67711246, 354.56855763,
       362.55444902, 344.73218217, 353.51588103, 354.64375492,
       347.46320545, 367.51531796, 343.02371889, 352.00832418,
       359.83381895, 368.08806777, 363.44708216, 359.59393654,
       349.23629601, 355.04762997, 361.32090155, 348.57134418,
       348.53140945, 346.20241582, 353.04591413])
cat_mlr_rw_MM_nov1st = np.zeros(len(years))*np.nan
cat_mlr_rw_MM_dec1st = np.zeros(len(years))*np.nan
for iyr,year in enumerate(years[:]):
    if ~np.isnan(avg_freezeup_doy[iyr]):
        if mlr_rw_MM_pred_nov1st[iyr] <= p33_obs:
            cat_mlr_rw_MM_nov1st[iyr] = -1
        elif mlr_rw_MM_pred_nov1st[iyr] > p66_obs:
            cat_mlr_rw_MM_nov1st[iyr] = 1
        else:
            cat_mlr_rw_MM_nov1st[iyr] = 0

        if mlr_rw_MM_pred_dec1st[iyr] <= p33_obs:
            cat_mlr_rw_MM_dec1st[iyr] = -1
        elif mlr_rw_MM_pred_dec1st[iyr] > p66_obs:
            cat_mlr_rw_MM_dec1st[iyr] = 1
        else:
            cat_mlr_rw_MM_dec1st[iyr] = 0
mlr_rw_MM_mae_nov1st = 7.492818394892767
mlr_rw_MM_mae_dec1st = 5.1965447991618365
mlr_rw_MM_rmse_nov1st = 9.464311845580548
mlr_rw_MM_rmse_dec1st = 6.369128357396457
mlr_rw_MM_rsqr_nov1st = 0.0869079920032595
mlr_rw_MM_rsqr_dec1st = 0.5864800555408598
mlr_rw_MM_acc7_nov1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_MM_pred_nov1st) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_MM_acc7_dec1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_MM_pred_dec1st) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_MM_acc10_nov1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_MM_pred_nov1st) <= 10)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_MM_acc10_dec1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_MM_pred_dec1st) <= 10)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_MM_acc5_nov1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_MM_pred_nov1st) <= 5)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_MM_acc5_dec1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_MM_pred_dec1st) <= 5)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_MM_acc2_nov1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_MM_pred_nov1st) <= 2)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_MM_acc2_dec1st = ((np.sum(np.abs(avg_freezeup_doy-mlr_rw_MM_pred_dec1st) <= 2)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_rw_MM_acc_nov1st = (np.sum(cat_mlr_rw_MM_nov1st == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100
mlr_rw_MM_acc_dec1st = (np.sum(cat_mlr_rw_MM_dec1st == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100
mlr_rw_MM_ss_nov1st = 1-(mlr_rw_MM_mae_nov1st/clim_mae)
mlr_rw_MM_ss_dec1st = 1-(mlr_rw_MM_mae_dec1st/clim_mae)

mlr_rw_MM_mae = np.array([mlr_rw_MM_mae_nov1st,mlr_rw_MM_mae_nov1st,mlr_rw_MM_mae_nov1st,mlr_rw_MM_mae_nov1st,mlr_rw_MM_mae_dec1st])
mlr_rw_MM_rmse =np.array([mlr_rw_MM_rmse_nov1st,mlr_rw_MM_rmse_nov1st,mlr_rw_MM_rmse_nov1st,mlr_rw_MM_rmse_nov1st,mlr_rw_MM_rmse_dec1st])
mlr_rw_MM_rsqr = np.array([mlr_rw_MM_rsqr_nov1st,mlr_rw_MM_rsqr_nov1st,mlr_rw_MM_rsqr_nov1st,mlr_rw_MM_rsqr_nov1st,mlr_rw_MM_rsqr_dec1st])
mlr_rw_MM_acc7 = np.array([mlr_rw_MM_acc7_nov1st,mlr_rw_MM_acc7_nov1st,mlr_rw_MM_acc7_nov1st,mlr_rw_MM_acc7_nov1st,mlr_rw_MM_acc7_dec1st])
mlr_rw_MM_acc10 = np.array([mlr_rw_MM_acc10_nov1st,mlr_rw_MM_acc10_nov1st,mlr_rw_MM_acc10_nov1st,mlr_rw_MM_acc10_nov1st,mlr_rw_MM_acc10_dec1st])
mlr_rw_MM_acc5 = np.array([mlr_rw_MM_acc5_nov1st,mlr_rw_MM_acc5_nov1st,mlr_rw_MM_acc5_nov1st,mlr_rw_MM_acc5_nov1st,mlr_rw_MM_acc5_dec1st])
mlr_rw_MM_acc2 = np.array([mlr_rw_MM_acc2_nov1st,mlr_rw_MM_acc2_nov1st,mlr_rw_MM_acc2_nov1st,mlr_rw_MM_acc2_nov1st,mlr_rw_MM_acc2_dec1st])
mlr_rw_MM_acc = np.array([mlr_rw_MM_acc_nov1st,mlr_rw_MM_acc_nov1st,mlr_rw_MM_acc_nov1st,mlr_rw_MM_acc_nov1st,mlr_rw_MM_acc_dec1st])
mlr_rw_MM_ss = np.array([mlr_rw_MM_ss_nov1st,mlr_rw_MM_ss_nov1st,mlr_rw_MM_ss_nov1st,mlr_rw_MM_ss_nov1st,mlr_rw_MM_ss_dec1st])


# ML - MEOPAR POSTER: IDEAL FORECAST WITH OBS. TA_MEAN ONLY
ml_pred = np.array([[360., 362., 366., 343.,  355, 358., 366., 361., 345.,  355, 366.,
                     355, 354., 351.,  355, 346., 351., 354., 353.,  355, 359., 348.,
                     355,  355, 366., 359., 348., 355.],
                   [359., 361., 366., 342., 364., 357., 363., 361., 347., 372., 365.,
                    373., 354., 349.,  355, 347., 346., 355., 355., 368., 360., 348.,
                     355,  355, 367., 364., 348., 353.],
                   [359., 360., 365., 343., 365., 358., 363., 363., 347., 368., 367.,
                    372., 354., 350.,  355, 348., 351., 358., 352., 369., 361., 350.,
                    371., 380., 367., 377., 359., 355.],
                   [359., 358., 365., 342., 365., 358., 362., 364., 344., 368., 370.,
                    372., 354., 348., 383., 346., 346., 356., 354., 367., 362., 350.,
                    370., 380., 366., 358., 348., 353.],
                   [359., 359., 365., 342., 367., 356., 363., 365., 346., 370., 346.,
                    372., 353., 350., 383., 343., 353., 357., 352., 369., 362., 349.,
                    370., 380., 370., 353., 358., 353.]])
ml_pred[np.isnan(ml_pred)] = FUD_clim

ml_mae = np.array([5.4       , 6.07407407, 6.37037037, 5.32142857, 4.57142857])
ml_rmse = np.array([ 8.06225775,  9.45946518, 10.28123065,  8.90224691,  6.94879229])
ml_rsqr = np.array([0.18700706, 0.18364894, 0.29678957, 0.47252545, 0.70432206])
ml_acc = np.array([0.48      , 0.62962963, 0.7037037 , 0.71428571, 0.75      ])*100
ml_acc7 = np. array([0.68      , 0.74074074, 0.77777778, 0.75      , 0.89285714])*100
ml_ss = 1-(ml_mae/clim_mae)


# ML2 - REDO MEOPAR POSTER WITH OTHER SEED: IDEAL FORECAST WITH OBS. TA_MEAN ONLY
ml2_pred = np.array([[355., 359., 366., 344., 361., 346., 358., 359., 346., 355., 354., 355., 356., 349.,
                      355., 348., 347., 355., 349., 364., 358., 347., 355., 355., 351., 360., 363., 355.],
                     [355., 358., 366., 345., 363., 355., 355., 360., 347., 361., 356., 373., 355., 349.,
                      355., 347., 345., 355., 348., 364., 357., 348., 368., 355., 354., 359., 361., 355.],
                     [355., 360., 369., 342., 365., 355., 357., 361., 346., 363., 361., 351., 354., 347.,
                      355., 348., 351., 357., 353., 368., 361., 351., 365., 375., 366., 356., 359., 353.],
                     [355., 362., 369., 343., 363., 354., 359., 364., 346., 365., 367., 372., 356., 350.,
                      385., 349., 351., 359., 351., 366., 361., 356., 356., 374., 354., 356., 348., 354.],
                     [355., 360., 365., 341., 361., 355., 359., 364., 345., 363., 352., 364., 356., 352.,
                      385., 348., 350., 357., 351., 368., 358., 349., 356., 371., 356., 353., 361., 345.]])

ml2_pred[np.isnan(ml2_pred)] = FUD_clim

ml2_mae = np.array([4.04166667, 5.26923077, 4.53846154, 5.88888889, 4.66666667])
ml2_rmse = np.array([5.75543222, 7.99759579, 6.528046  , 8.59155485, 5.91294875])
ml2_rsqr = np.array([0.49749607, 0.2769224 , 0.57460013, 0.44342901, 0.68272849])
ml2_acc = np.array([0.70833333, 0.69230769, 0.61538462, 0.55555556, 0.62962963])*100
ml2_acc7 = np.array([0.83333333, 0.76923077, 0.84615385, 0.74074074, 0.85185185])*100
ml2_ss = 1-(ml2_mae/clim_mae)



# ML3 - REDO MEOPAR POSTER WITH + SNOWFALL: IDEAL FORECAST OBS. TA_MEAN + SNOWFALL
ml3_pred = np.array([[355., 360., 355., 349., 355., 365., 355., 365., 348., 355., 355., 350., 355., 353.,
                      355., 351., 349., 359., 352., 355., 356., 350., 355., 355., 353., 355., 355., 355.],
                     [355., 357., 370., 345., 355., 366., 368., 364., 354., 367., 355., 352., 354., 355.,
                      355., 351., 352., 360., 355., 370., 359., 350., 368., 355., 359., 364., 347., 356.],
                     [355., 358., 370., 344., 374., 359., 365., 362., 354., 369., 357., 353., 356., 355.,
                      355., 350., 352., 362., 355., 370., 363., 354., 352., 380., 365., 360., 358., 353.],
                     [355., 359., 373., 341., 369., 359., 368., 373., 352., 370., 370., 372., 359., 355.,
                      384., 349., 351., 364., 357., 372., 365., 354., 368., 375., 359., 353., 359., 355.],
                     [355., 359., 370., 337., 365., 357., 367., 365., 351., 367., 358., 353., 359., 357.,
                      383., 346., 352., 363., 364., 371., 361., 349., 348., 372., 360., 356., 357., 353.]])

ml3_mae = np.array([6.        , 6.46153846, 6.38461538, 7.03703704, 5.51851852])
ml3_rmse = np.array([7.37676533, 8.21349733, 7.90326125, 9.72587233, 7.16731263])
ml3_rsqr = np.array([0.17544086, 0.32780075, 0.55738306, 0.57511754, 0.64341146])
ml3_acc = np.array([0.54166667, 0.53846154, 0.5       , 0.59259259, 0.55555556])*100
ml3_acc7 = np.array([0.66666667, 0.61538462, 0.61538462, 0.62962963, 0.77777778])*100
ml3_ss = 1-(ml3_mae/clim_mae)


# ML3 - REAL-WORLD FORECAST - REDO MEOPAR POSTER WITH + SNOWFALL: IDEAL FORECAST OBS. TA_MEAN + SNOWFALL
ml3_rw_pred = np.array([[355., 363., 359., 350., 356., 353., 359., 358., 355., 359., 359., 350., 366., 364.,
                         355., 355., 355., 359., 352., 362., 358., 365., 355., 359., 355., 362., 361., 355.],
                        [355., 358., 373., 349., 358., 353., 357., 360., 355., 355., 357., 357., 360., 363.,
                         369., 364., 355., 358., 356., 365., 361., 367., 367., 362., 362., 366., 359., 355.],
                        [355., 357., 355., 345., 357., 359., 363., 355., 375., 358., 359., 355., 358., 364.,
                         368., 373., 376., 360., 356., 362., 366., 368., 365., 367., 367., 372., 354., 369.],
                        [355., 352., 365., 347., 353., 357., 370., 362., 365., 361., 347., 362., 365., 366.,
                         369., 367., 362., 361., 358., 361., 368., 365., 364., 371., 362., 366., 355., 373.],
                        [355., 361., 364., 341., 348., 351., 380., 369., 364., 372., 354., 356., 362., 353.,
                         373., 357., 361., 365., 350., 379., 369., 357., 370., 371., 361., 373., 356., 370.]])

ml3_rw_mae = np.array([ 7.75      ,  8.07692308, 10.96153846,  9.59259259,  9.48148148])
ml3_rw_rmse = np.array([ 9.42514368, 10.24695077, 14.60637374, 11.87746076, 11.57903597])
ml3_rw_rsqr = np.array([0.03732219, 0.07518711, 0.04007538, 0.1106217 , 0.30495089])
ml3_rw_acc = np.array([0.375     , 0.46153846, 0.42307692, 0.44444444, 0.33333333])*100
ml3_rw_acc7 = np.array([0.54166667, 0.5       , 0.42307692, 0.44444444, 0.37037037])*100
ml3_rw_ss = 1-(ml3_rw_mae/clim_mae)


#%%

istart_label = ['Nov. 3', 'Nov. 10', 'Nov. 17', 'Nov. 24', 'Dec. 1' ]
istart_plot = [0,1,2,3,4]
plot_ml = False
plot_ml2 = False
plot_ml3 = True
plot_ml3_rw = True
plot_mlr = True
plot_mlr_rw = True
plot_clim = True


# fig_mae_ts,ax_mae_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
# fig_rsqr_ts,ax_rsqr_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
# fig_acc_ts,ax_acc_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
# fig_acc7_ts,ax_acc7_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
fig_metrics,ax_metrics = plt.subplots(nrows = 2, ncols = 3,figsize=(10,5.7), sharex = True)

for i, label in enumerate(('A', 'B', 'C', 'D', 'E', 'F')):
    ax = ax_metrics[i//3,i-3*(i//3)]
    ax.text(-0.1, 1.15, label, transform=ax.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

ax_mae = ax_metrics[0,0] ;  #ax_mae.grid(axis='y')
ax_rmse = ax_metrics[0,1] ; #ax_rsqr.grid(axis='y')
ax_rsqr = ax_metrics[0,2] ; #ax_rsqr.grid(axis='y')
ax_ss = ax_metrics[1,0];   #ax_acc.grid(axis='y')
ax_acc = ax_metrics[1,1];   #ax_acc.grid(axis='y')
ax_acc7 = ax_metrics[1,2];  #ax_acc7.grid(axis='y')

color_ml = plt.get_cmap('tab20c')(0)
color_ml2 = plt.get_cmap('tab20c')(0)
color_ml3 = plt.get_cmap('tab20c')(0)
color_ml3_rw = plt.get_cmap('tab20c')(2)
color_mlr = plt.get_cmap('Dark2')(5)
color_mlr_rw = plt.get_cmap('Set2')(5)
color_clim = 'k'
color_cb = 'gray'


if plot_ml: ax_mae.plot(np.arange(len(istart_plot)),ml_mae[istart_plot],'o-',label='E-D', color=color_ml)
if plot_ml2: ax_mae.plot(np.arange(len(istart_plot)),ml2_mae[istart_plot],'x--',label='E-D', color=color_ml2)
if plot_ml3: ax_mae.plot(np.arange(len(istart_plot)),ml3_mae[istart_plot],'s-',label='E-D', color=color_ml3)
if plot_ml3_rw: ax_mae.plot(np.arange(len(istart_plot)),ml3_rw_mae[istart_plot],'s-',label='E-D  - Real-world', color=color_ml3_rw)
if plot_mlr: ax_mae.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_mae,'o-',label='MLR', color=color_mlr)
if plot_mlr_rw: ax_mae.plot(np.arange(len(istart_plot)),mlr_rw_EM_mae[istart_plot],'o-',label='MLR - Real-world', color=color_mlr_rw)
if plot_clim: ax_mae.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_mae,'o-',color=color_clim,label='Climatology')
ax_mae.set_ylabel('MAE (days)')
# ax_mae.set_xlabel('Forecast start date')
ax_mae.set_xticks(np.arange(len(istart_plot)))
ax_mae.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_mae.legend()
ax_mae.grid(linestyle=':')

if plot_ml: ax_rmse.plot(np.arange(len(istart_plot)),ml_rmse[istart_plot],'o-',label='E-D', color=color_ml)
if plot_ml2: ax_rmse.plot(np.arange(len(istart_plot)),ml2_rmse[istart_plot],'x--',label='E-D', color=color_ml2)
if plot_ml3: ax_rmse.plot(np.arange(len(istart_plot)),ml3_rmse[istart_plot],'s-',label='E-D', color=color_ml3)
if plot_ml3_rw: ax_rmse.plot(np.arange(len(istart_plot)),ml3_rw_rmse[istart_plot],'s-',label='E-D  - Real-world', color=color_ml3_rw)
if plot_mlr: ax_rmse.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_rmse,'o-',label='MLR', color=color_mlr)
if plot_mlr_rw: ax_rmse.plot(np.arange(len(istart_plot)),mlr_rw_EM_rmse[istart_plot],'o-',label='MLR - Real-world', color=color_mlr_rw)
if plot_clim: ax_rmse.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_rmse,'o-',color=color_clim,label='Climatology')
ax_rmse.set_ylabel('RSME (days)')
# ax_rmse.set_xlabel('Forecast start date')
ax_rmse.set_xticks(np.arange(len(istart_plot)))
ax_rmse.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_rmse.legend()
ax_rmse.grid(linestyle=':')

if plot_ml: ax_rsqr.plot(np.arange(len(istart_plot)),ml_rsqr[istart_plot],'o-',label='E-D', color=color_ml)
if plot_ml2: ax_rsqr.plot(np.arange(len(istart_plot)),ml2_rsqr[istart_plot],'x--',label='E-D', color=color_ml2)
if plot_ml3: ax_rsqr.plot(np.arange(len(istart_plot)),ml3_rsqr[istart_plot],'s-',label='E-D', color=color_ml3)
if plot_ml3_rw: ax_rsqr.plot(np.arange(len(istart_plot)),ml3_rw_rsqr[istart_plot],'s-',label='E-D  - Real-world', color=color_ml3_rw)
if plot_mlr: ax_rsqr.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_rsqr,'o-',label='MLR', color=color_mlr)
if plot_mlr_rw: ax_rsqr.plot(np.arange(len(istart_plot)),mlr_rw_EM_rsqr[istart_plot],'o-',label='MLR - Real-world', color=color_mlr_rw)
ax_rsqr.set_ylabel('R$^{2}$')
# ax_rsqr.set_xlabel('Forecast start date')
ax_rsqr.set_xticks(np.arange(len(istart_plot)))
ax_rsqr.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_rsqr.legend()
ax_rsqr.grid(linestyle=':')


if plot_ml: ax_ss.plot(np.arange(len(istart_plot)),ml_ss[istart_plot],'o-',label='E-D', color=color_ml)
if plot_ml2: ax_ss.plot(np.arange(len(istart_plot)),ml2_ss[istart_plot],'x--',label='E-D', color=color_ml2)
if plot_ml3: ax_ss.plot(np.arange(len(istart_plot)),ml3_ss[istart_plot],'s-',label='E-D', color=color_ml3)
if plot_ml3_rw: ax_ss.plot(np.arange(len(istart_plot)),ml3_rw_ss[istart_plot],'s-',label='E-D  - Real-world', color=color_ml3_rw)
if plot_mlr: ax_ss.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_ss,'o-',label='MLR', color=color_mlr)
if plot_mlr_rw: ax_ss.plot(np.arange(len(istart_plot)),mlr_rw_EM_ss[istart_plot],'o-',label='MLR - Real-world', color=color_mlr_rw)
if plot_clim: ax_ss.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*0,'o-',color=color_clim,label='Climatology')
ax_ss.set_ylabel('SS$_{MAE}$')
ax_ss.set_xlabel('Forecast start date')
ax_ss.set_xticks(np.arange(len(istart_plot)))
ax_ss.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_ss.legend()
ax_ss.grid(linestyle=':')

if plot_ml: ax_acc.plot(np.arange(len(istart_plot)),ml_acc[istart_plot],'o-',label='E-D', color=color_ml)
if plot_ml2: ax_acc.plot(np.arange(len(istart_plot)),ml2_acc[istart_plot],'x--',label='E-D', color=color_ml2)
if plot_ml3: ax_acc.plot(np.arange(len(istart_plot)),ml3_acc[istart_plot],'s-',label='E-D', color=color_ml3)
if plot_ml3_rw: ax_acc.plot(np.arange(len(istart_plot)),ml3_rw_acc[istart_plot],'s-',label='E-D  - Real-world', color=color_ml3_rw)
if plot_mlr: ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_acc,'o-',label='MLR', color=color_mlr)
if plot_mlr_rw: ax_acc.plot(np.arange(len(istart_plot)),mlr_rw_EM_acc[istart_plot],'o-',label='MLR - Real-world', color=color_mlr_rw)
if plot_clim: ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*cb_acc,'o-',color=color_cb,label='Categorical baseline')
if plot_clim: ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_acc,'o-',color=color_clim,label='Climatology')
ax_acc.set_ylabel('Categorical Accuracy (%)')
ax_acc.set_xlabel('Forecast start date')
ax_acc.set_xticks(np.arange(len(istart_plot)))
ax_acc.set_xticklabels([istart_label[k] for k in istart_plot])
ax_acc.grid(linestyle=':')
# ax_acc.legend()
# ax_acc.plot(np.arange(len(istart_plot)),ml_rmse[istart_plot],'o-',label='E-D', color=color_ml)
# ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_rmse,'o-',label='MLR', color=color_mlr)
# ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_rmse,'o-',color=color_clim,label='Climatology')
# ax_acc.set_ylabel('RMSE (days)')
# ax_acc.set_xlabel('Forecast start date')
# ax_acc.set_xticks(np.arange(len(istart_plot)))
# ax_acc.set_xticklabels([istart_label[k] for k in istart_plot])

if plot_ml: ax_acc7.plot(np.arange(len(istart_plot)),ml_acc7[istart_plot],'o-',label='E-D', color=color_ml)
if plot_ml2: ax_acc7.plot(np.arange(len(istart_plot)),ml2_acc7[istart_plot],'x--',label='E-D', color=color_ml2)
if plot_ml3: ax_acc7.plot(np.arange(len(istart_plot)),ml3_acc7[istart_plot],'s-',label='E-D', color=color_ml3)
if plot_ml3_rw: ax_acc7.plot(np.arange(len(istart_plot)),ml3_rw_acc7[istart_plot],'s-',label='E-D  - Real-world', color=color_ml3_rw)
if plot_mlr: ax_acc7.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_acc7,'o-',label='MLR', color=color_mlr)
if plot_mlr_rw: ax_acc7.plot(np.arange(len(istart_plot)),mlr_rw_EM_acc7[istart_plot],'o-',label='MLR - Real-world', color=color_mlr_rw)
if plot_clim: ax_acc7.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_acc7,'o-',color=color_clim,label='Climatology')
ax_acc7.set_ylabel('7-day Accuracy (%)')
ax_acc7.set_xlabel('Forecast start date')
ax_acc7.set_xticks(np.arange(len(istart_plot)))
ax_acc7.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_acc7.legend()
ax_acc7.grid(linestyle=':')

fig_metrics.subplots_adjust(wspace=0.25,right=0.95,left=0.06,top=0.9,bottom =0.2)
plt.tight_layout()
# plt.savefig(save_folder+'test_metrics_fcst_MLR_perfectfcst_ML_perfectfcst_MLR_realworld_ML_realworld.png', dpi=600)

#%%

fig_ts,ax_ts = plt.subplots(nrows = 1, ncols = 1, figsize=[12.5,4.5])

color_ml_dec1 = plt.get_cmap('tab20c')(0)
color_ml_nov1 = plt.get_cmap('tab20c')(2)
color_mlr = plt.get_cmap('Dark2')(5)
color_mlr_rw = plt.get_cmap('Set2')(5)

ax_ts.plot(years,avg_freezeup_doy,'o-',color='black', label = 'Observed')
ax_ts.plot(years,np.ones(len(years))*FUD_clim ,'--',color = 'gray',linewidth=1, label='Climatology')
# ax_ts.plot(years,clim_pred ,'--',color = 'gray',linewidth=1, label='Climatology')⎄


plot_label = ''
for i in range(len(mlr_predictors)):
    plot_label += mlr_predictors[i]
    if i != (len(mlr_predictors)-1):
        plot_label += ', '
ax_ts.plot(years,mlr_pred,'o-',color=color_mlr,label='MLR')
ax_ts.plot(years,mlr_rw_EM_pred_dec1st ,'o-',color=color_mlr_rw,label='MLR - Dec.1 - Real World ')
# ax_ts.plot(years,ml_pred[0,:],'o-',color=color_ml_nov1,label='Encoder-Decoder - Nov. 1')
# ax_ts.plot(years,ml_pred[4,:],'o-',color=color_ml_dec1,label='Encoder-Decoder - Dec. 1')
# ax_ts.plot(years,ml2_pred[4,:],'x--',color=color_ml_dec1,label='Encoder-Decoder - Dec. 1')
ax_ts.plot(years,ml3_pred[4,:],'s-',color=plt.get_cmap('tab20c')(0),label='Encoder-Decoder - Dec. 1 (w/ snow)')
ax_ts.plot(years,ml3_rw_pred[4,:],'s-',color=plt.get_cmap('tab20c')(2),label='Encoder-Decoder - Dec. 1 (w/ snow) - Real World')
ax_ts.legend()
ax_ts.set_ylabel('FUD (day of year)')
ax_ts.set_xlabel('Year')
ax_ts.grid(linestyle=':')
ax_ts.set_ylim(334,386)
ax_ts.set_xlim(1989,2021)

plt.tight_layout()
# plt.savefig(save_folder+'FUDts_fcst_MLR_perfectfcst_ML_perfectfcst_MLR_realworld_ML_realworld.png', dpi=600)
