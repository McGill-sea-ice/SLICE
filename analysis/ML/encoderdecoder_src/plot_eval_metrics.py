#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:40:54 2023

@author: amelie
"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/''
#%%
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import calendar
from netCDF4 import Dataset
import statsmodels.api as sm
#%%

def ecdf(x, data):
        r'''
        For computing the empirical cumulative distribution function (ecdf) for a
        data sample at values x.

        Args:
            x (float or ndarray):
                The value(s) at which the ecdf is evaluated

            data (float or ndarray):
                A sample for which to compute the ecdf.

        Returns: ecdf_vals (ndarray):
            The ecdf for data, evaluated at x.
        '''

        if isinstance(x,float):
            #if x comes in as float, turn it into a numpy array
            x = np.array([x])


        if isinstance(data,float):
            #if data comes in as float, turn it into a numpy array
            data = np.array([data])


        # sort the values of data from smallest to largest
        xs = np.sort(data[~np.isnan(data)])

        # get the sample size of xs satisfying xs<=x for each x
        def func(vals):
            return len(xs[xs<=vals])

        ys = [len(xs[xs<=vals]) for vals in x]

        return np.array(ys)/float(len(xs))


def get_FUD_category(ts_pred_in,p33_ref,p66_ref):
    cat = np.zeros(len(ts_pred_in))*np.nan
    for iyr in range(len(ts_pred_in)):
        if ~np.isnan(ts_pred_in[iyr]):
            if ts_pred_in[iyr] <= p33_ref:
                cat[iyr] = -1
            elif ts_pred_in[iyr] > p66_ref:
                cat[iyr] = 1
            else:
                cat[iyr] = 0

    return cat


def eval_forecast(ts_fcst_in, ts_obs_in, years_in, clim_out = False):

    if (years_in.shape != ts_fcst_in.shape) | (years_in.shape != ts_obs_in.shape) | (ts_fcst_in.shape != ts_obs_in.shape):
        raise Exception('ERROR... dimensions of observations, forecast, and years does not match.')

    else:
        # Make sure to remove the year 1992 from observations and forecast since it is not available for SEAS5 RWE
        if 1992 in years_in:
            ts_fcst_in[np.where(years_in == 1992)[0][0]] = np.nan
            ts_obs_in[np.where(years_in == 1992)[0][0]] = np.nan

        # FIRST: COMPUTE LOO CLIMATOLOGY FROM TS_OBS_IN:
        ts_clim = np.zeros(len(years_in))*np.nan
        cat_clim = np.zeros(len(years_in))*np.nan
        for iyr,year in enumerate(years_in[:]):
            if ~np.isnan(ts_obs_in[iyr]):
                # REMOVE THE LOO FORECAST YEAR FROM DATA FOR MAKING CLIMATOLOGY
                FUD_in = ts_obs_in.copy()
                FUD_in[iyr] = np.nan
                # COMPUTE TW CLIMATOLOGY AND AVERAGE OBSERVED FUD FOR ALL OTHER YEARS
                mean_obs_FUD = (np.nanmean(FUD_in))
                # ts_clim[iyr] = np.floor(mean_obs_FUD)
                ts_clim[iyr] = (mean_obs_FUD)


        # MAE:
        MAE = np.nanmean(np.abs(ts_obs_in-ts_fcst_in))
        MAE_clim = np.nanmean(np.abs(ts_obs_in-ts_clim))

        # RMSE:
        RMSE = np.sqrt(np.nanmean((ts_obs_in-ts_fcst_in)**2.))
        RMSE_clim = np.sqrt(np.nanmean((ts_obs_in-ts_clim)**2.))

        # R2:
        model = sm.OLS(ts_obs_in, sm.add_constant(ts_fcst_in,has_constant='skip'), missing='drop').fit()
        Rsqr = model.rsquared
        Rsqr_pvalue = model.f_pvalue
        Rsqradj = model.rsquared_adj

        model_clim = sm.OLS(ts_obs_in, sm.add_constant(ts_clim,has_constant='skip'), missing='drop').fit()
        Rsqr_clim = model_clim.rsquared
        Rsqr_pvalue_clim = model_clim.f_pvalue
        Rsqradj_clim = model_clim.rsquared_adj

        # CATEGORICAL ACCURACY:
        p33_obs = np.nanpercentile(ts_obs_in,33.33)
        p66_obs = np.nanpercentile(ts_obs_in,66.66)
        cat_obs = get_FUD_category(ts_obs_in,p33_obs,p66_obs)
        cat_fcst = get_FUD_category(ts_fcst_in,p33_obs,p66_obs)
        cat_clim = get_FUD_category(ts_clim,p33_obs,p66_obs)
        acc = (np.sum(cat_fcst == cat_obs)/(np.sum(~np.isnan(ts_obs_in))))*100
        acc_clim = (np.sum(cat_clim == cat_obs)/(np.sum(~np.isnan(ts_obs_in))))*100

        # 7-DAY ACCURACY:
        acc7 = ((np.sum(np.abs(ts_obs_in-ts_fcst_in) <= 7)/(np.sum(~np.isnan(ts_obs_in)))))*100
        acc7_clim = ((np.sum(np.abs(ts_obs_in-ts_clim) <= 7)/(np.sum(~np.isnan(ts_obs_in)))))*100

        # SS_MAE:
        ss_MAE = 1-(MAE/MAE_clim)

        if clim_out:
            return [MAE, RMSE, Rsqr, Rsqradj, Rsqr_pvalue, acc, acc7, ss_MAE], [MAE_clim, RMSE_clim, Rsqr_clim, Rsqradj_clim, Rsqr_pvalue_clim, acc_clim, acc7_clim]
        else:
            return MAE, RMSE, Rsqr, Rsqradj, Rsqr_pvalue, acc, acc7, ss_MAE


def eval_forecast_probabilistic(ts_fcst_in, ts_obs_in, ts_obs_in_all, years_in, clim_out = False, verbose = True):
    if (years_in.shape[0] != ts_fcst_in.shape[0]) | (years_in.shape[0] != ts_obs_in.shape[0]) | (ts_fcst_in.shape[0] != ts_obs_in.shape[0]):
        raise Exception('ERROR... dimensions of observations, forecast, and years does not match.')

    else:
        # Make sure to remove the year 1992 from observations and forecast since it is not available for SEAS5 RWE
        if 1992 in years_in:
            ts_fcst_in[np.where(years_in == 1992)[0][0]] = np.nan
            ts_obs_in[np.where(years_in == 1992)[0][0]] = np.nan


        # FIRST: COMPUTE LOO CLIMATOLOGY FROM TS_OBS_IN:
        ts_clim = np.zeros(len(years_in))*np.nan
        cat_clim = np.zeros(len(years_in))*np.nan
        for iyr,year in enumerate(years_in[:]):
            if ~np.isnan(ts_obs_in_all[iyr]):
                # REMOVE THE LOO FORECAST YEAR FROM DATA FOR MAKING CLIMATOLOGY
                FUD_in = ts_obs_in_all.copy()
                FUD_in[iyr] = np.nan
                # COMPUTE TW CLIMATOLOGY AND AVERAGE OBSERVED FUD FOR ALL OTHER YEARS
                mean_obs_FUD = (np.nanmean(FUD_in))
                # ts_clim[iyr] = np.floor(mean_obs_FUD)
                ts_clim[iyr] = (mean_obs_FUD)


        p33_obs = np.nanpercentile(ts_obs_in_all,33.33)
        p66_obs = np.nanpercentile(ts_obs_in_all,66.66)
        cat_obs = get_FUD_category(ts_obs_in_all,p33_obs,p66_obs)


        # Get categories for contigency tables
        prob_early = np.zeros((len(years_in)))*np.nan
        prob_normal = np.zeros((len(years_in)))*np.nan
        prob_late = np.zeros((len(years_in)))*np.nan

        obs_early = np.zeros((len(years_in)))*np.nan
        obs_normal = np.zeros((len(years_in)))*np.nan
        obs_late = np.zeros((len(years_in)))*np.nan

        N_members = ts_fcst_in.shape[-1]
        for iyr in range(len(years_in)):
            prob_early[iyr] = np.sum(ts_fcst_in[iyr,0:N_members]<= p33_obs)/N_members
            prob_normal[iyr] = np.sum((ts_fcst_in[iyr,0:N_members] > p33_obs) & (ts_fcst_in[iyr,0:N_members]<= p66_obs))/N_members
            prob_late[iyr] = np.sum(ts_fcst_in[iyr,0:N_members]> p66_obs)/N_members

            obs_early[iyr] = np.sum(ts_obs_in[iyr]<= p33_obs)
            obs_normal[iyr] = np.sum((ts_obs_in[iyr] > p33_obs) & (ts_obs_in[iyr]<= p66_obs))
            obs_late[iyr] = np.sum(ts_obs_in[iyr]> p66_obs)

        d = np.concatenate((np.expand_dims(obs_early,axis=1),np.expand_dims(obs_normal,axis=1),np.expand_dims(obs_late,axis=1)),axis=1)
        r = np.concatenate((np.expand_dims(prob_early,axis=1),np.expand_dims(prob_normal,axis=1),np.expand_dims(prob_late,axis=1)),axis=1)

        # Get probabilistic categories
        cat_prob = np.zeros(len(years_in))*np.nan
        for iyr in range(len(years_in)):
            if np.where(r[iyr] == np.max(r[iyr]))[0][0] == 0:
                cat_prob[iyr] = -1
            elif np.where(r[iyr] == np.max(r[iyr]))[0][0] == 2:
                cat_prob[iyr] = 1
            else:
                cat_prob[iyr] = 0


        # MCRPSS CALCULATION FOR FUD
        CRPS = np.zeros(len(years_in))*np.nan
        CRPS_clim = np.zeros(len(years_in))*np.nan
        x = ts_fcst_in
        y_obs = ts_obs_in
        for iyr,yr in enumerate(years_in):
            if ~np.isnan(y_obs[iyr]) & ~np.all(np.isnan(x[iyr,:])):
                ecdf_model = ecdf(np.arange(325,440,1),x[iyr,:])
                ecdf_clim = ecdf(np.arange(325,440,1),ts_obs_in_all[np.where(years_in != 1992)][np.where(years_in[np.where(years_in != 1992)] != years_in[iyr])])
                ecdf_obs = np.zeros(len(np.arange(325,440,1)))
                ecdf_obs[int(np.where(np.arange(325,440,1) == y_obs[iyr])[0]):] = 1

                CRPS[iyr] = np.nansum((ecdf_obs-ecdf_model)**2.)
                CRPS_clim[iyr] = np.nansum((ecdf_obs-ecdf_clim)**2.)

        if verbose:
            print('--------------------------------------------')
            print('Mean CRPS : ', np.nanmean(CRPS))
            print('Mean CRPS clim: ', np.nanmean(CRPS_clim))
            print('MCRPSS MLR: ',(1-(np.nanmean(CRPS)/np.nanmean(CRPS_clim)))*100, '%')


        # Spread-Error relationship
        var_x = np.nanvar(x,axis=1, ddof=1)
        avg_var_x = np.nanmean(var_x)
        mean_x = np.nanmean(x,axis=1)
        MSE = np.nanmean((mean_x-y_obs)**2.)
        MSE_fair = MSE - (avg_var_x/N_members)
        alpha_fair = MSE_fair/avg_var_x
        print(MSE_fair,avg_var_x)
        # plt.figure()
        # plt.plot(years_in,(mean_x-y_obs)**2./var_x,'o')
        if verbose:
            print('Spread-Error alpha_fair: ', alpha_fair)




#%%
# SETUP/OPTIONS OF THE ANALYSIS
istart_label = ['Nov. 3', 'Nov. 10', 'Nov. 17', 'Nov. 24', 'Dec. 1' ]
istart_plot = [0,1,2,3,4]

plot_clim = True
plot_mlr = True
plot_mlr_rw_EM = True
plot_mlr_rw_MM = False
plot_ml = True

savefig = False
savefig = True
save_folder = './metrics/'

#%%================================================================
# OBSERVATIONS (LONGUEUIL 1992-2019)
years = np. array([1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
                   2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                   2014, 2015, 2016, 2017, 2018, 2019])

avg_freezeup_doy = np.array([np.nan, 358., 364., 342., 365., 350., 365., 360., 343., 367., 339.,
                               348., 354., 350., 381., 341., 347., 352., 357., 364., 358., 347.,
                               365., 371., 351., 348., 356., 354.])

# When evaluating ML forecasts we need to take into account the fact that the observed Tw samples may not have been long enough to detect freeze-up from the model anyway (so it is not the model's fault that it didn't detect freeze-up...)
# So the FUD observations we use will differ for each start date as below.
avg_freezeup_doy_ml_eval = np.array([ [np.nan, 358., 364., 342., 365., 350., 365., 360., 343.,  np.nan, 339., 348.,
                                          354., 350.,  np.nan, 341., 347., 352., 357., 364., 358., 347., 365.,
                                          np.nan, 351., 348., 356., 354.],
                                      [np.nan, 358., 364., 342., 365., 350., 365., 360., 343., 367., 339., 348.,
                                          354., 350.,  np.nan, 341., 347., 352., 357., 364., 358., 347., 365.,
                                          371., 351., 348., 356., 354.],
                                      [np.nan, 358., 364., 342., 365., 350., 365., 360., 343., 367., 339., 348.,
                                          354., 350.,  np.nan, 341., 347., 352., 357., 364., 358., 347., 365.,
                                          371., 351., 348., 356., 354.],
                                      [np.nan, 358., 364., 342., 365., 350., 365., 360., 343., 367., 339., 348.,
                                          354., 350., 381., 341., 347., 352., 357., 364., 358., 347., 365.,
                                          371., 351., 348., 356., 354.],
                                      [np.nan, 358., 364., 342., 365., 350., 365., 360., 343., 367., 339., 348.,
                                          354., 350., 381., 341., 347., 352., 357., 364., 358., 347., 365.,
                                          371., 351., 348., 356., 354.]
                                     ])

FUD_clim = np.nanmean(avg_freezeup_doy)


# CATEGORICAL BASELINE
# # From probabilistic most probable category of SEAS5 Dec. Ta Forecast
# cb_acc_nov1 = 22.22222222222222
# cb_acc_dec1 = 66.66666666666666
# cb_acc = [cb_acc_nov1, cb_acc_nov1, cb_acc_nov1, cb_acc_nov1, cb_acc_dec1]
# # From category of ensemble mean of SEAS5 Dec. Ta Forecast
# cb_acc_nov1_em = 37.03703703703704
# cb_acc_dec1_em = 70.37037037037037
# cb_acc_em = [cb_acc_nov1_em, cb_acc_nov1_em, cb_acc_nov1_em, cb_acc_nov1_em, cb_acc_dec1_em]


#%%================================================================
# INITIALIZE METRICS PLOT
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


color_mlr = plt.get_cmap('Dark2')(5)
color_mlr_rw = plt.get_cmap('Set2')(5)
color_clim = 'k'
color_cb = 'gray'

#%%================================================================
# INITIALIZE TIME SERIES PLOT:
fig_ts,ax_ts = plt.subplots(nrows = 1, ncols = 1, figsize=[14.5,5.5])

color_ml_dec1 = plt.get_cmap('tab20c')(0)
color_ml_nov1 = plt.get_cmap('tab20c')(2)
color_mlr = plt.get_cmap('Dark2')(5)
color_mlr_rw = plt.get_cmap('Set2')(5)

ax_ts.plot(years,avg_freezeup_doy,'o-',color='black', label = 'Observed')
# ax_ts.plot(years,np.ones(len(years))*FUD_clim ,'--',color = 'gray',linewidth=1, label='Climatology')

#%%================================================================
# CLIM FUD BASELINE & MLR - PERFECT FORECAST
valid_scheme = 'LOOk'
mlr_folder = local_path+'slice/prog/analysis/MLR/all_combinations_monthly_predictor/output/all_coefficients_significant_05/'
file_name = mlr_folder +'MLR_monthly_pred_varslast6monthsp05only_Jan1st_maxpred4_valid_scheme_LOOk'
df_select_test = pd.read_pickle(file_name+'_df_select_test')
mlr_predictors = df_select_test['predictors']
mlr_pred = np.array(df_select_test['test_predictions'])
mlr_pred[0] = np.nan

mlr_metrics, clim_metrics = eval_forecast(mlr_pred,avg_freezeup_doy,years,clim_out=True)

[mlr_mae, mlr_rmse, mlr_rsqr, mlr_rsqradj, mlr_rsqr_pvalue, mlr_acc, mlr_acc7, mlr_ss]  = mlr_metrics
[clim_mae, clim_rmse, clim_rsqr, clim_rsqradj, clim_rsqr_pvalue, clim_acc, clim_acc7] = clim_metrics

if plot_clim:
    ax_mae.plot(istart_plot,np.ones(len(istart_plot))*clim_mae,'o-',color=color_clim,label='Climatology')
    ax_rmse.plot(istart_plot,np.ones(len(istart_plot))*clim_rmse,'o-',color=color_clim,label='Climatology')
    ax_ss.plot(istart_plot,np.ones(len(istart_plot))*0,'o-',color=color_clim,label='Climatology')
    # ax_acc.plot(istart_plot,cb_acc,'o-',color=color_cb,label='Categorical baseline')
    # ax_acc.plot(istart_plot,cb_acc_em,'o--',color=color_cb,label='Categorical baseline')
    ax_acc.plot(istart_plot,np.ones(len(istart_plot))*clim_acc,'o-',color=color_clim,label='Climatology')
    ax_acc7.plot(istart_plot,np.ones(len(istart_plot))*clim_acc7,'o-',color=color_clim,label='Climatology')

    # ax_ts.plot(years,clim_pred ,'--',color = 'gray',linewidth=1, label='Climatology')

if plot_mlr:
    ax_mae.plot(istart_plot,np.ones(len(istart_plot))*mlr_mae,'o-',label='MLR', color=color_mlr)
    ax_rmse.plot(istart_plot,np.ones(len(istart_plot))*mlr_rmse,'o-',label='MLR', color=color_mlr)
    ax_rsqr.plot(istart_plot,np.ones(len(istart_plot))*mlr_rsqr,'o-',label='MLR', color=color_mlr)
    ax_ss.plot(istart_plot,np.ones(len(istart_plot))*mlr_ss,'o-',label='MLR', color=color_mlr)
    ax_acc.plot(istart_plot,np.ones(len(istart_plot))*mlr_acc,'o-',label='MLR', color=color_mlr)
    ax_acc7.plot(istart_plot,np.ones(len(istart_plot))*mlr_acc7,'o-',label='MLR', color=color_mlr)

    ax_ts.plot(years,mlr_pred,'o-',color=color_mlr,label='MLR - perfect forecast')

    plot_obs_vs_model_pred_scatter = False
    if plot_obs_vs_model_pred_scatter:

        fig_scatter,ax_scatter = plt.subplots()
        ax_scatter.plot(avg_freezeup_doy,mlr_pred,'.')
        min_plot = (np.min([np.nanmin(mlr_pred),np.nanmin(avg_freezeup_doy)]))
        max_plot = (np.max([np.nanmax(mlr_pred),np.nanmax(avg_freezeup_doy)]))
        ax_scatter.plot(np.arange(331,385),np.arange(331,385),'-', color='k')
        ax_scatter.fill_between(np.arange(331,385),np.arange(331,385)-7,np.arange(331,385)+7,alpha=0.4,color=plt.get_cmap('tab20c')(1))
        # ax_scatter.fill_between(np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)-9,np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)+9,alpha=0.4,color=plt.get_cmap('tab20c')(1))

        for i, yr_label in enumerate(years):
            ax_scatter.annotate(
                str(yr_label), (avg_freezeup_doy[i],mlr_pred[i])),
                                # xytext=(ml_pred[istart_scatter][i]+3, avg_freezeup_doy[i]-3),
                                # arrowprops = dict(arrowstyle="->"))


#%%================================================================
# MLR - REAL-WORLD FORECAST - ENSEMBLE MEAN FORECASTS
mlr_pred_rw_EM_nov1st = np.array([np.nan, 349.08863958, 365.46479098, 343.12084977, 364.42560775,
        364.05898944, 350.44933272, 360.57673107, 350.49539438,
        350.60422056, 350.9271575 , 353.80872058, 348.72246418,
        358.00406604, 365.73426841, 348.00383761, 356.21691112,
        359.90672395, 360.10827662, 363.84013537, 352.75056742,
        355.94697869, 346.79142819, 359.6744208 , 352.21996029,
        348.68745208, 361.51100995, 350.28340285])
mlr_pred_rw_EM_dec1st = np.array([np.nan, 360.74445162, 359.03442245, 338.22189667, 360.42479061,
       358.71721676, 362.56537427, 366.37178395, 347.88139193,
       369.17916134, 346.63521875, 352.15441466, 351.97735595,
       344.36387918, 366.64108927, 340.30506913, 349.65729634,
       362.17561229, 366.32355327, 365.16513067, 359.72314979,
       347.28850197, 356.3459345 , 365.10805073, 350.0180384 ,
       345.39744417, 348.76749579, 354.14555796])

[mlr_mae_rw_EM_nov1st, mlr_rmse_rw_EM_nov1st, mlr_rsqr_rw_EM_nov1st, mlr_rsqradj_rw_EM_nov1st, mlr_rsqr_pvalue_rw_EM_nov1st, mlr_acc_rw_EM_nov1st, mlr_acc7_rw_EM_nov1st, mlr_ss_rw_EM_nov1st]  = eval_forecast(mlr_pred_rw_EM_nov1st,avg_freezeup_doy,years,clim_out=False)
[mlr_mae_rw_EM_dec1st, mlr_rmse_rw_EM_dec1st, mlr_rsqr_rw_EM_dec1st, mlr_rsqradj_rw_EM_dec1st, mlr_rsqr_pvalue_rw_EM_dec1st, mlr_acc_rw_EM_dec1st, mlr_acc7_rw_EM_dec1st, mlr_ss_rw_EM_dec1st]  = eval_forecast(mlr_pred_rw_EM_dec1st,avg_freezeup_doy,years,clim_out=False)

mlr_rw_EM_mae = []
mlr_rw_EM_rmse = []
mlr_rw_EM_rsqr = []
mlr_rw_EM_acc = []
mlr_rw_EM_acc7 = []
mlr_rw_EM_ss = []

for istart in istart_plot:
    if istart < 4:
        mlr_rw_EM_mae.append(mlr_mae_rw_EM_nov1st)
        mlr_rw_EM_rmse.append(mlr_rmse_rw_EM_nov1st)
        mlr_rw_EM_rsqr.append(mlr_rsqr_rw_EM_nov1st)
        mlr_rw_EM_acc.append(mlr_acc_rw_EM_nov1st)
        mlr_rw_EM_acc7.append(mlr_acc7_rw_EM_nov1st)
        mlr_rw_EM_ss.append(mlr_ss_rw_EM_nov1st)
    else:
        mlr_rw_EM_mae.append(mlr_mae_rw_EM_dec1st)
        mlr_rw_EM_rmse.append(mlr_rmse_rw_EM_dec1st)
        mlr_rw_EM_rsqr.append(mlr_rsqr_rw_EM_dec1st)
        mlr_rw_EM_acc.append(mlr_acc_rw_EM_dec1st)
        mlr_rw_EM_acc7.append(mlr_acc7_rw_EM_dec1st)
        mlr_rw_EM_ss.append(mlr_ss_rw_EM_dec1st)

# cat_mlr_rw_probabilistic_nov1st = np.array([np.nan, -1.,  1., -1.,  1.,  1., -1.,  1.,  0.,  1.,  0.,  1., -1.,  1.,
#                                             1., -1.,  1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1., -1.,  1.,
#                                            -1.])
# cat_mlr_rw_probabilistic_dec1st = np.array([ np.nan, 1.,  1., -1.,  1.,  1.,  1.,  1., -1.,  1., -1.,  0.,  0., -1.,
#                                                 1., -1., -1.,  1.,  1.,  1.,  1., -1.,  0.,  1.,  0., -1., -1.,
#                                                 0.])
# mlr_rw_probabilistic_acc_nov1st = (np.sum(cat_mlr_rw_probabilistic_nov1st == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100
# mlr_rw_probabilistic_acc_dec1st = (np.sum(cat_mlr_rw_probabilistic_dec1st == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100
# mlr_rw_probabilistic_acc = np.array([mlr_rw_probabilistic_acc_nov1st,mlr_rw_probabilistic_acc_nov1st,mlr_rw_probabilistic_acc_nov1st,mlr_rw_probabilistic_acc_nov1st,mlr_rw_probabilistic_acc_dec1st])

if plot_mlr_rw_EM:
    # Ensemble Mean
    ax_mae.plot(istart_plot,mlr_rw_EM_mae,'o-',label='MLR - Real-world (EM)', color=color_mlr_rw)
    ax_rmse.plot(istart_plot,mlr_rw_EM_rmse,'o-',label='MLR - Real-world (EM)', color=color_mlr_rw)
    ax_rsqr.plot(istart_plot,mlr_rw_EM_rsqr,'o-',label='MLR - Real-world (EM)', color=color_mlr_rw)
    ax_ss.plot(istart_plot,mlr_rw_EM_ss,'o-',label='MLR - Real-world (EM)', color=color_mlr_rw)
    ax_acc.plot(istart_plot,mlr_rw_EM_acc,'o-',label='MLR - Real-world (EM) ', color=color_mlr_rw)
    ax_acc7.plot(istart_plot,mlr_rw_EM_acc7,'o-',label='MLR - Real-world (EM)', color=color_mlr_rw)

    ax_ts.plot(years,mlr_pred_rw_EM_dec1st ,'o-',color=color_mlr_rw,label = 'MLR - real world (ensemble mean) - Dec.1')

    plot_obs_vs_model_pred_scatter = False
    if plot_obs_vs_model_pred_scatter:

        fig_scatter,ax_scatter = plt.subplots()
        ax_scatter.plot(avg_freezeup_doy,mlr_pred_rw_EM_dec1st,'.')
        min_plot = (np.min([np.nanmin(mlr_pred_rw_EM_dec1st),np.nanmin(avg_freezeup_doy)]))
        max_plot = (np.max([np.nanmax(mlr_pred_rw_EM_dec1st),np.nanmax(avg_freezeup_doy)]))
        ax_scatter.plot(np.arange(331,385),np.arange(331,385),'-', color='k')
        ax_scatter.fill_between(np.arange(331,385),np.arange(331,385)-7,np.arange(331,385)+7,alpha=0.4,color=plt.get_cmap('tab20c')(1))
        # ax_scatter.fill_between(np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)-9,np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)+9,alpha=0.4,color=plt.get_cmap('tab20c')(1))
        ax_scatter.set_title('MLR - Ensemble Mean - Dec 1')
        for i, yr_label in enumerate(years):
            ax_scatter.annotate(
                str(yr_label), (avg_freezeup_doy[i],mlr_pred_rw_EM_dec1st[i]),
                                xytext=(avg_freezeup_doy[i]-4.4,mlr_pred_rw_EM_dec1st[i]+1,),
                                arrowprops = dict(arrowstyle="->"))



#%%================================================================
# MLR - REAL-WORLD FORECAST - MEAN OF ALL MEMBERS FORECASTS
mlr_pred_rw_MM_nov1st = np.array([np.nan, 357.97764508, 359.73527413, 352.16672313, 361.15179515,
       362.7972494 , 351.34825538, 357.11217209, 355.97464643,
       352.22442375, 354.69321617, 355.77603183, 351.66922721,
       352.60506111, 360.63315909, 349.52928235, 356.58250198,
       352.2558313 , 362.61248615, 358.6225232 , 353.87775256,
       354.50767802, 353.63642225, 352.29305961, 349.97289969,
       348.55843656, 360.89707376, 354.60448766])
mlr_pred_rw_MM_dec1st = np.array([np.nan, 360.74872863, 359.92026795, 339.31756501, 360.00853521,
       357.98042982, 359.46088047, 364.67711246, 354.56855763,
       362.55444902, 344.73218217, 353.51588103, 354.64375492,
       347.46320545, 367.51531796, 343.02371889, 352.00832418,
       359.83381895, 368.08806777, 363.44708216, 359.59393654,
       349.23629601, 355.04762997, 361.32090155, 348.57134418,
       348.53140945, 346.20241582, 353.04591413])
[mlr_mae_rw_MM_nov1st, mlr_rmse_rw_MM_nov1st, mlr_rsqr_rw_MM_nov1st, mlr_rsqradj_rw_MM_nov1st, mlr_rsqr_pvalue_rw_MM_nov1st, mlr_acc_rw_MM_nov1st, mlr_acc7_rw_MM_nov1st, mlr_ss_rw_MM_nov1st]  = eval_forecast(mlr_pred_rw_MM_nov1st,avg_freezeup_doy,years,clim_out=False)
[mlr_mae_rw_MM_dec1st, mlr_rmse_rw_MM_dec1st, mlr_rsqr_rw_MM_dec1st, mlr_rsqradj_rw_MM_dec1st, mlr_rsqr_pvalue_rw_MM_dec1st, mlr_acc_rw_MM_dec1st, mlr_acc7_rw_MM_dec1st, mlr_ss_rw_MM_dec1st]  = eval_forecast(mlr_pred_rw_MM_dec1st,avg_freezeup_doy,years,clim_out=False)

mlr_rw_MM_mae = []
mlr_rw_MM_rmse = []
mlr_rw_MM_rsqr = []
mlr_rw_MM_acc = []
mlr_rw_MM_acc7 = []
mlr_rw_MM_ss = []

for istart in istart_plot:
    if istart < 4:
        mlr_rw_MM_mae.append(mlr_mae_rw_MM_nov1st)
        mlr_rw_MM_rmse.append(mlr_rmse_rw_MM_nov1st)
        mlr_rw_MM_rsqr.append(mlr_rsqr_rw_MM_nov1st)
        mlr_rw_MM_acc.append(mlr_acc_rw_MM_nov1st)
        mlr_rw_MM_acc7.append(mlr_acc7_rw_MM_nov1st)
        mlr_rw_MM_ss.append(mlr_ss_rw_MM_nov1st)
    else:
        mlr_rw_MM_mae.append(mlr_mae_rw_MM_dec1st)
        mlr_rw_MM_rmse.append(mlr_rmse_rw_MM_dec1st)
        mlr_rw_MM_rsqr.append(mlr_rsqr_rw_MM_dec1st)
        mlr_rw_MM_acc.append(mlr_acc_rw_MM_dec1st)
        mlr_rw_MM_acc7.append(mlr_acc7_rw_MM_dec1st)
        mlr_rw_MM_ss.append(mlr_ss_rw_MM_dec1st)


if plot_mlr_rw_MM:
    # Mean of all memebers
    ax_mae.plot(istart_plot,mlr_rw_MM_mae,'x--',label='MLR - Real-world', color=color_mlr_rw)
    ax_rmse.plot(istart_plot,mlr_rw_MM_rmse,'x--',label='MLR - Real-world', color=color_mlr_rw)
    ax_rsqr.plot(istart_plot,mlr_rw_MM_rsqr,'x--',label='MLR - Real-world', color=color_mlr_rw)
    ax_ss.plot(istart_plot,mlr_rw_MM_ss,'x--',label='MLR - Real-world', color=color_mlr_rw)
    ax_acc.plot(istart_plot,mlr_rw_MM_acc,'x--',label='MLR - Real-world', color=color_mlr_rw)
    ax_acc7.plot(istart_plot,mlr_rw_EM_acc7,'x--',label='MLR - Real-world', color=color_mlr_rw)

    ax_ts.plot(years,mlr_pred_rw_MM_dec1st ,'o-',color=color_mlr_rw,label='MLR - Dec.1 - Real World ')

    plot_obs_vs_model_pred_scatter = False
    if plot_obs_vs_model_pred_scatter:

        fig_scatter,ax_scatter = plt.subplots()
        ax_scatter.plot(avg_freezeup_doy,mlr_pred_rw_MM_dec1st,'.')
        min_plot = (np.min([np.nanmin(mlr_pred_rw_MM_dec1st),np.nanmin(avg_freezeup_doy)]))
        max_plot = (np.max([np.nanmax(mlr_pred_rw_MM_dec1st),np.nanmax(avg_freezeup_doy)]))
        ax_scatter.plot(np.arange(331,385),np.arange(331,385),'-', color='k')
        ax_scatter.fill_between(np.arange(331,385),np.arange(331,385)-7,np.arange(331,385)+7,alpha=0.4,color=plt.get_cmap('tab20c')(1))
        # ax_scatter.fill_between(np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)-9,np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)+9,alpha=0.4,color=plt.get_cmap('tab20c')(1))
        ax_scatter.set_title('MLR - Mean of all members - Dec 1')
        for i, yr_label in enumerate(years):
            ax_scatter.annotate(
                str(yr_label), (avg_freezeup_doy[i],mlr_pred_rw_MM_dec1st[i]),
                                xytext=(avg_freezeup_doy[i]-4.4,mlr_pred_rw_MM_dec1st[i]+1,),
                                arrowprops = dict(arrowstyle="->"))



#%%================================================================
# ML - PERFECT FORECAST EXPERIMENT & ENSEMBLE MEAN REAL-WORLD EXPERIMENT
def load_data(pred_len,input_len,latent_dim,n_epochs,nb_layers,norm_type,dense_act_func_name,loss_name,anomaly_target,suffix,fpath,seed):
    fname = 'encoderdecoder_horizon'+str(pred_len)+'_context'+str(input_len)+'_nneurons'+str(latent_dim)+(not np.isnan(seed))*('_seed'+str(seed))+'_nepochs'+str(n_epochs)+(nb_layers > 1)*('_'+str(nb_layers)+'layers')+'_'+norm_type+'_'+dense_act_func_name+'_'+loss_name+'_'+anomaly_target*'anomaly_target_'+suffix+'_TEST_METRICS'
    return np.load(fpath+'metrics/'+fname+'.npz', allow_pickle=True)


fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
suffix = ['reducelrexp_0_05_lr0_004_PFE','reducelrexp_0_05_lr0_004_RWE','reducelrexp_0_05_lr0_004_PFE_withsnow_glouton','reducelrexp_0_05_lr0_004_RWE_ensemblemean_withsnow_glouton','reducelrexp_0_025_lr0_001_RWE_withsnow']
pred_len = [60,60,60,60,60,60]
input_len = [128,128,128,128,128,128]
n_epochs = [100,100,100,100,120,100]
latent_dim = [50,50,50,50,50,50]
nb_layers = [1,1,1,1,1,1]
loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
dense_act_func_name = ['None','None','None','None','None','None']
norm_type=['Standard','Standard','Standard','Standard','Standard','Standard']
anomaly_target = [False,False,False,False,False,False,False,False]
seed = [np.nan, np.nan, np.nan, np.nan, 442, np.nan, np.nan]
color_ml = [plt.get_cmap('tab20c')(0),plt.get_cmap('tab20c')(2),plt.get_cmap('tab20c')(12),plt.get_cmap('tab20c')(14),plt.get_cmap('tab20c')(15)]
label_ml = ['E-D', 'E-D - Real world','E-D (w/snow)', 'E-D - Real world (w/snow)', 'E-D - Real world (w/snow) - lower LR']
# color_ml = plt.get_cmap('tab20c')(0)
# color_ml2 = plt.get_cmap('tab20c')(0)
# color_ml3 = plt.get_cmap('tab20c')(0)
# color_ml3_rw = plt.get_cmap('tab20c')(2)

# fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
# suffix = ['reducelrexp_0_05_lr0_004_PFE','reducelrexp_0_025_lr0_001_PFE','reducelrexp_0_025_lr0_001_PFE', 'reducelrexp_0_025_lr0_001_PFE', 'reducelrexp_0_025_lr0_001_PFE']
# pred_len = [60,60,60,60,60,60]
# input_len = [128,128,128,128,128,128]
# n_epochs = [100,120,120,120,120,100]
# latent_dim = [50,50,50,50,50,50]
# nb_layers = [1,1,1,1,1,1]
# loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
# dense_act_func_name = ['None','None','None','None','None','None']
# norm_type=['Standard','Standard','Standard','Standard','Standard','Standard']
# anomaly_target = [False,False,False,False,False,False,False,False]
# seed = [np.nan, 442, 2, 84, 17]
# color_ml = [plt.get_cmap('tab20c')(0),plt.get_cmap('tab20c')(8),plt.get_cmap('tab20c')(9),plt.get_cmap('tab20c')(10),plt.get_cmap('tab20c')(11),plt.get_cmap('tab20b')(7)]
# label_ml = ['E-D', 'E-D - lower LR', 'E-D - lower LR','E-D - lower LR', 'E-D  - lower LR', 'E-D - Real world - All memb. (w/snow)']
# # color_ml = plt.get_cmap('tab20c')(0)
# # color_ml2 = plt.get_cmap('tab20c')(0)
# # color_ml3 = plt.get_cmap('tab20c')(0)
# # color_ml3_rw = plt.get_cmap('tab20c')(2)

fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
suffix = ['reducelrexp_0_05_lr0_004_PFE_withsnow_glouton','reducelrexp_0_05_lr0_004_PFE_nonrecursive_withsnow_trainshuffled','reducelrexp_0_025_lr0_001_PFE_nonrecursive_withsnow_trainshuffled']
input_len = [128,128,128,128]
n_epochs = [100,100,200,100]
latent_dim = [50,50,50,50]
nb_layers = [1,1,1,1]
loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
dense_act_func_name = ['None','None','None','None']
norm_type=['Standard','Standard','Standard','Standard']
anomaly_target = [False,False,False,False]
seed = [np.nan, 442, 442, 442]
color_ml = [plt.get_cmap('tab20c')(0),plt.get_cmap('tab20c')(2),plt.get_cmap('tab20c')(12),plt.get_cmap('tab20c')(14),plt.get_cmap('tab20c')(15)]
label_ml = ['E-D (w/snow)', 'E-D (w/snow, non-recurs. + shuffle)', 'E-D (w/snow, non-recurs. + shuffle) - lower LR']


fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
suffix = ['reducelrexp_0_05_lr0_004_PFE_withsnow_glouton','reducelrexp_0_05_lr0_004_RWE_ensemblemean_withsnow_glouton','reducelrexp_0_05_lr0_004_PFE_nonrecursive_withsnow_trainshuffled','reducelrexp_0_05_lr0_004_RWE_ensemblemean_nonrecursive_withsnow_trainshuffled']
input_len = [128,128,128,128]
n_epochs = [100,100,100,100]
latent_dim = [50,50,50,50]
nb_layers = [1,1,1,1]
loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
dense_act_func_name = ['None','None','None','None']
norm_type=['Standard','Standard','Standard','Standard']
anomaly_target = [False,False,False,False]
seed = [np.nan, np.nan, 442, 442]
color_ml = [plt.get_cmap('tab20c')(0),plt.get_cmap('tab20c')(2),plt.get_cmap('tab20c')(12),plt.get_cmap('tab20c')(14),plt.get_cmap('tab20c')(15)]
label_ml = ['E-D (w/snow)', 'E-D RWE-EM (w/snow)', 'E-D (w/snow, non-recurs. + shuffle)', 'E-D RWE-EM (w/snow, non-recurs. + shuffle)']

# fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
# suffix = ['reducelrexp_0_05_lr0_004_PFE_withsnow_glouton','reducelrexp_0_05_lr0_004_RWE_ensemblemean_withsnow_glouton','reducelrexp_0_05_lr0_004_RWE_allmembers_withsnow_glouton']
# input_len = [128,128,128,128]
# n_epochs = [100,100,100,100]
# latent_dim = [50,50,50,50]
# nb_layers = [1,1,1,1]
# loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
# dense_act_func_name = ['None','None','None','None']
# norm_type=['Standard','Standard','Standard','Standard']
# anomaly_target = [False,False,False,False]
# seed = [np.nan, np.nan, np.nan, 442]
# color_ml = [plt.get_cmap('tab20c')(0),plt.get_cmap('tab20c')(2),plt.get_cmap('tab20c')(14),plt.get_cmap('tab20c')(14),plt.get_cmap('tab20c')(15)]
# label_ml = ['ED-LSTM', 'ED-LSTM Real world (ensemble mean)', 'ED-LSTM Real world (all members at once)']


fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
suffix = ['reducelrexp_0_05_lr0_004_PFE_withsnow_EMtrain','reducelrexp_0_05_lr0_004_RWE_ensemblemean_withsnow_EMtrain','reducelrexp_0_05_lr0_004_PFE_withsnow_EMtrain','reducelrexp_0_05_lr0_004_RWE_ensemblemean_withsnow_EMtrain']
input_len = [128,128,128,128,128,128]
pred_len = [60,60,90,90]
n_epochs = [100,100,100,100]
latent_dim = [50,50,100,100]
nb_layers = [1,1,1,1,1,1]
loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
dense_act_func_name = ['None','None','None','None','None','None']
norm_type=['Standard','Standard','Standard','Standard','Standard','Standard']
anomaly_target = [False,False,False,False,False,False]
seed = [442, 442, 442, 442, 442, 442]
color_ml = [plt.get_cmap('tab20c')(0),plt.get_cmap('tab20c')(2),plt.get_cmap('tab20c')(8),plt.get_cmap('tab20c')(10),plt.get_cmap('tab20c')(12),plt.get_cmap('tab20c')(14)]
label_ml = ['LSTM-ED - perfect forecast', 'LSTM-ED - Real-world ensemble mean forecast','LSTM-ED - perfect forecast 90 days', 'LSTM-ED - Real-world ensemble mean forecast 90 days']


fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
suffix = ['reducelrexp_0_05_lr0_004_PFE_withsnow_EMtrain','reducelrexp_0_05_lr0_004_RWE_ensemblemean_withsnow_EMtrain']
input_len = [128,128,128,128,128,128]
pred_len = [60,60]
n_epochs = [100,100,100,100]
latent_dim = [50,50]
nb_layers = [1,1,1,1,1,1]
loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
dense_act_func_name = ['None','None','None','None','None','None']
norm_type=['Standard','Standard','Standard','Standard','Standard','Standard']
anomaly_target = [False,False,False,False,False,False]
seed = [442, 442, 442, 442, 442, 442]
color_ml = [plt.get_cmap('tab20c')(0),plt.get_cmap('tab20c')(2),plt.get_cmap('tab20c')(8),plt.get_cmap('tab20c')(10),plt.get_cmap('tab20c')(12),plt.get_cmap('tab20c')(14)]
label_ml = ['LSTM-ED - perfect forecast', 'LSTM-ED - Real-world ensemble mean forecast','LSTM-ED - perfect forecast 90 days', 'LSTM-ED - Real-world ensemble mean forecast 90 days']


nruns = len(suffix)
for irun in range(nruns):
    data = load_data(pred_len[irun],input_len[irun],latent_dim[irun],n_epochs[irun],nb_layers[irun],norm_type[irun],dense_act_func_name[irun],loss_name[irun],anomaly_target[irun],suffix[irun],fpath[irun],seed[irun])
    ml_pred = data['FUD_frcst_arr']
    ml_pred[:,0] = np.nan

    ml_mae = []
    ml_rmse = []
    ml_rsqr = []
    ml_acc = []
    ml_acc7 = []
    ml_ss = []
    for i in istart_plot:
        [ml_mae_i,ml_rmse_i,ml_rsqr_i,ml_rsqradj_i,ml_rsqr_pvalue_i,ml_acc_i,ml_acc7_i,ml_ss_i]  = eval_forecast(ml_pred[i,:],avg_freezeup_doy_ml_eval[i],years,clim_out=False)
        # skill score should be calculated with full FUD_obs time series for all start dates:
        ml_ss_i = 1-(ml_mae_i/clim_mae)
        ml_mae.append(ml_mae_i)
        ml_rmse.append(ml_rmse_i)
        ml_rsqr.append(ml_rsqr_i)
        ml_acc.append(ml_acc_i)
        ml_acc7.append(ml_acc7_i)
        ml_ss.append(ml_ss_i)


    if plot_ml:
            ax_mae.plot(istart_plot,ml_mae,'s-',label=label_ml[irun], color=color_ml[irun])
            ax_rmse.plot(istart_plot,ml_rmse,'s-',label=label_ml[irun], color=color_ml[irun])
            ax_rsqr.plot(istart_plot,ml_rsqr,'s-',label=label_ml[irun], color=color_ml[irun])
            ax_ss.plot(istart_plot,ml_ss,'s-',label=label_ml[irun], color=color_ml[irun])
            ax_acc.plot(istart_plot,ml_acc,'s-',label=label_ml[irun], color=color_ml[irun])
            ax_acc7.plot(istart_plot,ml_acc7,'s-',label=label_ml[irun], color=color_ml[irun])

            ax_ts.plot(years,ml_pred[4,:],'s-',color=color_ml[irun],label=label_ml[irun] +' - Dec. 1')

    plot_obs_vs_model_pred_scatter = True
    if plot_obs_vs_model_pred_scatter:
        istart_scatter = 4
        fig_scatter,ax_scatter = plt.subplots()
        ax_scatter.plot(avg_freezeup_doy,ml_pred[istart_scatter],'.')
        min_plot = (np.min([np.nanmin(ml_pred[istart_scatter]),np.nanmin(avg_freezeup_doy)]))
        max_plot = (np.max([np.nanmax(ml_pred[istart_scatter]),np.nanmax(avg_freezeup_doy)]))
        ax_scatter.plot(np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),'-', color='k')
        ax_scatter.fill_between(np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)-7,np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)+7,alpha=0.4,color=plt.get_cmap('tab20c')(1))


        for i, yr_label in enumerate(years):
            ax_scatter.annotate(
                str(yr_label), (avg_freezeup_doy[i],ml_pred[istart_scatter][i])),
                                # xytext=(ml_pred[istart_scatter][i]+3, avg_freezeup_doy[i]-3),
                                # arrowprops = dict(arrowstyle="->"))


#%%================================================================
# ML - REAL WORLD EXPERIMENT

fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
# suffix = ['reducelrexp_0_05_lr0_004_RWE_allmembers_withsnow_EMtrain','reducelrexp_0_05_lr0_004_RWE_allmembers_withsnow_EMtrain','reducelrexp_0_05_lr0_004_RWE_allmembers_withsnow_EMtrain',]
# suffix = ['reducelrexp_0_05_lr0_004_RWE_allmembers_withsnow_EMtrain','reducelrexp_0_05_lr0_004_RWE_allmembers_withsnow_EMtrain',]
suffix = ['reducelrexp_0_05_lr0_004_RWE_allmembers_withsnow_EMtrain']
input_len = [128,128,128]
# pred_len = [60,90]
pred_len = [60]
n_epochs = [100,100,100]
# latent_dim = [50,100]
latent_dim = [50]
nb_layers = [1,1,1]
loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
dense_act_func_name = ['None','None','None','None']
norm_type=['Standard','Standard','Standard','Standard']
anomaly_target = [False,False,False,False]
seed = [442, 442, 442]
color_ml = [plt.get_cmap('tab20c')(12),plt.get_cmap('tab20c')(14),plt.get_cmap('tab20c')(15)]
label_ml = ['LSTM-ED - Real-world ensemble forecast - 50 neurons, 60 days','LSTM-ED - Real-world ensemble forecast - 100 neurons, 90 days']

color_ml_em = plt.get_cmap('tab20c')(14)

nruns = len(suffix)
for irun in range(nruns):
    # data_em = load_data(pred_len[irun],input_len[irun],latent_dim[irun],n_epochs[irun],nb_layers[irun],norm_type[irun],dense_act_func_name[irun],loss_name[irun],anomaly_target[irun],suffix_em[irun],fpath[irun],seed[irun])
    # ml_pred_em = data_em['FUD_frcst_arr']
    # ml_pred_em[:,0] = np.nan

    data = load_data(pred_len[irun],input_len[irun],latent_dim[irun],n_epochs[irun],nb_layers[irun],norm_type[irun],dense_act_func_name[irun],loss_name[irun],anomaly_target[irun],suffix[irun],fpath[irun],seed[irun])
    ml_pred = data['FUD_frcst_arr']
    ml_pred[:,0,:] = np.nan

    ml_mae = []
    ml_rmse = []
    ml_rsqr = []
    ml_acc = []
    ml_acc7 = []
    ml_ss = []
    for i in istart_plot:
        [ml_mae_i,ml_rmse_i,ml_rsqr_i,ml_rsqradj_i,ml_rsqr_pvalue_i,ml_acc_i,ml_acc7_i,ml_ss_i]  = eval_forecast(np.nanmean(ml_pred[i,:,:],axis=1),avg_freezeup_doy_ml_eval[i],years,clim_out=False)
        # skill score should be calculated with full FUD_obs time series for all start dates:
        ml_ss_i = 1-(ml_mae_i/clim_mae)
        ml_mae.append(ml_mae_i)
        ml_rmse.append(ml_rmse_i)
        ml_rsqr.append(ml_rsqr_i)
        ml_acc.append(ml_acc_i)
        ml_acc7.append(ml_acc7_i)
        ml_ss.append(ml_ss_i)

        f, ax = plt.subplots(figsize=[12,5])
        ax.plot(years,avg_freezeup_doy,'-o',color='k', label='Observed FUD')
        # ax.plot(years,ml_pred_em[i] ,'-o',color=plt.get_cmap('tab10')(0), label='LSTM-ED - EM')#' (MAE:' +str(round(float(mae_em),2))+', RMSE:'+str(round(float(rmse_em),2))+', Rsqr:'+str(round(float(r2_em),2))+')')
        ax.plot(years,np.nanmean(ml_pred[i,:,:],axis=1) ,'-o',color=plt.get_cmap('tab20c')(8), label='LSTM-ED - Real-world ensemble mean forecast')#' (MAE:' +str(round(float(mae_em),2))+', RMSE:'+str(round(float(rmse_em),2))+', Rsqr:'+str(round(float(r2_em),2))+')')


        eval_forecast_probabilistic(ml_pred[i,:,:],avg_freezeup_doy_ml_eval[i],avg_freezeup_doy,years,clim_out=False)


        Nmembers = 25
        for im in range(Nmembers):
            if im ==0:
                ax.plot(years,ml_pred[i,:,im],'-',color=plt.get_cmap('tab20c')(10), linewidth=0.86, label='LSTM-ED - Real-world individual members forecasts')
            else:
                ax.plot(years,ml_pred[i,:,im],'-',color=plt.get_cmap('tab20c')(10), linewidth=0.86)
        ax.plot(years,avg_freezeup_doy,'-o',color='k')
        ax.plot(years,np.nanmean(ml_pred[i,:,:],axis=1),'-o',color=plt.get_cmap('tab20c')(8))#' (MAE:' +str(round(float(mae_em),2))+', RMSE:'+str(round(float(rmse_em),2))+', Rsqr:'+str(round(float(r2_em),2))+')')
        # ax.plot(years,ml_pred_em[i] ,'-o',color=plt.get_cmap('tab10')(0))#' (MAE:' +str(round(float(mae_em),2))+', RMSE:'+str(round(float(rmse_em),2))+', Rsqr:'+str(round(float(r2_em),2))+')')

        ax.legend()
        ax.set_ylabel('FUD')
        ax.set_xlabel('Year')
        plt.tight_layout()
        # plt.minorticks_on()
        ax.grid(linestyle=':', which='major')
        ax.grid(linestyle=':', which='minor')



    # if plot_ml:
    #         ax_mae.plot(istart_plot,ml_mae,'s-',label=label_ml[irun], color=color_ml_em)
    #         ax_rmse.plot(istart_plot,ml_rmse,'s-',label=label_ml[irun], color=color_ml_em)
    #         ax_rsqr.plot(istart_plot,ml_rsqr,'s-',label=label_ml[irun], color=color_ml_em)
    #         ax_ss.plot(istart_plot,ml_ss,'s-',label=label_ml[irun], color=color_ml_em)
    #         ax_acc.plot(istart_plot,ml_acc,'s-',label=label_ml[irun], color=color_ml_em)
    #         ax_acc7.plot(istart_plot,ml_acc7,'s-',label=label_ml[irun], color=color_ml_em)

    #         ax_ts.plot(years,ml_pred[4,:],'s-',color=color_ml[irun],label=label_ml[irun] +' - Dec. 1')

    # plot_obs_vs_model_pred_scatter = False
    # if plot_obs_vs_model_pred_scatter:
    #     istart_scatter = 4
    #     fig_scatter,ax_scatter = plt.subplots()
    #     ax_scatter.plot(avg_freezeup_doy,ml_pred[istart_scatter],'.')
    #     min_plot = (np.min([np.nanmin(ml_pred[istart_scatter]),np.nanmin(avg_freezeup_doy)]))
    #     max_plot = (np.max([np.nanmax(ml_pred[istart_scatter]),np.nanmax(avg_freezeup_doy)]))
    #     ax_scatter.plot(np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),'-', color='k')
    #     ax_scatter.fill_between(np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot),np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)-7,np.arange(min_plot-0.01*min_plot,max_plot+0.01*max_plot)+7,alpha=0.4,color=plt.get_cmap('tab20c')(1))


    #     for i, yr_label in enumerate(years):
    #         ax_scatter.annotate(
    #             str(yr_label), (avg_freezeup_doy[i],ml_pred[istart_scatter][i])),
    #                             # xytext=(ml_pred[istart_scatter][i]+3, avg_freezeup_doy[i]-3),
    #                             # arrowprops = dict(arrowstyle="->"))


#%%
# METRICS PLOT DECORATIONS
ax_mae.set_ylabel('MAE (days)')
# ax_mae.set_xlabel('Forecast start date')
ax_mae.set_xticks(np.arange(len(istart_plot)))
ax_mae.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_mae.legend()
ax_mae.grid(linestyle=':')

ax_rmse.set_ylabel('RSME (days)')
# ax_rmse.set_xlabel('Forecast start date')
ax_rmse.set_xticks(np.arange(len(istart_plot)))
ax_rmse.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_rmse.legend()
ax_rmse.grid(linestyle=':')

ax_rsqr.set_ylabel('R$^{2}$')
# ax_rsqr.set_xlabel('Forecast start date')
ax_rsqr.set_xticks(np.arange(len(istart_plot)))
ax_rsqr.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_rsqr.legend()
ax_rsqr.grid(linestyle=':')

ax_ss.set_ylabel('SS$_{MAE}$')
ax_ss.set_xlabel('Forecast start date')
ax_ss.set_xticks(np.arange(len(istart_plot)))
ax_ss.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_ss.legend()
ax_ss.grid(linestyle=':')

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

ax_acc7.set_ylabel('7-day Accuracy (%)')
ax_acc7.set_xlabel('Forecast start date')
ax_acc7.set_xticks(np.arange(len(istart_plot)))
ax_acc7.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_acc7.legend()
ax_acc7.grid(linestyle=':')

fig_metrics.subplots_adjust(wspace=0.25,right=0.95,left=0.06,top=0.9,bottom =0.2)
plt.tight_layout()
if savefig: fig_metrics.savefig(save_folder+'test_metrics_fcst.png', dpi=600)

#%%
# TIME SERIES PLOT DECORATIONS
ax_ts.legend()
ax_ts.set_ylabel('FUD (day of year)')
ax_ts.set_xlabel('Year')
ax_ts.grid(linestyle=':')
ax_ts.set_ylim(332,388)
ax_ts.set_xlim(1992,2020)

plt.tight_layout()
if savefig: fig_ts.savefig(save_folder+'FUDts_fcst.png', dpi=600)


#%%
# ML TIME SERIES FOR ALL START DATES
# for irun in range(nruns):
#     data = load_data(pred_len[irun],input_len[irun],latent_dim[irun],n_epochs[irun],nb_layers[irun],norm_type[irun],dense_act_func_name[irun],loss_name[irun],anomaly_target[irun],suffix[irun],fpath[irun],seed[irun])
#     ml_pred = data['FUD_frcst_arr']
#     ml_pred[:,0] = np.nan

#     fig_ts_ML,ax_ts_ML = plt.subplots(nrows = 1, ncols = 1, figsize=[14.5,5.5])
#     ax_ts_ML.plot(years,avg_freezeup_doy,'o-',color='black', label = 'Observed')

#     istart_label = ['Nov. 3', 'Nov. 10', 'Nov. 17', 'Nov. 24', 'Dec. 1' ]
#     for istart in range(5):
#         if istart == 4:
#             color_ml_istart=plt.get_cmap('tab20c')(0)
#         else:
#             color_ml_istart=plt.get_cmap('tab20c')(7-istart)
#         ax_ts_ML.plot(years,ml_pred[istart,:],'s-',color=color_ml_istart,label=label_ml[irun] +' - '+istart_label[istart])

#     ax_ts_ML.legend()
#     ax_ts_ML.set_ylabel('FUD (day of year)')
#     ax_ts_ML.set_xlabel('Year')
#     ax_ts_ML.grid(linestyle=':')
#     ax_ts_ML.set_ylim(332,388)
#     ax_ts_ML.set_xlim(1992,2020)

#     plt.tight_layout()
    # if savefig: fig_ts_ML.savefig(save_folder+'FUDts_ML_perfectfcst.png', dpi=600)
