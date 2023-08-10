#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 06:54:46 2021

@author: Amelie
"""
import numpy as np
import scipy
from scipy import ndimage

from statsmodels.formula.api import ols
import pandas as pd
import statsmodels.api as sm

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import running_nanmean,find_freezeup_Tw_all_yrs,season_mask
from functions import linear_fit, rolling_climo, r_confidence_interval

#%%
def get_window_vars(vars_in,names_in,end_dates,window_arr,years,time,month_start_day=1):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),len(window_arr),2))*np.nan
    window_size = window_arr[1]-window_arr[0]

    for iyr, year in enumerate(years):

        i0 = (dt.date(int(year),1,month_start_day)-date_ref).days
        i0 = np.where(time == i0)[0][0]

        i1 = (dt.date(int(year)+1,3,month_start_day)-date_ref).days
        try:
            i1 = np.where(time == i1)[0][0]
        except:
            i1 = len(time)-1

        doy0 = (dt.date(int(year),1,month_start_day)-(dt.date(int(year),1,1))).days + 1
        doy_arr = np.arange(doy0, doy0+(i1-i0))

        if ~np.isnan(end_dates[iyr]):
            for iw,w in enumerate(window_arr):
                # window_type == 'moving':
                ied = np.where(doy_arr == end_dates[iyr])[0][0]
                ifd = ied-(iw)*(window_size)
                iw0 = ied-(iw+1)*(window_size)

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,iw,0] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        vars_out[ivar,iyr,iw,0] = np.nansum(var_year[iw0:ifd])
                        # if np.nansum(var_year[iw0:ifd]) == 0:
                        #     vars_out[ivar,iyr,iw,0] = np.nan
                        # else:
                        #     vars_out[ivar,iyr,iw,0] = np.nansum(var_year[iw0:ifd])

                # window_type == 'increasing':
                ifd = np.where(doy_arr == end_dates[iyr])[0][0]
                iw0 = ifd-w

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,iw,1] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        vars_out[ivar,iyr,iw,1] = np.nansum(var_year[iw0:ifd])
                        # if np.nansum(var_year[iw0:ifd]) == 0:
                        #     vars_out[ivar,iyr,iw,1] = np.nan
                        # else:
                        #     vars_out[ivar,iyr,iw,1] = np.nansum(var_year[iw0:ifd])

    return vars_out



def get_window_monthly_vars(vars_in,names_in,end_dates,window_arr,years,time,month_start_day=1):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),len(window_arr),2))*np.nan

    for iyr, year in enumerate(years):

        i0 = (dt.date(int(year),1,month_start_day)-date_ref).days
        i0 = np.where(time == i0)[0][0]

        i1 = (dt.date(int(year)+1,3,month_start_day)-date_ref).days
        try:
            i1 = np.where(time == i1)[0][0]
        except:
            i1 = len(time)-1

        doy0 = (dt.date(int(year),1,month_start_day)-(dt.date(int(year),1,1))).days + 1
        doy_arr = np.arange(doy0, doy0+(i1-i0))

        if ~np.isnan(end_dates[iyr]):
            month_end = (dt.date(year,1,1)+dt.timedelta(days=int(end_dates[iyr]-1))).month
            iend = np.where(doy_arr == end_dates[iyr])[0][0]

            for imonth in range(len(window_arr)):
                month = month_end-(imonth+1)
                doy_month_1st = (dt.date(year,month,1)-dt.date(year,1,1)).days+1
                imonth_1st =  np.where(doy_arr == doy_month_1st)[0][0]
                doy_monthp1_1st = (dt.date(year,month+1,1)-dt.date(year,1,1)).days+1
                imonthp1_1st = np.where(doy_arr == doy_monthp1_1st)[0][0]

                # window_type == 'moving':
                ifd = imonthp1_1st
                iw0 = imonth_1st

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,imonth,0] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        vars_out[ivar,iyr,imonth,0] = np.nansum(var_year[iw0:ifd])

                    if (varname[0:3] == 'Max'):
                        vars_out[ivar,iyr,imonth,0] = np.nanmax(var_year[iw0:ifd])

                    if (varname[0:3] == 'Min'):
                        vars_out[ivar,iyr,imonth,0] = np.nanmin(var_year[iw0:ifd])


                # window_type == 'increasing':
                ifd = iend
                iw0 = imonth_1st

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,imonth,1] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        vars_out[ivar,iyr,imonth,1] = np.nansum(var_year[iw0:ifd])

                    if (varname[0:3] == 'Max'):
                        vars_out[ivar,iyr,imonth,1] = np.nanmax(var_year[iw0:ifd])

                    if (varname[0:3] == 'Min'):
                        vars_out[ivar,iyr,imonth,1] = np.nanmin(var_year[iw0:ifd])

    return vars_out


def deasonalize_ts(Nwindow,vars_in,varnames,time_spec,time,years):
    vars_out = np.zeros(vars_in.shape)*np.nan

    for ivar in range(len(varnames)):
        var_mean, var_std, weather_window = rolling_climo(Nwindow,vars_in[:,ivar],time_spec,time,years)
        # if weather_varnames[ivar][0:3] == 'Tot' :
        #     weather_vars[:,ivar] = weather_vars[:,ivar]
        # else:
        #     weather_vars[:,ivar] = weather_vars[:,ivar]-var_mean
        vars_out[:,ivar] = vars_in[:,ivar]-var_mean

    return vars_out


def bootstrap(xvar_in, yvar_in, nboot=1000):

    nyears = len(xvar_in)
    r_out = np.zeros((nboot))*np.nan

    for n in range(nboot):
        if nboot >1:
            boot_indx = np.random.choice(nyears,size=nyears,replace=True)
        else:
            boot_indx = np.random.choice(nyears,size=nyears,replace=False)


        xvar_boot = xvar_in[boot_indx].copy()
        yvar_boot = yvar_in[boot_indx].copy()

        lincoeff, Rsqr = linear_fit(xvar_boot,yvar_boot)

        r_out[n] = np.sqrt(Rsqr)
        if (lincoeff[0]< 0):
            r_out[n] *= -1

    return r_out


def detrend_ts(xvar_in,yvar_in,years,anomaly_type):

    if anomaly_type == 'linear':
        [mx,bx],_ = linear_fit(years, xvar_in)
        [my,by],_ = linear_fit(years, yvar_in)
        x_trend = mx*years + bx
        y_trend = my*years + by

        xvar_out = xvar_in-x_trend
        yvar_out = yvar_in-y_trend

    if anomaly_type == 'mean':
        x_mean = np.nanmean(xvar_in)
        y_mean = np.nanmean(yvar_in)

        xvar_out = xvar_in-x_mean
        yvar_out = yvar_in-y_mean

    return xvar_out, yvar_out


def freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training = False,verbose = False):

    if rolling_training:
        yh = np.zeros((nyears-training_size,training_size))*np.nan
        yh_hat = np.zeros((nyears-training_size,training_size))*np.nan

        yf = np.zeros((nyears-training_size,1))*np.nan
        yf_hat = np.zeros((nyears-training_size,1))*np.nan
        xf = np.zeros((nyears-training_size,1))*np.nan

        for n in range(nyears-training_size):
            i = 0+n
            f = i+training_size

            # Hindcast Data: rolling 15 years
            data_h = {'Year': df['Year'][i:f],
                    'Freeze-up': df['Freeze-up'][i:f],
                    'Freeze-up Anomaly': df['Freeze-up Anomaly'][i:f],
                    'May Ta_avg Anomaly': df['May Ta_avg Anomaly'][i:f],
                    'Jan. PDO Anomaly': df['Jan. PDO Anomaly'][i:f],
                    'Sept. NAO Anomaly': df['Sept. NAO Anomaly'][i:f],
                    'Feb. PDO Anomaly': df['Feb. PDO Anomaly'][i:f],
                    'Apr. Snowfall Anomaly': df['Apr. Snowfall Anomaly'][i:f],
                    'Apr. NAO Anomaly': df['Apr. NAO Anomaly'][i:f],
                    'Apr. aFDD Anomaly': df['Apr. aFDD Anomaly'][i:f],
                    'Nov. Snowfall Anomaly': df['Nov. Snowfall Anomaly'][i:f],
                    'Nov. Ta_avg Anomaly': df['Nov. Ta_avg Anomaly'][i:f],
                    'Nov. SLP Anomaly': df['Nov. SLP Anomaly'][i:f],
                    'Nov. aTDD Anomaly': df['Nov. aTDD Anomaly'][i:f],
                    'Nov. aFDD Anomaly': df['Nov. aFDD Anomaly'][i:f],
                    'Aug. NAO Anomaly': df['Aug. NAO Anomaly'][i:f],
                    'Sept. RH Anomaly': df['Sept. RH Anomaly'][i:f],
                    'Sept. Cloud Anomaly': df['Sept. Cloud Anomaly'][i:f],
                    'Oct. windspeed Anomaly': df['Oct. windspeed Anomaly'][i:f],
                    'Oct. Twater Anomaly': df['Oct. Twater Anomaly'][i:f],
                    'Nov. Twater Anomaly': df['Nov. Twater Anomaly'][i:f],
                    'May Twater Anomaly': df['May Twater Anomaly'][i:f]
                    }
            df_h = pd.DataFrame(data_h,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly','Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. SLP Anomaly', 'Nov. aTDD Anomaly','Nov. aFDD Anomaly','Aug. NAO Anomaly',
                                'Sept. RH Anomaly',
                                'Sept. Cloud Anomaly',
                                'Oct. windspeed Anomaly',
                                'Oct. Twater Anomaly',
                                'Nov. Twater Anomaly',
                                'May Twater Anomaly'])

            # Forecast Data: following year
            data_f = {'Year': [df['Year'][f]],
                    'Freeze-up': [df['Freeze-up'][f]],
                    'Freeze-up Anomaly': [df['Freeze-up Anomaly'][f]],
                    'May Ta_avg Anomaly': [df['May Ta_avg Anomaly'][f]],
                    'Jan. PDO Anomaly': [df['Jan. PDO Anomaly'][f]],
                    'Sept. NAO Anomaly': [df['Sept. NAO Anomaly'][f]],
                    'Feb. PDO Anomaly': [df['Feb. PDO Anomaly'][f]],
                    'Apr. Snowfall Anomaly': [df['Apr. Snowfall Anomaly'][f]],
                    'Apr. NAO Anomaly': [df['Apr. NAO Anomaly'][f]],
                    'Apr. aFDD Anomaly': [df['Apr. aFDD Anomaly'][f]],
                    'Nov. Snowfall Anomaly': [df['Nov. Snowfall Anomaly'][f]],
                    'Nov. Ta_avg Anomaly': [df['Nov. Ta_avg Anomaly'][f]],
                    'Nov. SLP Anomaly': [df['Nov. SLP Anomaly'][f]],
                    'Nov. aTDD Anomaly': [df['Nov. aTDD Anomaly'][f]],
                    'Nov. aFDD Anomaly': [df['Nov. aFDD Anomaly'][f]],
                    'Aug. NAO Anomaly': [df['Aug. NAO Anomaly'][f]],
                    'Sept. RH Anomaly': [df['Sept. RH Anomaly'][f]],
                    'Sept. Cloud Anomaly': [df['Sept. Cloud Anomaly'][f]],
                    'Oct. windspeed Anomaly': [df['Oct. windspeed Anomaly'][f]],
                    'Oct. Twater Anomaly': [df['Oct. Twater Anomaly'][f]],
                    'Nov. Twater Anomaly': [df['Nov. Twater Anomaly'][f]],
                    'May Twater Anomaly': [df['May Twater Anomaly'][f]]
                    }
            df_f = pd.DataFrame(data_f,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly','Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. SLP Anomaly', 'Nov. aTDD Anomaly','Nov. aFDD Anomaly','Aug. NAO Anomaly',
                                'Sept. RH Anomaly',
                                'Sept. Cloud Anomaly',
                                'Oct. windspeed Anomaly',
                                'Oct. Twater Anomaly',
                                'Nov. Twater Anomaly',
                                'May Twater Anomaly'])

            # Hindcast
            xh = df_h[x_model]
            yh_m = df_h['Freeze-up']
            yh[n,:] = np.array(df_h['Freeze-up'])

            xh = sm.add_constant(xh, has_constant='add') # adding a constant
            model = sm.OLS(yh_m, xh).fit()
            yh_hat[n,:] = model.predict(xh)

            if verbose:
                print_model = model.summary()
                print(print_model)

            # Forecast
            xf[n,:] = np.array(df_f[['Year']])
            xf_m = df_f[x_model]
            yf[n,:] = df_f['Freeze-up']

            xf_m = sm.add_constant(xf_m, has_constant='add') # adding a constant
            yf_hat[n,:] = model.predict(xf_m)
    else:
        # Hindcast Data: rolling 15 years
        data_h = {'Year': df['Year'][0:training_size],
                'Freeze-up': df['Freeze-up'][0:training_size],
                'Freeze-up Anomaly': df['Freeze-up Anomaly'][0:training_size],
                'May Ta_avg Anomaly': df['May Ta_avg Anomaly'][0:training_size],
                'Jan. PDO Anomaly': df['Jan. PDO Anomaly'][0:training_size],
                'Sept. NAO Anomaly': df['Sept. NAO Anomaly'][0:training_size],
                'Feb. PDO Anomaly': df['Feb. PDO Anomaly'][0:training_size],
                'Apr. Snowfall Anomaly': df['Apr. Snowfall Anomaly'][0:training_size],
                'Apr. NAO Anomaly': df['Apr. NAO Anomaly'][0:training_size],
                'Apr. aFDD Anomaly': df['Apr. aFDD Anomaly'][0:training_size],
                'Nov. Snowfall Anomaly': df['Nov. Snowfall Anomaly'][0:training_size],
                'Nov. Ta_avg Anomaly': df['Nov. Ta_avg Anomaly'][0:training_size],
                'Nov. SLP Anomaly': df['Nov. SLP Anomaly'][0:training_size],
                'Nov. aTDD Anomaly': df['Nov. aTDD Anomaly'][0:training_size],
                'Nov. aFDD Anomaly': df['Nov. aFDD Anomaly'][0:training_size],
                'Aug. NAO Anomaly': df['Aug. NAO Anomaly'][0:training_size],
                'Sept. RH Anomaly': df['Sept. RH Anomaly'][0:training_size],
                'Sept. Cloud Anomaly': df['Sept. Cloud Anomaly'][0:training_size],
                'Oct. windspeed Anomaly': df['Oct. windspeed Anomaly'][0:training_size],
                'Oct. Twater Anomaly': df['Oct. Twater Anomaly'][0:training_size],
                'Nov. Twater Anomaly': df['Nov. Twater Anomaly'][0:training_size],
                'May Twater Anomaly': df['May Twater Anomaly'][0:training_size]
                }
        df_h = pd.DataFrame(data_h,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly','Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. SLP Anomaly','Nov. aTDD Anomaly','Nov. aFDD Anomaly','Aug. NAO Anomaly',
                                'Sept. RH Anomaly',
                                'Sept. Cloud Anomaly',
                                'Oct. windspeed Anomaly',
                                'Oct. Twater Anomaly',
                                'Nov. Twater Anomaly',
                                'May Twater Anomaly'])

        # Forecast Data: following year
        data_f = {'Year': df['Year'][training_size:],
                'Freeze-up': df['Freeze-up'][training_size:],
                'Freeze-up Anomaly': df['Freeze-up Anomaly'][training_size:],
                'May Ta_avg Anomaly': df['May Ta_avg Anomaly'][training_size:],
                'Jan. PDO Anomaly': df['Jan. PDO Anomaly'][training_size:],
                'Sept. NAO Anomaly': df['Sept. NAO Anomaly'][training_size:],
                'Feb. PDO Anomaly': df['Feb. PDO Anomaly'][training_size:],
                'Apr. Snowfall Anomaly': df['Apr. Snowfall Anomaly'][training_size:],
                'Apr. NAO Anomaly': df['Apr. NAO Anomaly'][training_size:],
                'Apr. aFDD Anomaly': df['Apr. aFDD Anomaly'][training_size:],
                'Nov. Snowfall Anomaly': df['Nov. Snowfall Anomaly'][training_size:],
                'Nov. Ta_avg Anomaly': df['Nov. Ta_avg Anomaly'][training_size:],
                'Nov. SLP Anomaly': df['Nov. SLP Anomaly'][training_size:],
                'Nov. aTDD Anomaly': df['Nov. aTDD Anomaly'][training_size:],
                'Nov. aFDD Anomaly': df['Nov. aFDD Anomaly'][training_size:],
                'Aug. NAO Anomaly': df['Aug. NAO Anomaly'][training_size:],
                'Sept. RH Anomaly': df['Sept. RH Anomaly'][training_size:],
                'Sept. Cloud Anomaly': df['Sept. Cloud Anomaly'][training_size:],
                'Oct. windspeed Anomaly': df['Oct. windspeed Anomaly'][training_size:],
                'Oct. Twater Anomaly': df['Oct. Twater Anomaly'][training_size:],
                'Nov. Twater Anomaly': df['Nov. Twater Anomaly'][training_size:],
                'May Twater Anomaly': df['May Twater Anomaly'][training_size:]
                }
        df_f = pd.DataFrame(data_f,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly','Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. SLP Anomaly','Nov. aTDD Anomaly','Nov. aFDD Anomaly','Aug. NAO Anomaly',
                                'Sept. RH Anomaly',
                                'Sept. Cloud Anomaly',
                                'Oct. windspeed Anomaly',
                                'Oct. Twater Anomaly',
                                'Nov. Twater Anomaly',
                                'May Twater Anomaly'])

        xh = df_h[x_model]
        xf = df_f[x_model]

        yh = df_h['Freeze-up']
        yf = df_f['Freeze-up']

        # Hindcast
        xh = sm.add_constant(xh) # adding a constant
        model = sm.OLS(yh, xh).fit()
        yh_hat = model.predict(xh)

        if verbose:
            print_model = model.summary()
            print(print_model)

        # Forecast
        xf = sm.add_constant(xf) # adding a constant
        yf_hat = model.predict(xf)



    # Evaluate Hindcast and Forecast:
    fig, ax = plt.subplots()
    ax.plot(np.array(df['Year']),np.array(df['Freeze-up']),'o-')

    if not rolling_training:
        std_h = np.nanstd(yh-yh_hat)
        std_f = np.nanstd(yf-yf_hat)

        mae_h = np.nanmean(np.abs(yh-yh_hat))
        mae_f = np.nanmean(np.abs(yf-yf_hat))

        rmse_h = np.sqrt(np.nanmean((yh-yh_hat)**2.))
        rmse_f = np.sqrt(np.nanmean((yf-yf_hat)**2.))

        _, Rsqr_h = linear_fit(yh,yh_hat)
        _, Rsqr_f = linear_fit(yf,yf_hat)

        Rsqr_adj_h = 1-(((1-Rsqr_h)*(yh.shape[0]-1))/(yh.shape[0]-model.df_model-1))
        Rsqr_adj_f = 1-(((1-Rsqr_f)*(yf.shape[0]-1))/(yf.shape[0]-model.df_model-1))

        print('------------------------------------')
        print('Hindcast: 1992-2006')
        print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
        print(mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h)
        print('')
        print('Forecast: 2007-2016')
        print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
        print(mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f)
        print('------------------------------------')

        ax.plot(df_f['Year'],yf_hat, 'o--', color= plt.get_cmap('tab20')(4))
        ax.plot(np.array(df_h['Year']),yh_hat, '-', color= plt.get_cmap('tab20')(2))
        ax.set_xlabel('Year')
        ax.set_label('Freeze-up DOY')

        return np.array(df['Year']),np.array(df['Freeze-up']),df_f['Year'],yf,yf_hat,np.array(df_h['Year']),yh_hat

    else:

        std_h = np.nanstd(yh[0,:]-yh_hat[0,:])
        std_f = np.nanstd(yf-yf_hat)

        mae_h = np.nanmean(np.abs(yh[0,:]-yh_hat[0,:]))
        mae_f = np.nanmean(np.abs(yf-yf_hat))

        rmse_h = np.sqrt(np.nanmean((yh[0,:]-yh_hat[0,:])**2.))
        rmse_f = np.sqrt(np.nanmean((yf-yf_hat)**2.))

        _, Rsqr_h = linear_fit(np.squeeze(yh[0,:]),np.squeeze(yh_hat[0,:]))
        _, Rsqr_f = linear_fit(np.squeeze(yf),np.squeeze(yf_hat))

        Rsqr_adj_h = 1-(((1-Rsqr_h)*(yh.shape[1]-1))/(yh.shape[1]-model.df_model-1))
        Rsqr_adj_f = 1-(((1-Rsqr_f)*(yf.shape[0]-1))/(yf.shape[0]-model.df_model-1))

        print('------------------------------------')
        print('First Hindcast: 1992-2006')
        print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
        print(mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h)
        print('')
        print('Forecast: 2007-2016')
        print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
        print(mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f)
        print('------------------------------------')

        # print(xf[:,0])
        ax.plot(xf[:,0],yf_hat, 'o--', color= plt.get_cmap('tab20')(4))
        ax.plot(np.array(df['Year'][0:training_size]),yh_hat[0,:], '-', color= plt.get_cmap('tab20')(2))
        ax.set_xlabel('Year')
        ax.set_ylabel('Freeze-up DOY')

        return np.array(df['Year']),np.array(df['Freeze-up']),xf[:,0],yf,yf_hat,np.array(df['Year'][0:training_size]),yh_hat[0,:]




# def freezeup_anomaly_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training = False,verbose = False):

#     if rolling_training:
#         yh = np.zeros((nyears-training_size,training_size))*np.nan
#         yh_hat = np.zeros((nyears-training_size,training_size))*np.nan

#         yf = np.zeros((nyears-training_size,1))*np.nan
#         yf_hat = np.zeros((nyears-training_size,1))*np.nan
#         xf = np.zeros((nyears-training_size,1))*np.nan

#         for n in range(nyears-training_size):
#             i = 0+n
#             f = i+training_size

#             # Hindcast Data: rolling 15 years
#             data_h = {'Year': df['Year'][i:f],
#                     'Freeze-up': df['Freeze-up'][i:f],
#                     'Freeze-up Anomaly': df['Freeze-up Anomaly'][i:f],
#                     'May Ta_avg Anomaly': df['May Ta_avg Anomaly'][i:f],
#                     'Jan. PDO Anomaly': df['Jan. PDO Anomaly'][i:f],
#                     'Sept. NAO Anomaly': df['Sept. NAO Anomaly'][i:f],
#                     'Feb. PDO Anomaly': df['Feb. PDO Anomaly'][i:f],
#                     'Apr. Snowfall Anomaly': df['Apr. Snowfall Anomaly'][i:f],
#                     'Apr. NAO Anomaly': df['Apr. NAO Anomaly'][i:f],
#                     'Apr. aFDD Anomaly': df['Apr. aFDD Anomaly'][i:f],
#                     'Nov. Snowfall Anomaly': df['Nov. Snowfall Anomaly'][i:f],
#                     'Nov. Ta_avg Anomaly': df['Nov. Ta_avg Anomaly'][i:f],
#                     'Nov. SLP Anomaly': df['Nov. SLP Anomaly'][i:f],
#                     'Nov. aTDD Anomaly': df['Nov. aTDD Anomaly'][i:f]
#                     }
#             df_h = pd.DataFrame(data_h,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly','Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. SLP Anomaly', 'Nov. aTDD Anomaly'])

#             # Forecast Data: following year
#             data_f = {'Year': [df['Year'][f]],
#                     'Freeze-up': [df['Freeze-up'][f]],
#                     'Freeze-up Anomaly': [df['Freeze-up Anomaly'][f]],
#                     'May Ta_avg Anomaly': [df['May Ta_avg Anomaly'][f]],
#                     'Jan. PDO Anomaly': [df['Jan. PDO Anomaly'][f]],
#                     'Sept. NAO Anomaly': [df['Sept. NAO Anomaly'][f]],
#                     'Feb. PDO Anomaly': [df['Feb. PDO Anomaly'][f]],
#                     'Apr. Snowfall Anomaly': [df['Apr. Snowfall Anomaly'][f]],
#                     'Apr. NAO Anomaly': [df['Apr. NAO Anomaly'][f]],
#                     'Apr. aFDD Anomaly': [df['Apr. aFDD Anomaly'][f]],
#                     'Nov. Snowfall Anomaly': [df['Nov. Snowfall Anomaly'][f]],
#                     'Nov. Ta_avg Anomaly': [df['Nov. Ta_avg Anomaly'][f]],
#                     'Nov. SLP Anomaly': [df['Nov. SLP Anomaly'][f]],
#                     'Nov. aTDD Anomaly': [df['Nov. aTDD Anomaly'][f]]
#                     }
#             df_f = pd.DataFrame(data_f,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly','Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. SLP Anomaly', 'Nov. aTDD Anomaly'])

#             # Hindcast
#             xh = df_h[x_model]
#             yh_m = df_h['Freeze-up Anomaly']
#             yh[n,:] = np.array(df_h['Freeze-up Anomaly'])

#             xh = sm.add_constant(xh, has_constant='add') # adding a constant
#             model = sm.OLS(yh_m, xh).fit()
#             yh_hat[n,:] = model.predict(xh)

#             if verbose:
#                 print_model = model.summary()
#                 print(print_model)

#             # Forecast
#             xf[n,:] = np.array(df_f[['Year']])
#             xf_m = df_f[x_model]
#             yf[n,:] = df_f['Freeze-up Anomaly']

#             xf_m = sm.add_constant(xf_m, has_constant='add') # adding a constant
#             yf_hat[n,:] = model.predict(xf_m)
#     else:
#         # Hindcast Data: rolling 15 years
#         data_h = {'Year': df['Year'][0:training_size],
#                 'Freeze-up': df['Freeze-up'][0:training_size],
#                 'Freeze-up Anomaly': df['Freeze-up Anomaly'][0:training_size],
#                 'May Ta_avg Anomaly': df['May Ta_avg Anomaly'][0:training_size],
#                 'Jan. PDO Anomaly': df['Jan. PDO Anomaly'][0:training_size],
#                 'Sept. NAO Anomaly': df['Sept. NAO Anomaly'][0:training_size],
#                 'Feb. PDO Anomaly': df['Feb. PDO Anomaly'][0:training_size],
#                 'Apr. Snowfall Anomaly': df['Apr. Snowfall Anomaly'][0:training_size],
#                 'Apr. NAO Anomaly': df['Apr. NAO Anomaly'][0:training_size],
#                 'Apr. aFDD Anomaly': df['Apr. aFDD Anomaly'][0:training_size],
#                 'Nov. Snowfall Anomaly': df['Nov. Snowfall Anomaly'][0:training_size],
#                 'Nov. Ta_avg Anomaly': df['Nov. Ta_avg Anomaly'][0:training_size],
#                 'Nov. SLP Anomaly': df['Nov. SLP Anomaly'][0:training_size],
#                 'Nov. aTDD Anomaly': df['Nov. aTDD Anomaly'][0:training_size]
#                 }
#         df_h = pd.DataFrame(data_h,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly','Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. SLP Anomaly','Nov. aTDD Anomaly'])

#         # Forecast Data: following year
#         data_f = {'Year': df['Year'][training_size:],
#                 'Freeze-up': df['Freeze-up'][training_size:],
#                 'Freeze-up Anomaly': df['Freeze-up Anomaly'][training_size:],
#                 'May Ta_avg Anomaly': df['May Ta_avg Anomaly'][training_size:],
#                 'Jan. PDO Anomaly': df['Jan. PDO Anomaly'][training_size:],
#                 'Sept. NAO Anomaly': df['Sept. NAO Anomaly'][training_size:],
#                 'Feb. PDO Anomaly': df['Feb. PDO Anomaly'][training_size:],
#                 'Apr. Snowfall Anomaly': df['Apr. Snowfall Anomaly'][training_size:],
#                 'Apr. NAO Anomaly': df['Apr. NAO Anomaly'][training_size:],
#                 'Apr. aFDD Anomaly': df['Apr. aFDD Anomaly'][training_size:],
#                 'Nov. Snowfall Anomaly': df['Nov. Snowfall Anomaly'][training_size:],
#                 'Nov. Ta_avg Anomaly': df['Nov. Ta_avg Anomaly'][training_size:],
#                 'Nov. SLP Anomaly': df['Nov. SLP Anomaly'][training_size:],
#                 'Nov. aTDD Anomaly': df['Nov. aTDD Anomaly'][training_size:]
#                 }
#         df_f = pd.DataFrame(data_f,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly','Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. SLP Anomaly','Nov. aTDD Anomaly'])

#         xh = df_h[x_model]
#         xf = df_f[x_model]

#         yh = df_h['Freeze-up Anomaly']
#         yf = df_f['Freeze-up Anomaly']

#         # Hindcast
#         xh = sm.add_constant(xh) # adding a constant
#         model = sm.OLS(yh, xh).fit()
#         yh_hat = model.predict(xh)

#         if verbose:
#             print_model = model.summary()
#             print(print_model)

#         # Forecast
#         xf = sm.add_constant(xf) # adding a constant
#         yf_hat = model.predict(xf)



#     # Evaluate Hindcast and Forecast:
#     fig, ax = plt.subplots()
#     ax.plot(np.array(df['Year']),np.array(df['Freeze-up Anomaly']),'o-')

#     if not rolling_training:
#         std_h = np.nanstd(yh-yh_hat)
#         std_f = np.nanstd(yf-yf_hat)

#         mae_h = np.nanmean(np.abs(yh-yh_hat))
#         mae_f = np.nanmean(np.abs(yf-yf_hat))

#         rmse_h = np.sqrt(np.nanmean((yh-yh_hat)**2.))
#         rmse_f = np.sqrt(np.nanmean((yf-yf_hat)**2.))

#         _, Rsqr_h = linear_fit(yh,yh_hat)
#         _, Rsqr_f = linear_fit(yf,yf_hat)

#         Rsqr_adj_h = 1-(((1-Rsqr_h)*(yh.shape[0]-1))/(yh.shape[0]-model.df_model-1))
#         Rsqr_adj_f = 1-(((1-Rsqr_f)*(yf.shape[0]-1))/(yf.shape[0]-model.df_model-1))

#         print('------------------------------------')
#         print('Hindcast: 1992-2006')
#         print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
#         print(mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h)
#         print('')
#         print('Forecast: 2007-2016')
#         print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
#         print(mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f)
#         print('------------------------------------')

#         ax.plot(df_f['Year'],yf_hat, 'o--', color= plt.get_cmap('tab20')(4))
#         ax.plot(np.array(df_h['Year']),yh_hat, '-', color= plt.get_cmap('tab20')(2))
#         ax.set_xlabel('Year')
#         ax.set_label('Freeze-up Anomaly')
#     else:

#         std_h = np.nanstd(yh[0,:]-yh_hat[0,:])
#         std_f = np.nanstd(yf-yf_hat)

#         mae_h = np.nanmean(np.abs(yh[0,:]-yh_hat[0,:]))
#         mae_f = np.nanmean(np.abs(yf-yf_hat))

#         rmse_h = np.sqrt(np.nanmean((yh[0,:]-yh_hat[0,:])**2.))
#         rmse_f = np.sqrt(np.nanmean((yf-yf_hat)**2.))

#         print(np.squeeze(yf).shape, np.squeeze(yf_hat).shape)
#         print(yf)
#         print(yf_hat)
#         # _, Rsqr_h = linear_fit(np.squeeze(yh[0,:]),np.squeeze(yh_hat[0,:]))
#         # _, Rsqr_f = linear_fit(np.squeeze(yf)[0:-1],np.squeeze(yf_hat)[0:-1])

#         _, Rsqr_h = linear_fit(np.squeeze(yh[0,:]),np.squeeze(yh_hat[0,:]))
#         _, Rsqr_f = linear_fit(yf,yf_hat)


#         Rsqr_adj_h = 1-(((1-Rsqr_h)*(yh.shape[1]-1))/(yh.shape[1]-model.df_model-1))
#         Rsqr_adj_f = 1-(((1-Rsqr_f)*(yf.shape[0]-1))/(yf.shape[0]-model.df_model-1))

#         print('------------------------------------')
#         print('First Hindcast: 1992-2006')
#         print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
#         print(mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h)
#         print('')
#         print('Forecast: 2007-2016')
#         print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
#         print(mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f)
#         print('------------------------------------')

#         ax.plot(xf[:,0],yf_hat, 'o--', color= plt.get_cmap('tab20')(4))
#         ax.plot(np.array(df['Year'][0:training_size]),yh_hat[0,:], '-', color= plt.get_cmap('tab20')(2))
#         ax.set_xlabel('Year')
#         ax.set_ylabel('Freeze-up Anomaly')
#     # plt.figure();plt.plot(yf,yf_hat,'o')

#%%
years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020]

fp = '../../../data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)

#%%
end_dates_arr = np.zeros((len(years),4))*np.nan
for iyear,year in enumerate(years):
    doy_sep1 = (dt.date(int(year),9,1)-(dt.date(int(year),1,1))).days + 1
    doy_oct1 = (dt.date(int(year),10,1)-(dt.date(int(year),1,1))).days + 1
    doy_nov1 = (dt.date(int(year),11,1)-(dt.date(int(year),1,1))).days + 1
    doy_dec1 = (dt.date(int(year),12,1)-(dt.date(int(year),1,1))).days + 1
    doy_dec15 = (dt.date(int(year),12,15)-(dt.date(int(year),1,1))).days + 1
    # end_dates_arr[iyear,0] = doy_dec15
    end_dates_arr[iyear,0] = doy_dec1
    end_dates_arr[iyear,1] = doy_nov1
    end_dates_arr[iyear,2] = doy_oct1
    end_dates_arr[iyear,3] = doy_sep1
enddate_labels = ['Dec. 1st', 'Nov. 1st', 'Oct. 1st', 'Sept. 1st']

p_critical = 0.05

deseasonalize = False
detrend = True
anomaly = 'linear'

nboot = 1

#window_arr = 2*2**np.arange(0,8) # For powers of 2
# window_arr = np.arange(1,39)*7 # For weeks, up to Jan 1st
# window_arr = np.arange(1,3)*7
window_arr = np.arange(1,9)*30 # For months

#%%
# LOAD FREEZEUP DATES OR FIND FROM TWATER TIME SERIES
water_name_list = ['Longueuil_cleaned_filled']
station_labels = ['Longueuil']
station_type = 'cities'

load_freezeup = False
freezeup_opt = 1
month_start_day = 1

# OPTION 1
if freezeup_opt == 1:
    def_opt = 1
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = False
    T_thresh = 0.75
    dTdt_thresh = 0.25
    d2Tdt2_thresh = 0.25
    nd = 1

# OPTION 2
if freezeup_opt == 2:
    def_opt = 3
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = True
    sig_dog = 3.5
    T_thresh = 3.
    dTdt_thresh = 0.15
    d2Tdt2_thresh = 0.15
    # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
    # d2Tdt2_thresh = 0.20
    nd = 30

if load_freezeup:
    print('ERROR: STILL NEED TO DEFINE THIS FROM SAVED ARRAYS...')
else:
    freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
    freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
    freezeup_temp = np.zeros((len(years),len(water_name_list)))*np.nan

    Twater = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_DoG1 = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_DoG2 = np.zeros((len(time),len(water_name_list)))*np.nan

    for iloc,loc in enumerate(water_name_list):
        loc_water_loc = water_name_list[iloc]
        water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
        Twater_tmp = water_loc_data['Twater'][:,1]

        # APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST
        Twater[:,iloc] = Twater_tmp
        if loc == 'Candiac_cleaned_filled':
            Twater[:,iloc] = Twater_tmp-0.8
        if (loc == 'Atwater_cleaned_filled'):
            Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7


        # THEN FIND DTDt, D2TDt2, etc.
        Twater_tmp = Twater[:,iloc].copy()
        if round_T:
            if round_type == 'unit':
                Twater_tmp = np.round(Twater_tmp.copy())
            if round_type == 'half_unit':
                Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
        if smooth_T:
            Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

        dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

        dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
        dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
        dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

        Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
        Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

        if Gauss_filter:
            Twater_DoG1[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
            Twater_DoG2[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

        # THEN FIND FREEZEUP ACCORDING TO CHOSEN OPTION:
        if def_opt == 3:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_DoG1[:,iloc],Twater_DoG2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
            freezeup_temp[:,iloc] = ftw
        else:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
            freezeup_temp[:,iloc] = ftw

        # FINALLY, TRANSFORM FREEZEUP FROM DATE FORMAT TO DOY FORMAT:
        for iyr,year in enumerate(years):
            if ~np.isnan(freezeup_dates[iyr,0,iloc]):
                fd_yy = int(freezeup_dates[iyr,0,iloc])
                fd_mm = int(freezeup_dates[iyr,1,iloc])
                fd_dd = int(freezeup_dates[iyr,2,iloc])

                fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
                if fd_doy < 60: fd_doy += 365

                freezeup_doy[iyr,iloc]=fd_doy

# Average all stations to get mean freezeup DOY for each year
avg_freezeup_doy = np.round(np.nanmean(freezeup_doy,axis=1))

# end_dates_arr = np.zeros((len(years),1))*np.nan
# end_dates_arr[:,0] = avg_freezeup_doy

#%%
# MAKE TWATER INTO VARIABLE

Twater_varnames = ['Avg. water temp.']
Twater_vars = np.zeros((len(time),len(Twater_varnames)))*np.nan
Twater_vars[:,0] = np.nanmean(Twater,axis=1)
Twater_vars = np.squeeze(Twater_vars)

if deseasonalize:
    Nwindow = 31
    Twater_vars = deasonalize_ts(Nwindow,Twater_vars,['Twater'],'all_time',time,years)

Twater_vars_all = np.zeros((1,len(years),len(window_arr),end_dates_arr.shape[1],2,1))*np.nan
for iend in range(end_dates_arr.shape[1]):
    Twater_vars_all[:,:,:,iend,:,0] = get_window_vars(np.expand_dims(Twater_vars,axis=1),['Avg. Twater'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)


Twater_vars_all = np.squeeze(Twater_vars_all[:,:,:,:,0,:])

#%%
# LOAD WEATHER DATA

fp2 = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/detected_freezeup_correlation_analysis/'

weather_data1 = np.load(fp2+'weather_vars_all_monthly.npz',allow_pickle='TRUE')
weather_data2 = np.load(fp2+'weather_vars2_all_monthly.npz',allow_pickle='TRUE')
# avg_freezeup_doy2 = weather_data1['avg_freezeup_doy']

vars_all1 = weather_data1['weather_vars']
varnames1 = weather_data1['varnames']
locnames1 = weather_data1['locnames']
vars_all1 = vars_all1[:,:,:,:,:,1:] # Remove NCEI data
locnames1 = locnames1[1:]# Remove NCEI data

vars_all2 = weather_data2['weather_vars']
varnames2 = weather_data2['varnames']
locnames2 = weather_data2['locnames']

vars_all1 = vars_all1[:,:,:,:,:,0] # Select location: MLO+OR
vars_all2 = vars_all2[:,:,:,:,:,0] # Select location: MLO+OR
locname = locnames1[0]

# MERGE ALL WEATHER DATA TOGETHER:
vars_all = np.zeros((vars_all1.shape[0]+vars_all2.shape[0],vars_all1.shape[1],vars_all1.shape[2],vars_all1.shape[3]))
vars_all[0:vars_all1.shape[0],:,:,:] = vars_all1[:,:,:,:,0]
vars_all[vars_all1.shape[0]:,:,:,:] = vars_all2[:,:,:,:,0]

varnames = [n for n in varnames1]+[n for n in varnames2]


#%%
# LOAD NAO DATA

NAO_data = np.load(fp+'NAO_index_NOAA/NAO_index_NOAA_monthly.npz',allow_pickle='TRUE')
NAO_vars = NAO_data['NAO_data']
NAO_varnames = ['NAO']

if deseasonalize:
    Nwindow = 31
    NAO_vars = deasonalize_ts(Nwindow,NAO_vars,NAO_varnames,'all_time',time,years)

NAO_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
NAO_max_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
NAO_min_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
# boot_iyears_all = np.zeros((len(years),end_dates_arr.shape[1],nboot))*np.nan
for iend in range(end_dates_arr.shape[1]):
    NAO_vars_all[:,:,iend,:]= get_window_monthly_vars(NAO_vars,['Avg. monthly NAO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
    NAO_max_vars_all[:,:,iend,:]= get_window_monthly_vars(NAO_vars,['Max. monthly NAO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
    NAO_min_vars_all[:,:,iend,:]= get_window_monthly_vars(NAO_vars,['Min. monthly NAO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

NAO_vars_all = np.expand_dims(np.expand_dims(NAO_vars_all[:,:,:,:],axis=-1),axis=0)
NAO_max_vars_all = np.expand_dims(np.expand_dims(NAO_max_vars_all[:,:,:,:],axis=-1),axis=0)
NAO_min_vars_all = np.expand_dims(np.expand_dims(NAO_min_vars_all[:,:,:,:],axis=-1),axis=0)

NAO_vars_all = np.squeeze(NAO_vars_all[:,:,:,:,0,:])


#%%
# LOAD PDO DATA

fn = 'PDO_index_NOAA_monthly_ersstv3.npz'
# fn = 'PDO_index_NOAA_monthly_ersstv5.npz'
# fn = 'PDO_index_NOAA_monthly_hadisst1.npz'
PDO_data = np.load(fp+'PDO_index_NOAA/'+fn,allow_pickle='TRUE')
PDO_vars = PDO_data['PDO_data']
PDO_varnames = ['PDO']

if deseasonalize:
    Nwindow = 31
    PDO_vars = deasonalize_ts(Nwindow,PDO_vars,PDO_varnames,'all_time',time,years)

PDO_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
PDO_max_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
PDO_min_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
# boot_iyears_all = np.zeros((len(years),end_dates_arr.shape[1],nboot))*np.nan
for iend in range(end_dates_arr.shape[1]):
    PDO_vars_all[:,:,iend,:]= get_window_monthly_vars(PDO_vars,['Avg. monthly PDO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
    PDO_max_vars_all[:,:,iend,:]= get_window_monthly_vars(PDO_vars,['Max. monthly PDO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
    PDO_min_vars_all[:,:,iend,:]= get_window_monthly_vars(PDO_vars,['Min. monthly PDO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

PDO_vars_all = np.expand_dims(np.expand_dims(PDO_vars_all[:,:,:,:],axis=-1),axis=0)
PDO_max_vars_all = np.expand_dims(np.expand_dims(PDO_max_vars_all[:,:,:,:],axis=-1),axis=0)
PDO_min_vars_all = np.expand_dims(np.expand_dims(PDO_min_vars_all[:,:,:,:],axis=-1),axis=0)

PDO_vars_all = np.squeeze(PDO_vars_all[:,:,:,:,0,:])


#%%
nvars =vars_all.shape[0]
nyears = vars_all.shape[1]
nwindows = vars_all.shape[2]
nend = vars_all.shape[3]


#%%
# DETREND VARIABLES
if detrend:
    vars_all_detrended = np.zeros(vars_all.shape)*np.nan
    NAO_vars_all_detrended = np.zeros(NAO_vars_all.shape)*np.nan
    PDO_vars_all_detrended = np.zeros(PDO_vars_all.shape)*np.nan
    Twater_vars_all_detrended = np.zeros(Twater_vars_all.shape)*np.nan

    for ivar in range(nvars):
        for iw in range(nwindows):
            for iend in range(nend):
                xvar = vars_all[ivar,:,iw,iend]
                yvar = avg_freezeup_doy

                vars_all_detrended[ivar,:,iw,iend], avg_freezeup_doy_detrended = detrend_ts(xvar,yvar,years,anomaly)

    for iw in range(nwindows):
         for iend in range(nend):
             xvar = NAO_vars_all[:,iw,iend]
             yvar = PDO_vars_all[:,iw,iend]

             NAO_vars_all_detrended[:,iw,iend], PDO_vars_all_detrended[:,iw,iend] = detrend_ts(xvar,yvar,years,anomaly)


    for iw in range(nwindows):
        for iend in range(nend):
            Txvar = Twater_vars_all[:,iw,iend]
            Tyvar = Twater_vars_all[:,iw,iend].copy()

            Twater_vars_all_detrended[:,iw,iend], _ = detrend_ts(Txvar,Tyvar,years,anomaly)


else:
    vars_all_detrended = vars_all.copy()
    NAO_vars_all_detrended = NAO_vars_all.copy()
    PDO_vars_all_detrended = PDO_vars_all.copy()
    avg_freezeup_doy_detrended = avg_freezeup_doy.copy()
    Twater_vars_all_detrended = Twater_vars_all.copy()


#%%
# RESHAPE VARS TO HAVE 11 MONTHS DATA IN CORRECT ORDER:
weather_monthly = np.zeros((nvars,nyears,11))
NAO_monthly = np.zeros((nyears,11))
PDO_monthly = np.zeros((nyears,11))
Twater_monthly = np.zeros((nyears,11))

for ivar in range(nvars):
    weather_monthly[ivar,:,3:]= np.fliplr(vars_all_detrended[ivar,:,:,0])
    weather_monthly[ivar,:,0:3]= np.fliplr(vars_all_detrended[ivar,:,5:8,-1])

NAO_monthly[:,3:]= np.fliplr(NAO_vars_all_detrended[:,:,0])
NAO_monthly[:,0:3]= np.fliplr(NAO_vars_all_detrended[:,5:8,-1])

PDO_monthly[:,3:]= np.fliplr(PDO_vars_all_detrended[:,:,0])
PDO_monthly[:,0:3]= np.fliplr(PDO_vars_all_detrended[:,5:8,-1])

Twater_monthly[:,3:]= np.fliplr(Twater_vars_all_detrended[:,:,0])
Twater_monthly[:,0:3]= np.fliplr(Twater_vars_all_detrended[:,5:8,-1])

#%%
# KEEP ONLY 1992-2016
weather_monthly = weather_monthly[:,1:26,:]
NAO_monthly = NAO_monthly[1:26,:]
PDO_monthly = PDO_monthly[1:26,:]
Twater_monthly = Twater_monthly[1:26,:]
years = years[1:26]
avg_freezeup_doy_detrended = avg_freezeup_doy_detrended[1:26]
avg_freezeup_doy = avg_freezeup_doy[1:26]


#%%
# SELECT PREDICTORS
May_Ta_mean = weather_monthly[2,:,4]
Jan_PDO = PDO_monthly[:,0]
Sept_NAO = NAO_monthly[:,8]

Aug_NAO = NAO_monthly[:,7]
Sept_RH = weather_monthly[14,:,8]
Sept_clouds = weather_monthly[12,:,8]
Oct_wind = weather_monthly[8,:,9]

Feb_PDO = PDO_monthly[:,1]
Apr_snow = weather_monthly[11,:,3]
Apr_NAO = NAO_monthly[:,3]
Apr_FDD = weather_monthly[4,:,3]

Nov_snow = weather_monthly[11,:,10]
Nov_Ta_mean = weather_monthly[2,:,10]
Nov_SLP = weather_monthly[7,:,10]
Nov_TDD = weather_monthly[3,:,10]
Nov_FDD = weather_monthly[4,:,10]

Oct_Twater = Twater_monthly[:,9]
Nov_Twater = Twater_monthly[:,10]
May_Twater = Twater_monthly[:,4]

#%%
# Prepare predictor data in DataFrame
data = {'Year': years,
        'Freeze-up': avg_freezeup_doy,
        'Freeze-up Anomaly': avg_freezeup_doy_detrended,
        'May Ta_avg Anomaly': May_Ta_mean,
        'Jan. PDO Anomaly': Jan_PDO,
        'Sept. NAO Anomaly': Sept_NAO,
        'Feb. PDO Anomaly': Feb_PDO,
        'Apr. Snowfall Anomaly': Apr_snow,
        'Apr. NAO Anomaly': Apr_NAO,
        'Apr. aFDD Anomaly': Apr_FDD,
        'Nov. Snowfall Anomaly': Nov_snow,
        'Nov. Ta_avg Anomaly': Nov_Ta_mean,
        'Nov. SLP Anomaly': Nov_SLP,
        'Nov. aTDD Anomaly': Nov_TDD,
        'Nov. aFDD Anomaly': Nov_FDD,
        'Aug. NAO Anomaly': Aug_NAO,
        'Sept. RH Anomaly': Sept_RH,
        'Sept. Cloud Anomaly': Sept_clouds,
        'Oct. windspeed Anomaly': Oct_wind,
        'Oct. Twater Anomaly': Oct_Twater,
        'Nov. Twater Anomaly': Nov_Twater,
        'May Twater Anomaly': May_Twater
        }


df = pd.DataFrame(data,columns=['Year','Freeze-up','Freeze-up Anomaly','May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly',
                                'Feb. PDO Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly','Apr. aFDD Anomaly','Nov. Snowfall Anomaly',
                                'Nov. Ta_avg Anomaly','Nov. SLP Anomaly','Nov. aTDD Anomaly','Nov. aFDD Anomaly',
                                'Aug. NAO Anomaly',
                                'Sept. RH Anomaly',
                                'Sept. Cloud Anomaly',
                                'Oct. windspeed Anomaly',
                                'Oct. Twater Anomaly',
                                'Nov. Twater Anomaly',
                                'May Twater Anomaly'])

nyears = years.shape[0]
training_size = 15



#%%
# # MODEL 1:
# x_model = ['Year']
# # freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)
#%%


# rolling_training = True
# x_model = ['Year','Oct. windspeed Anomaly','Jan. PDO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)


# rolling_training = True
# x_model = ['Year','Aug. NAO Anomaly','Apr. NAO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
#

#%%

fig, ax = plt.subplots()
ax.set_xlabel('Year')
ax.set_label('Freeze-up DOY')
# xall = np.array(df['Year']) # All years
# yall = np.array(df['Freeze-up']) # Observed FUD for all years
# xf = xf[0:-1,0] # Years of forecast period
# yf_true = yf # Observed FUD during forecast period
# yf = yf_hat[0:-1] # Predictions during forecast period
# xh = np.array(df['Year'][0:training_size]) # Years of hindcast period
# yh = yh_hat[0,:] # Predictions during hindcast period

rolling_training = True
x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
xall,yall,xf,yf_true,yf,xh,yh = freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(2))
ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(3))

# rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly', 'Nov. Ta_avg Anomaly']
# xall,yall,xf,yf_true,yf,xh,yh = freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
# ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(4))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(5))

rolling_training = True
x_model = ['Year','Oct. windspeed Anomaly','Apr. Snowfall Anomaly', 'Nov. Ta_avg Anomaly']
xall,yall,xf,yf_true,yf,xh,yh = freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(6))
ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(7))

rolling_training = True
x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Nov. aTDD Anomaly']
xall,yall,xf,yf_true,yf,xh,yh = freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(8))
ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(9))


rolling_training = True
x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly', 'Apr. NAO Anomaly','Nov. Twater Anomaly']
xall,yall,xf,yf_true,yf,xh,yh = freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
ax.plot(xall,yall,'o-',color='k')
ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(10))
ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(11))

#%%
# 'May Ta_avg Anomaly': May_Ta_mean,
# 'Jan. PDO Anomaly': Jan_PDO,
# 'Sept. NAO Anomaly': Sept_NAO,
# 'Feb. PDO Anomaly': Feb_PDO,
# 'Apr. Snowfall Anomaly': Apr_snow,
# 'Apr. NAO Anomaly': Apr_NAO,
# 'Apr. aFDD Anomaly': Apr_FDD,
# 'Nov. Snowfall Anomaly': Nov_snow,
# 'Nov. Ta_avg Anomaly': Nov_Ta_mean,
# 'Nov. SLP Anomaly': Nov_SLP,
# 'Nov. aTDD Anomaly': Nov_TDD,
# 'Nov. aFDD Anomaly': Nov_FDD,
# 'Aug. NAO Anomaly': Aug_NAO,
# 'Sept. RH Anomaly': Sept_RH,
# 'Sept. Cloud Anomaly': Sept_clouds,
# 'Oct. windspeed Anomaly': Oct_wind

# #%%

# rolling_training = True
# x_model = ['Apr. Snowfall Anomaly', 'Apr. NAO Anomaly','Nov. Twater Anomaly']
# xall,yall,xf,yf_true,yf,xh,yh = freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
# ax.plot(xall,yall,'o-',color='k')
# ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(10))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(11))

# rolling_training = True
# x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly', 'Apr. NAO Anomaly','Nov. Twater Anomaly']
# xall,yall,xf,yf_true,yf,xh,yh = freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
# ax.plot(xall,yall,'o-',color='k')
# ax.plot(xf,yf, 'o:', color= plt.get_cmap('tab20')(10))
# ax.plot(xh,yh, '-', color= plt.get_cmap('tab20')(11))


#%%
# rolling_training = True
# x_model = ['Year','Oct. windspeed Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# #%%
# # rolling_training = True
# x_model = ['Oct. windspeed Anomaly','Apr. NAO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
# #%%
# # rolling_training = True
# x_model = ['Year','Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
# #%%
# # rolling_training = True
# x_model = ['Year','Oct. windspeed Anomaly','Apr. Snowfall Anomaly','Apr. NAO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)



#%%%

# rolling_training = True
# x_model = ['Oct. windspeed Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Oct. windspeed Anomaly','Apr. aFDD Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Oct. windspeed Anomaly','Apr. NAO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Oct. windspeed Anomaly','Apr. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Oct. windspeed Anomaly','Apr. NAO Anomaly','Apr. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Oct. windspeed Anomaly','Apr. aFDD Anomaly','Apr. NAO Anomaly','Apr. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

#%%

# rolling_training = True
# x_model = ['Nov. Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Year','Nov. Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly','Nov. Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# # rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly','Nov. Ta_avg Anomaly','Oct. windspeed Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

#%%
# rolling_training = True
# x_model = ['Apr. Snowfall Anomaly','Nov. Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly','Nov. aFDD Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly','Nov. aTDD Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly','Feb. PDO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly','Jan. PDO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)
#
# rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly', 'Sept. NAO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)


# x_model = ['Apr. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# x_model = ['Nov. Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)





#%%
# # rolling_training = True
# # x_model = ['Year','Apr. Snowfall Anomaly', 'Jan. PDO Anomaly', 'Feb. PDO Anomaly']
# # freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# rolling_training = True
# x_model = ['Apr. Snowfall Anomaly', 'May Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly', 'May Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# #%%
# rolling_training = True
# x_model = ['Year','Apr. Snowfall Anomaly','Apr. aFDD Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)


# #%%
# rolling_training = True
# x_model = ['Apr. Snowfall Anomaly','Nov. Ta_avg Anomaly','Nov. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)



#%%

# MODEL 2:
# x_model = ['Year','May Ta_avg Anomaly']
# # freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)

# # MODEL 3:
# x_model = ['Year','Feb. PDO Anomaly']
# # freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)

# # MODEL 3:
# x_model = ['Year','Jan. PDO Anomaly']
# # freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)


# # MODEL 4:
# x_model = ['Feb. PDO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)

# # MODEL 5:
# x_model = ['Sept. NAO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)

# MODEL 6:
# x_model = ['Apr. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)

# # MODEL 7:
# x_model = ['Apr. NAO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)



#%%
# # # # MODEL 8:
# x_model = [ 'Apr. aFDD Anomaly']
# # freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=False)
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=True)


#%%



# rolling_training = False

# # MODEL 2:
# x_model = ['May Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)


# # MODEL 3:
# x_model = ['May Ta_avg Anomaly','Jan. PDO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)


# # MODEL 3:
# x_model = ['May Ta_avg Anomaly','Jan. PDO Anomaly','Sept. NAO Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)


# # MODEL 3:
# x_model = ['May Ta_avg Anomaly','Jan. PDO Anomaly','Apr. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)


# # MODEL 3:
# x_model = ['May Ta_avg Anomaly','Jan. PDO Anomaly','Apr. aFDD Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)





#%%
# MODEL 3:

# rolling_training = True

# x_model = ['Nov. Snowfall Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# x_model = ['Nov. Ta_avg Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# x_model = ['Nov. aTDD Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)

# x_model = ['Nov. SLP Anomaly']
# freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,rolling_training=rolling_training)




#%%

