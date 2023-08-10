#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:45:58 2021

@author: Amelie
"""
import numpy as np
import scipy
from scipy import ndimage

import pandas
from statsmodels.formula.api import ols

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import running_nanmean,find_freezeup_Tw,season_mask
from functions import linear_fit, rolling_climo, r_confidence_interval

#%%

def get_window_vars(window_type,vars_in,names_in,end_dates,window_arr,years,time,month_start_day=1):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),len(window_arr)))*np.nan
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
                if window_type == 'increasing':
                    ifd = np.where(doy_arr == end_dates[iyr])[0][0]
                    iw0 = ifd-w
                if window_type == 'moving':
                    ied = np.where(doy_arr == end_dates[iyr])[0][0]
                    ifd = ied-(iw)*(window_size)
                    iw0 = ied-(iw+1)*(window_size)

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,iw] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'): #& (varname[-2:] != 'DD'):
                        vars_out[ivar,iyr,iw] = np.nansum(var_year[iw0:ifd])

    return vars_out


def get_window_monthly_vars(vars_in,names_in,end_dates_in,window_arr,years_in,time,month_start_day=1):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years_in),len(window_arr),2))*np.nan

    years = years_in
    end_dates = end_dates_in

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

    return vars_out


#%%

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

def multilocs_rollingcorr_scatter_plot(window_list,xvars,yvar,varnames,locnames,ystart,yend,pc,detrend=False,anomaly_type='linear'):

    plot_colors = [plt.get_cmap('tab20b')(15),plt.get_cmap('tab20b')(14),plt.get_cmap('tab20b')(13),plt.get_cmap('tab20b')(12)]

    nlocs = xvars.shape[-1]
    nwindows = len(window_list)

    yr_start = int(np.where(years == ystart)[0][0])
    yr_end = int(np.where(years == yend)[0][0])

    if len(yvar.shape) == 1:
        yvar = np.array([yvar,]*nwindows).transpose() # Repeat the same values for all seasons

    for ivar,var1 in enumerate(varnames):
        fig,ax = plt.subplots(nrows=4,ncols=nlocs,figsize=((nlocs)*(8/5.),8),sharey='row')
        plt.suptitle(var1)

        if nlocs > 1:

            for iloc in range(nlocs):

                for iwindow in range(nwindows):
                    x_fit = xvars[ivar,yr_start:yr_end,iwindow,iloc]
                    y_fit = yvar[yr_start:yr_end,iwindow]

                    if detrend:
                        if anomaly_type == 'linear':
                            [mx,bx],_ = linear_fit(years[yr_start:yr_end], x_fit)
                            [my,by],_ = linear_fit(years[yr_start:yr_end], y_fit)
                            x_trend = mx*years[yr_start:yr_end] + bx
                            y_trend = my*years[yr_start:yr_end] + by

                            x_fit = x_fit-x_trend
                            y_fit = y_fit-y_trend

                        if anomaly_type == 'mean':
                            x_mean = np.nanmean(x_fit)
                            y_mean = np.nanmean(y_fit)

                            x_fit = x_fit-x_mean
                            y_fit = y_fit-y_mean

                    ax[iwindow,iloc].plot(x_fit,y_fit,'o',color=plot_colors[iwindow],alpha=0.5)
                    xvar_min = np.nanmin(x_fit)
                    xvar_max = np.nanmax(x_fit)
                    xvar_range = xvar_max-xvar_min
                    ax[iwindow,iloc].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
                    yvar_min = np.nanmin(y_fit)
                    yvar_max = np.nanmax(y_fit)
                    yvar_range = yvar_max-yvar_min
                    ax[iwindow,iloc].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

                    mask_x = ~np.isnan(x_fit)
                    mask_y = ~np.isnan(y_fit)
                    x_fit = x_fit[mask_x&mask_y]
                    y_fit = y_fit[mask_x&mask_y]
                    lincoeff, Rsqr = linear_fit(x_fit,y_fit)

                    data = pandas.DataFrame({'x': x_fit, 'y': y_fit})
                    model = ols("y ~ x", data).fit() # Fit the model to get p-value from F-test

                    if xvar_range > 0:
                        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.,xvar_range/10.)
                        yplot = lincoeff[0]*xplot + lincoeff[1]
                    # if np.round(Rsqr,2) >=0.1:
                    if model.f_pvalue <= pc:
                        ax[iwindow,iloc].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'R$^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.15,0.15,0.15], fontweight='bold')
                        ax[iwindow,iloc].plot(xplot,yplot,'-',color=[0.15,0.15,0.15])
                    else:
                        ax[iwindow,iloc].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'R$^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])
                        ax[iwindow,iloc].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

                    if iloc ==0:
                        ax[iwindow,iloc].set_ylabel('Previous\n'+'%3i'%window_list[iwindow]+' days',fontsize=10)
                    if iwindow == 0:
                        ax[iwindow,iloc].xaxis.set_label_position('top')
                        ax[iwindow,iloc].set_xlabel(locnames[iloc],fontsize=10)

            if (nlocs)*(8/5.) > 12.75:
                plt.subplots_adjust(left=0.063,right=0.9508)

        else:
            iloc = 0
            for iwindow in range(4):
                x_fit = xvars[ivar,yr_start:yr_end,iwindow,iloc]
                y_fit = yvar[yr_start:yr_end,iwindow]

                if detrend:
                    if anomaly_type == 'linear':
                        [mx,bx],_ = linear_fit(years[yr_start:yr_end], x_fit)
                        [my,by],_ = linear_fit(years[yr_start:yr_end], y_fit)
                        x_trend = mx*years[yr_start:yr_end] + bx
                        y_trend = my*years[yr_start:yr_end] + by

                        x_fit = x_fit-x_trend
                        y_fit = y_fit-y_trend

                    if anomaly_type == 'mean':
                        x_mean = np.nanmean(x_fit)
                        y_mean = np.nanmean(y_fit)

                        x_fit = x_fit-x_mean
                        y_fit = y_fit-y_mean

                ax[iwindow].plot(x_fit,y_fit,'o',color=plot_colors[iwindow],alpha=0.5)
                xvar_min = np.nanmin(x_fit)
                xvar_max = np.nanmax(x_fit)
                xvar_range = xvar_max-xvar_min
                ax[iwindow].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
                yvar_min = np.nanmin(y_fit)
                yvar_max = np.nanmax(y_fit)
                yvar_range = yvar_max-yvar_min
                ax[iwindow].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

                mask_x = ~np.isnan(x_fit)
                mask_y = ~np.isnan(y_fit)
                x_fit = x_fit[mask_x&mask_y]
                y_fit = y_fit[mask_x&mask_y]
                lincoeff, Rsqr = linear_fit(x_fit,y_fit)

                data = pandas.DataFrame({'x': x_fit, 'y': y_fit})
                model = ols("y ~ x", data).fit() # Fit the model to get p-value from F-test

                if xvar_range > 0:
                    xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.,xvar_range/10.)
                    yplot = lincoeff[0]*xplot + lincoeff[1]
                # if np.round(Rsqr,2) >=0.1:
                if model.f_pvalue <= pc:
                    ax[iwindow].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'R$^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.15,0.15,0.15], fontweight='bold')
                    ax[iwindow].plot(xplot,yplot,'-',color=[0.15,0.15,0.15])
                else:
                    ax[iwindow].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'R$^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])
                    ax[iwindow].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

                if iloc ==0:
                    ax[iwindow].set_ylabel('Previous\n'+'%3i'%window_list[iwindow]+' days',fontsize=10)
                if iwindow == 0:
                    ax[iwindow].xaxis.set_label_position('top')
                    ax[iwindow].set_xlabel(locnames[iloc],fontsize=10)



def multilocs_rollingcorr_correlation_plot(window_list,xvars_mean,yvar,varnames,locnames,ystart,yend,pc,detrend=False,anomaly_type='linear',enddate_str=''):

    plot_colors = [plt.get_cmap('tab20b')(0),plt.get_cmap('tab20b')(4),plt.get_cmap('tab20b')(8),plt.get_cmap('tab20b')(12)]

    nlocs = xvars_mean.shape[-1]
    nplots = xvars_mean.shape[-2]
    nwindows = len(window_list)

    yr_start = int(np.where(years == ystart)[0][0])
    yr_end = int(np.where(years == yend)[0][0])

    if len(yvar.shape) == 1:
        yvar = np.array([yvar,]*nwindows).transpose() # Repeat the same values for all seasons

    for ivar,var1 in enumerate(varnames):
        fig,ax = plt.subplots(nrows=nlocs,ncols=1,figsize=(5,(nlocs)*(8/5.)),sharex=True,sharey=True)
        plt.suptitle(var1)

        if nlocs > 1:

            for iloc in range(nlocs):

                for ip in range(nplots):
                    r = np.zeros((nwindows))*np.nan
                    for iw,w in enumerate(window_list):
                        x_fit = xvars_mean[ivar,yr_start:yr_end,iw,ip,iloc]
                        y_fit = yvar[yr_start:yr_end,iw]

                        if detrend:
                            if anomaly_type == 'linear':
                                [mx,bx],_ = linear_fit(years[yr_start:yr_end], x_fit)
                                [my,by],_ = linear_fit(years[yr_start:yr_end], y_fit)
                                x_trend = mx*years[yr_start:yr_end] + bx
                                y_trend = my*years[yr_start:yr_end] + by

                                x_fit = x_fit-x_trend
                                y_fit = y_fit-y_trend

                            if anomaly_type == 'mean':
                                x_mean = np.nanmean(x_fit)
                                y_mean = np.nanmean(y_fit)

                                x_fit = x_fit-x_mean
                                y_fit = y_fit-y_mean


                        mask_x = ~np.isnan(x_fit)
                        mask_y = ~np.isnan(y_fit)
                        x_fit = x_fit[mask_x&mask_y]
                        y_fit = y_fit[mask_x&mask_y]
                        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
                        rc_m1, rc_p1 = r_confidence_interval(0,pc,len(x_fit),tailed='one')
                        rc_m2, rc_p2 = r_confidence_interval(0,pc,len(x_fit),tailed='two')

                        r[iw] = np.sqrt(Rsqr)
                        if (lincoeff[0]< 0):
                            r[iw] *= -1

                    ax[iloc].plot(window_list,r,'.-',color=plot_colors[ip])
                    ax[iloc].plot(window_list,np.ones(len(window_list))*rc_p2,':', color='gray')
                    ax[iloc].plot(window_list,np.ones(len(window_list))*rc_m2,':', color='gray')

                plt.subplots_adjust(left=0.2,right=0.9)
                ax[iloc].set_xlim(0,np.nanmax(window_list)+np.nanmax(window_list)/10.)
                ax[iloc].set_ylim(-1,1)
                ax[iloc].set_ylabel(locnames[iloc],fontsize=10)

                if iloc == nlocs-1:
                    if (window_list[1]-window_list[0]) == 7:
                        ax[iloc].set_xlabel('Previous X weeks (XW) from '+enddate_str,fontsize=10)
                        labels_list = [str(np.arange(1,len(window_list)+1,2)[i])+'W' for i in range(len(np.arange(1,len(window_list)+1,2)))]
                        labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                    if (window_list[1]-window_list[0]) == 30:
                        ax[iloc].set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
                        labels_list = [str(np.arange(1,len(window_list)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_list)+1,2)))]
                        labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                    ax[iloc].set_xticks(window_list)
                    ax[iloc].set_xticklabels(labels_list)


        else:
            iloc = 0
            for ip in range(nplots):
                r = np.zeros(len(window_list))*np.nan
                for iw,w in enumerate(window_list):
                    x_fit = xvars_mean[ivar,yr_start:yr_end,iw,ip,iloc]
                    y_fit = yvar[yr_start:yr_end,iw]

                    if detrend:
                        if anomaly_type == 'linear':
                            [mx,bx],_ = linear_fit(years[yr_start:yr_end], x_fit)
                            [my,by],_ = linear_fit(years[yr_start:yr_end], y_fit)
                            x_trend = mx*years[yr_start:yr_end] + bx
                            y_trend = my*years[yr_start:yr_end] + by

                            x_fit = x_fit-x_trend
                            y_fit = y_fit-y_trend

                        if anomaly_type == 'mean':
                            x_mean = np.nanmean(x_fit)
                            y_mean = np.nanmean(y_fit)

                            x_fit = x_fit-x_mean
                            y_fit = y_fit-y_mean

                    mask_x = ~np.isnan(x_fit)
                    mask_y = ~np.isnan(y_fit)
                    x_fit = x_fit[mask_x&mask_y]
                    y_fit = y_fit[mask_x&mask_y]
                    lincoeff, Rsqr = linear_fit(x_fit,y_fit)
                    rc_m1, rc_p1 = r_confidence_interval(0,pc,len(x_fit),tailed='one')
                    rc_m2, rc_p2 = r_confidence_interval(0,pc,len(x_fit),tailed='two')

                    r[iw] = np.sqrt(Rsqr)
                    if (lincoeff[0]< 0):
                        r[iw] *= -1

                ax.plot(window_list,r,'.-',color=plot_colors[ip])
                ax.plot(window_list,np.ones(len(window_list))*rc_p2,':', color='gray')
                ax.plot(window_list,np.ones(len(window_list))*rc_m2,':', color='gray')

            plt.subplots_adjust(left=0.2,right=0.9)
            ax.set_xlim(0,np.nanmax(window_list)+np.nanmax(window_list)/10.)
            ax.set_ylim(-1,1)
            ax.set_ylabel(locnames,fontsize=10)
            # ax.set_xlabel('Previous X weeks (XW) from '+enddate_str,fontsize=10)
            # ax.set_xticks(window_arr)
            # labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'W' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
            # labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')
            # ax.set_xticklabels(labels_list)

            if (window_list[1]-window_list[0]) == 7:
                ax.set_xlabel('Previous X weeks (XW) from '+enddate_str,fontsize=10)
                labels_list = [str(np.arange(1,len(window_list)+1,2)[i])+'W' for i in range(len(np.arange(1,len(window_list)+1,2)))]
                labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

            if (window_list[1]-window_list[0]) == 30:
                ax.set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
                labels_list = [str(np.arange(1,len(window_list)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_list)+1,2)))]
                labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

            ax.set_xticks(window_list)
            ax.set_xticklabels(labels_list)


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
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)

#%%
# end_dates_arr = np.zeros((len(years),4))*np.nan
# for iyear,year in enumerate(years):
#     doy_oct1 = (dt.date(int(year),10,1)-(dt.date(int(year),1,1))).days + 1
#     doy_nov1 = (dt.date(int(year),11,1)-(dt.date(int(year),1,1))).days + 1
#     doy_dec1 = (dt.date(int(year),12,1)-(dt.date(int(year),1,1))).days + 1
#     doy_dec15 = (dt.date(int(year),12,15)-(dt.date(int(year),1,1))).days + 1
#     end_dates_arr[iyear,0] = doy_dec1
#     end_dates_arr[iyear,1] = doy_dec1
#     end_dates_arr[iyear,2] = doy_nov1
#     end_dates_arr[iyear,3] = doy_oct1
# enddate_labels = ['Dec. 1','Dec. 1st', 'Nov. 1st', 'Oct. 1st']

end_dates_arr = np.zeros((len(years),2))*np.nan
for iyear,year in enumerate(years):
    doy_dec1 = (dt.date(int(year),12,1)-(dt.date(int(year),1,1))).days + 1
    end_dates_arr[iyear,0] = doy_dec1
    end_dates_arr[iyear,1] = doy_dec1
enddate_labels = ['Dec. 1st','Dec. 1st']


p_critical = 0.01

deseasonalize = False
detrend = False
anomaly = 'linear'

#window_arr = 2*2**np.arange(0,8) # For powers of 2
# window_arr = np.arange(1,39)*7 # For weeks, up to Jan 1st
#window_arr = np.arange(1,17)*7
# window_arr = np.arange(1,9)*30 # For months
window_arr = np.arange(1,3)*30 # For months

#%%
# LOAD FREEZEUP DATES OR FIND FROM TWATER TIME SERIES
water_name_list = ['Atwater_cleaned_filled','DesBaillets_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
station_labels = ['Atwater','DesBaillets','Longueuil','Candiac']
station_type = 'cities'

load_freezeup = False
freezeup_opt = 2
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
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_DoG1[:,iloc],Twater_DoG2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
            freezeup_temp[:,iloc] = ftw
        else:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
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
# # MAKE TWATER INTO AN EXPLANATORY VARIABLE
# Twater_varnames = ['Avg. water temp.']
# Twater_vars = np.zeros((len(time),len(Twater_varnames)))*np.nan
# Twater_vars[:,0] = np.nanmean(Twater,axis=1)
# Twater_vars = np.squeeze(Twater_vars)

# if deseasonalize:
#     Nwindow = 31
#     Twater_vars = deasonalize_ts(Nwindow,Twater_vars,['Twater'],'all_time',time,years)

# Twater_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
# for iend in range(end_dates_arr.shape[1]):
#     Twater_vars_all[:,:,iend,0] = get_window_vars('moving',np.expand_dims(Twater_vars,axis=1),['Avg. Twater'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
#     Twater_vars_all[:,:,iend,1] = get_window_vars('increasing',np.expand_dims(Twater_vars,axis=1),['Avg. Twater'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

#%%

NAO_data = np.load(fp+'NAO_index_NOAA/NAO_index_NOAA_monthly.npz',allow_pickle='TRUE')
NAO_vars = NAO_data['NAO_data']
# NAO_vars = np.squeeze(NAO_vars)

NAO_varnames = ['NAO']

if deseasonalize:
    Nwindow = 31
    NAO_vars = deasonalize_ts(Nwindow,NAO_vars,NAO_varnames,'all_time',time,years)

NAO_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
for iend in range(end_dates_arr.shape[1]):
    NAO_vars_all[:,:,iend,:] = get_window_monthly_vars(NAO_vars,['Avg. monthly NAO'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
    print(np.nanmean(NAO_vars_all[:,:,iend,:]))

#%%
ystart = 1991
yend = 2019

for iend in range(end_dates_arr.shape[1]):
# for iend in range(1):
#     iend = np.where(np.array(enddate_labels) == 'Dec. 1st')[0][0]
    # iend = np.where(np.array(enddate_labels) == 'Nov. 1st')[0][0]

#    iend=1
    yvar= avg_freezeup_doy
    enddate_str = enddate_labels[iend]

    locnames=['Montreal']
    multilocs_rollingcorr_correlation_plot(window_arr,np.expand_dims(np.expand_dims(NAO_vars_all[:,:,iend,:],axis=-1),axis=0),yvar,NAO_varnames,locnames,ystart,yend,p_critical,detrend,anomaly,enddate_str)














#%%
# Show rolling correlation plots

show_corr_plots = False

if show_corr_plots:
    ystart = 1991
    yend = 2019

    for iend in range(end_dates_arr.shape[1]):
    # for iend in range(1):
    #     iend = np.where(np.array(enddate_labels) == 'Dec. 1st')[0][0]
        # iend = np.where(np.array(enddate_labels) == 'Nov. 1st')[0][0]

        yvar= avg_freezeup_doy
        enddate_str = enddate_labels[iend]

        locnames=['Montreal']
        multilocs_rollingcorr_correlation_plot(window_arr,np.expand_dims(np.expand_dims(NAO_vars_all_mean[:,:,iend,:],axis=-1),axis=0),yvar,NAO_varnames,locnames,ystart,yend,p_critical,detrend,anomaly,enddate_str)



#%%
# Show scatter plots
show_scatter_plots = False
if show_scatter_plots:
    ystart = 1991
    yend = 2019

    yvar= avg_freezeup_doy
    locnames=['Montreal']

    iend = np.where(np.array(enddate_labels) == 'Dec. 1st')[0][0]
    iend = 3

    w0, w1 = 0,4
    window_list = window_arr[w0:w1]
    multilocs_rollingcorr_scatter_plot(window_list,np.expand_dims(np.expand_dims(NAO_vars_all_mean[:,w0:w1,iend,1],axis=-1),axis=0),yvar,NAO_varnames,locnames,ystart,yend,p_critical,detrend,anomaly)

    w0, w1 = 4,8
    window_list = window_arr[w0:w1]
    multilocs_rollingcorr_scatter_plot(window_list,np.expand_dims(np.expand_dims(NAO_vars_all_mean[:,w0:w1,iend,1],axis=-1),axis=0),yvar,NAO_varnames,locnames,ystart,yend,p_critical,detrend,anomaly)

