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

                    if (varname[0:3] == 'Tot'): #& (varname[-2:] != 'DD'):
                        vars_out[ivar,iyr,iw,0] = np.nansum(var_year[iw0:ifd])

                # window_type == 'increasing':
                ifd = np.where(doy_arr == end_dates[iyr])[0][0]
                iw0 = ifd-w

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,iw,1] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'): #& (varname[-2:] != 'DD'):
                        vars_out[ivar,iyr,iw,1] = np.nansum(var_year[iw0:ifd])

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


def multilocs_rollingcorr_correlation_plot(window_list,xvars_mean,yvar,boot_arr,varnames,locnames,pc,detrend=False,anomaly_type='linear',enddate_str=''):

    plot_colors = [plt.get_cmap('tab20b')(0),plt.get_cmap('tab20b')(4),plt.get_cmap('tab20b')(8),plt.get_cmap('tab20b')(12)]

    nlocs = xvars_mean.shape[-1]
    nboot = xvars_mean.shape[-2]
    nplots = xvars_mean.shape[-3]
    nwindows = len(window_list)

    if len(yvar.shape) == 1:
        yvar = np.array([yvar,]*nwindows).transpose() # Repeat the same values for all seasons

    for ivar,var1 in enumerate(varnames):
        fig,ax = plt.subplots(nrows=nlocs,ncols=1,figsize=(5,(nlocs)*(8/5.)),sharex=True,sharey=True)
        plt.suptitle(var1)

        if nlocs > 1:

            for iloc in range(nlocs):

                for ip in range(nplots):
                    r = np.zeros((nwindows,nboot))*np.nan
                    for iw,w in enumerate(window_list):
                        for n in range(nboot):
                            x_fit = xvars_mean[ivar,:,iw,ip,n,iloc]
                            y_fit = yvar[boot_arr[:,n],iw]

                            if detrend:
                                if anomaly_type == 'linear':
                                    [mx,bx],_ = linear_fit(years[boot_arr[:,n]], x_fit)
                                    [my,by],_ = linear_fit(years[boot_arr[:,n]], y_fit)
                                    x_trend = mx*years[boot_arr[:,n]] + bx
                                    y_trend = my*years[boot_arr[:,n]] + by

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

                            r[iw,n] = np.sqrt(Rsqr)
                            if (lincoeff[0]< 0):
                                r[iw,n] *= -1

                    ax[iloc].plot(window_list,np.nanmean(r,axis=1),'.-',color=plot_colors[ip])
                    # ax.plot(window_list,np.nanmean(r,axis=1)+np.nanstd(r,axis=1),'-',linewidth=0.5,color=plot_colors[ip])
                    # ax.plot(window_list,np.nanmean(r,axis=1)-np.nanstd(r,axis=1),'-',linewidth=0.5,color=plot_colors[ip])
                    ax[iloc].fill_between(window_list,np.nanmean(r,axis=1)+np.nanstd(r,axis=1),np.nanmean(r,axis=1)-np.nanstd(r,axis=1),color=plot_colors[ip],alpha=0.15)
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
                r = np.zeros((len(window_list),nboot))*np.nan
                for iw,w in enumerate(window_list):
                    for n in range(nboot):
                        x_fit = xvars_mean[ivar,:,iw,ip,n,iloc]
                        y_fit = yvar[boot_arr[:,n],iw]

                        if detrend:
                            if anomaly_type == 'linear':
                                [mx,bx],_ = linear_fit(years[boot_arr[:,n]], x_fit)
                                [my,by],_ = linear_fit(years[boot_arr[:,n]], y_fit)
                                x_trend = mx*years[boot_arr[:,n]] + bx
                                y_trend = my*years[boot_arr[:,n]] + by

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

                        r[iw,n] = np.sqrt(Rsqr)
                        if (lincoeff[0]< 0):
                            r[iw,n] *= -1

                ax.plot(window_list,np.nanmean(r,axis=1),'.-',color=plot_colors[ip])
                # ax.plot(window_list,np.nanmean(r,axis=1)+np.nanstd(r,axis=1),'-',linewidth=0.5,color=plot_colors[ip])
                # ax.plot(window_list,np.nanmean(r,axis=1)-np.nanstd(r,axis=1),'-',linewidth=0.5,color=plot_colors[ip])
                plt.fill_between(window_list,np.nanmean(r,axis=1)+np.nanstd(r,axis=1),np.nanmean(r,axis=1)-np.nanstd(r,axis=1),color=plot_colors[ip],alpha=0.15)

                ax.plot(window_list,np.ones(len(window_list))*rc_p2,':', color='gray')
                ax.plot(window_list,np.ones(len(window_list))*rc_m2,':', color='gray')

            plt.subplots_adjust(left=0.2,right=0.9,bottom=0.2)
            ax.set_xlim(0,np.nanmax(window_list)+np.nanmax(window_list)/10.)
            ax.set_ylim(-1,1)
            ax.set_ylabel('r',fontsize=10)
            # ax.set_ylabel(locnames,fontsize=10)

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

#window_arr = 2*2**np.arange(0,8) # For powers of 2
# window_arr = np.arange(1,39)*7 # For weeks, up to Jan 1st
#window_arr = np.arange(1,17)*7
# window_arr = np.arange(1,9)*30 # For months
window_arr = np.arange(1,9)*30 # For months
# window_arr = np.arange(1,3)*30 # For months

#%%
# LOAD FREEZEUP DATES OR FIND FROM TWATER TIME SERIES
# water_name_list = ['Atwater_cleaned_filled','DesBaillets_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
# station_labels = ['Atwater','DesBaillets','Longueuil','Candiac']
# station_type = 'cities'

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


nvars = NAO_vars_all.shape[0]
nyears = NAO_vars_all.shape[1]
nwindows = NAO_vars_all.shape[2]
nend = NAO_vars_all.shape[3]
nwindowtype = NAO_vars_all.shape[4]
nlocs = NAO_vars_all.shape[5]

#%%
if detrend:
    NAO_vars_all_detrended = np.zeros(NAO_vars_all.shape)*np.nan
    for ivar in range(nvars):
        for iw in range(nwindows):
            for iend in range(nend):
                for ip in range(nwindowtype):
                    for iloc in range(nlocs):
                        xvar = NAO_vars_all[ivar,:,iw,iend,ip,iloc]
                        yvar = avg_freezeup_doy

                        NAO_vars_all_detrended[ivar,:,iw,iend,ip,iloc], avg_freezeup_doy_detrended = detrend_ts(xvar,yvar,years,anomaly)
else:
    NAO_vars_all_detrended = NAO_vars_all.copy()
    avg_freezeup_doy_detrended = avg_freezeup_doy.copy()


r_mean = np.zeros((nvars,nwindows,nend,nwindowtype,nlocs))*np.nan
r_std = np.zeros((nvars,nwindows,nend,nwindowtype,nlocs))*np.nan
for ivar in range(nvars):
    for iw in range(nwindows):
        for iend in range(nend):
            for ip in range(nwindowtype):
                for iloc in range(nlocs):
                    xvar = NAO_vars_all_detrended[ivar,:,iw,iend,ip,iloc]
                    yvar = avg_freezeup_doy_detrended

                    r = bootstrap(xvar,yvar,nboot=1)
                    r_mean[ivar,iw,iend,ip,iloc] = np.nanmean(r)
                    r_std[ivar,iw,iend,ip,iloc] = np.nanstd(r)


#%%
varnames = ['NAO']
locnames = ['Montreal']
pc = p_critical

rc_m1, rc_p1 = r_confidence_interval(0,pc,nyears,tailed='one')
rc_m2, rc_p2 = r_confidence_interval(0,pc,nyears,tailed='two')
plot_colors = [plt.get_cmap('tab20b')(0),plt.get_cmap('tab20b')(4),plt.get_cmap('tab20b')(8),plt.get_cmap('tab20b')(12)]

for iend in range(nend):
    enddate_str = enddate_labels[iend]

    for ivar in range(nvars):
        var1 = varnames[ivar]
        nrows = nlocs
        ncols = 1
        fig,ax = plt.subplots(nrows,figsize=(5,(nlocs)*(8/5.)),sharex=True,sharey=True,squeeze=False)
        if (nrows == 1) | (ncols == 1) :
            ax = ax.reshape(-1)
        plt.suptitle(var1)

        for ip in range(nwindowtype):
            # if nlocs > 1:
            for iloc in range(nlocs):
                r_mean_plot = r_mean[ivar,:,iend,ip,iloc]
                r_std_plot = r_std[ivar,:,iend,ip,iloc]


                ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plot_colors[ip])
                ax[iloc].fill_between(window_arr,r_mean_plot+r_std_plot,r_mean_plot-r_std_plot,color=plot_colors[ip],alpha=0.15)

                ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_p2,':', color='gray')
                ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_m2,':', color='gray')

                plt.subplots_adjust(left=0.2,right=0.9,bottom=0.2)
                ax[iloc].set_xlim(0,np.nanmax(window_arr)+np.nanmax(window_arr)/10.)
                ax[iloc].set_ylim(-1,1)
                ax[iloc].set_ylabel('r',fontsize=10)

                if iloc == nlocs-1:
                    if (window_arr[1]-window_arr[0]) == 7:
                        ax[iloc].set_xlabel('Previous X weeks (XW) from '+enddate_str,fontsize=10)
                        labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'W' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                        labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                    if (window_arr[1]-window_arr[0]) == 30:
                        ax[iloc].set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
                        labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                        labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                    ax[iloc].set_xticks(window_arr)
                    ax[iloc].set_xticklabels(labels_list)


#%%
# # MAKE TWATER INTO AN EXPLANATORY VARIABLE
# Twater_varnames = ['Avg. water temp.']
# Twater_vars = np.zeros((len(time),len(Twater_varnames)))*np.nan
# Twater_vars[:,0] = np.nanmean(Twater,axis=1)
# Twater_vars = np.squeeze(Twater_vars)

# if deseasonalize:
#     Nwindow = 31
#     Twater_vars = deasonalize_ts(Nwindow,Twater_vars,'Twater','all_time',time,years)

# Twater_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
# for iend in range(end_dates_arr.shape[1]):
#     Twater_vars_all[:,:,iend,0] = get_window_vars('moving',np.expand_dims(Twater_vars,axis=1),['Avg. Twater'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
#     Twater_vars_all[:,:,iend,1] = get_window_vars('increasing',np.expand_dims(Twater_vars,axis=1),['Avg. Twater'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

# #%%
# for iend in range(end_dates_arr.shape[1]):
#     yvar= Twater_vars_all[:,:,iend,0]
#     enddate_str = enddate_labels[iend]

#     locnames='Montreal'
#     multilocs_rollingcorr_correlation_plot(window_arr,np.expand_dims(np.expand_dims(NAO_vars_all[:,:,iend,:,:],axis=-1),axis=0),yvar,boot_iyears_all[:,iend,:].astype('int'),NAO_varnames,locnames,p_critical,detrend,anomaly,enddate_str)




#%%
ip = 0
iend = 0

NAO_monthly = NAO_vars_all_detrended[0,:,:,iend,ip,0]
# NAO_monthly = NAO_min_vars_all[0,:,:,iend,ip,0]

# plt.figure();plt.plot(years,NAO_monthly[:,7]);plt.title('Avg. NAO in April')
# plt.figure();plt.plot(years,NAO_monthly[:,6]);plt.title('Avg. NAO in May')
# plt.figure();plt.plot(years,NAO_monthly[:,5]);plt.title('Avg. NAO in June')
# plt.figure();plt.plot(years,NAO_monthly[:,4]);plt.title('Avg. NAO in July')
# plt.figure();plt.plot(years,NAO_monthly[:,3]);plt.title('Avg. NAO in August')
# plt.figure();plt.plot(years,NAO_monthly[:,2]);plt.title('Avg. NAO in Sept.')
# plt.figure();plt.plot(years,NAO_monthly[:,1]);plt.title('Avg. NAO in Oct.')
# plt.figure();plt.plot(years,NAO_monthly[:,0]);plt.title('Avg. NAO in Nov.')

x = NAO_monthly[:,2]
y = NAO_monthly[:,0]

fig, ax = plt.subplots()
ax.plot(years,avg_freezeup_doy_detrended,'o-', color= plt.get_cmap('tab10')(0))
ax.set_xlabel('Year')
ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
ax.grid()
ax2=ax.twinx()
ax2.plot(years,y,'o--', color= plt.get_cmap('tab10')(1))
ax2.set_ylabel('Avg. NAO anomaly in Nov.', color= plt.get_cmap('tab10')(1))
lincoeff, Rsqr = linear_fit(y,avg_freezeup_doy_detrended)
r =np.sqrt(Rsqr)
if lincoeff[0]<0: r*=-1
ax.text(2016,21,'r = %3.2f'%(r))

fig, ax = plt.subplots()
ax.plot(years,avg_freezeup_doy_detrended,'o-', color= plt.get_cmap('tab10')(0))
ax.set_xlabel('Year')
ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
ax.grid()
ax2=ax.twinx()
ax2.plot(years,x,'o--', color= plt.get_cmap('tab10')(1))
ax2.set_ylabel('Avg. NAO anomaly in Sept.', color= plt.get_cmap('tab10')(1))
lincoeff, Rsqr = linear_fit(x,avg_freezeup_doy_detrended)
r =np.sqrt(Rsqr)
if lincoeff[0]<0: r*=-1
ax.text(2001,21,'r = %3.2f'%(r))


fig, ax = plt.subplots()
ax.plot(years,avg_freezeup_doy_detrended,'o-', color= plt.get_cmap('tab10')(0))
ax.set_xlabel('Year')
ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
ax.grid()
ax2=ax.twinx()
ax2.plot(years,-x+y,'o--', color= plt.get_cmap('tab10')(1))
ax2.set_ylabel('- NAO Sept. + NAO Nov.', color= plt.get_cmap('tab10')(1))
lincoeff, Rsqr = linear_fit(-x+y,avg_freezeup_doy_detrended)
r =np.sqrt(Rsqr)
if lincoeff[0]<0: r*=-1
ax.text(2016,21,'r = %3.2f'%(r))


#%%
fig, ax = plt.subplots()
ax.plot(x,avg_freezeup_doy_detrended,'o', color= plt.get_cmap('tab10')(0))


#%%

# x = NAO_monthly[:,3]
# fig, ax = plt.subplots()
# ax.plot(years,avg_freezeup_doy_detrended,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years,x,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('Avg. NAO anomaly in Aug.', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,avg_freezeup_doy_detrended)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2016,21,'r = %3.2f'%(r))
# #%%

# x = NAO_monthly[:,3]
# y = NAO_monthly[:,2]
# fig, ax = plt.subplots()
# ax.plot(years,avg_freezeup_doy_detrended,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years,x+y,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('NAO Aug. + NAO Sept.', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x+y,avg_freezeup_doy_detrended)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2016,21,'r = %3.2f'%(r))


# #%%
# x = NAO_monthly[:,3]
# y = NAO_monthly[:,2]
# fig, ax = plt.subplots()
# ax.plot(years,y,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('NAO Sept.', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years,x,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('NAO Aug.', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,y)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2016,1,'r = %3.2f'%(r))
# #%%
# NAO_monthly = NAO_vars_all_detrended[0,:,:,0,ip,0]
# x = NAO_monthly[:,-1]
# fig, ax = plt.subplots()
# ax.plot(years,avg_freezeup_doy_detrended,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years,x,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('Avg. NAO anomaly in April', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,avg_freezeup_doy_detrended)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2016,21,'r = %3.2f'%(r))


# #%%
# NAO_monthly = NAO_vars_all_detrended[0,:,:,0,ip,0]
# x = NAO_monthly[:,3]+NAO_monthly[:,2]
# y = NAO_monthly[:,7]
# fig, ax = plt.subplots()
# ax.plot(years,avg_freezeup_doy_detrended,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years,-x+y,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('- NAO Aug.- NAO Sept. + NAO April', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(-x+y,avg_freezeup_doy_detrended)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2016,21,'r = %3.2f'%(r))

#%%
# plt.figure()
# plt.plot(x,avg_freezeup_doy_detrended,'o')
# plt.xlabel('Avg. NAO anomaly in Sept.')
# plt.ylabel('Freezeup Anomaly')

# plt.figure()
# plt.plot(y,avg_freezeup_doy_detrended,'o')
# plt.xlabel('Avg. NAO anomaly in Nov.')
# plt.ylabel('Freezeup Anomaly')

# plt.figure()
# plt.plot(x,y,'o')
# plt.xlabel('Avg. NAO anomaly in Sept.')
# plt.ylabel('Avg. NAO anomaly in Nov.')

# lincoeff, Rsqr = linear_fit(x,y)
# plt.plot(x,lincoeff[0]*x + lincoeff[1],'-',color='gray')
# plt.text(np.nanmin(x),np.nanmin(y), 'r = %3.2f'%(np.sqrt(Rsqr)))


#%%
# x = NAO_monthly[:,2]
# y = NAO_monthly[:,3]

# plt.figure()
# plt.plot(x,avg_freezeup_doy_detrended,'o')
# plt.xlabel('Avg. NAO anomaly in Sept.')
# plt.ylabel('Freezeup Anomaly')

# plt.figure()
# plt.plot(y,avg_freezeup_doy_detrended,'o')
# plt.xlabel('Avg. NAO anomaly in Aug.')
# plt.ylabel('Freezeup Anomaly')

# plt.figure()
# plt.plot(x,y,'o')
# plt.xlabel('Avg. NAO anomaly in Sept.')
# plt.ylabel('Avg. NAO anomaly in Aug.')

# lincoeff, Rsqr = linear_fit(x,y)

# plt.plot(x,lincoeff[0]*x + lincoeff[1],'-',color='gray')
# plt.text(np.nanmin(x),np.nanmin(y), 'r = %3.2f'%(np.sqrt(Rsqr)))







