#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:08:00 2021

@author: Amelie
"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'
#%%
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path+'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import scipy as sp
from scipy import ndimage

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import find_freezeup_Tw_all_yrs
# ==========================================================================
def running_nanmean(x, N=3, mean_type='centered'):
    xmean = np.ones(x.shape[0])*np.nan
    temp = np.vstack([x[i:-(N-i)] for i in range(N)]) # stacks vertically the strided arrays
    temp = np.nanmean(temp, axis=0)

    if mean_type == 'before':
        xmean[N-1:N-1+temp.shape[0]] = temp

    if mean_type == 'centered':
        xmean[int((N-1)/2):int((N-1)/2)+temp.shape[0]] = temp

    return xmean


def linear_fit(x_in,y_in):
    A = np.vstack([x_in, np.ones(len(x_in))]).T
    b = y_in
    lstsqr_fit = np.linalg.lstsq(A, b)
    coeff = lstsqr_fit[0]
    slope_fit = np.dot(A,coeff)

    SS_res = np.sum((slope_fit-y_in)**2.)
    SS_tot = np.sum((y_in - y_in.mean())**2.)
    R_sqr = 1-(SS_res/SS_tot)

    return coeff, R_sqr


def find_freezeup_Tw(Twater_in, time, years, thresh_T = 2.0, ndays = 7, date_ref = dt.date(1900,1,1)):

    # mask_tmp is True if T is below thresh_T:
    mask_tmp = Twater_in <= thresh_T

    mask_freezeup = mask_tmp.copy()
    mask_freezeup[:] = False

    freezeup_date=np.zeros((len(years),3))*np.nan
    iyr = 0
    flag = 0 # This flag will become 1 if the data time series starts during the freezing season, to indicate that we cannot use the first date of temp. below freezing point as the freezeup date.

    for im in range(1,mask_freezeup.size):

        if (im == 1) | (~mask_tmp[im-1]):

            sum_m = 0
            if ~mask_tmp[im]:
                sum_m = 0
            else:
                # start new group
                sum_m +=1
                istart = im


                if (sum_m >= ndays):
                    # Temperature has been lower than thresh_T
                    # for more than (or equal to) ndays.
                    # Define freezeup date as first date of group

                    date_start = date_ref+dt.timedelta(days=int(time[istart]))
                    doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

                    if doy_start > 300:
                        if (np.where(np.array(years) == date_start.year)[0].size > 0):
                            iyr = np.where(np.array(years) == date_start.year)[0][0]
                    elif doy_start < 60:
                        if (np.where(np.array(years) == date_start.year-1)[0].size > 0):
                            iyr = np.where(np.array(years) == date_start.year-1)[0][0]
                    else:
                        continue

                    if np.isnan(freezeup_date[iyr,0]):

                        if iyr == 0:
                            if (np.sum(np.isnan(Twater_in[istart-ndays-1-30:istart-ndays-1])) < 7) & (flag == 0):
                                freezeup_date[iyr,0] = date_start.year
                                freezeup_date[iyr,1] = date_start.month
                                freezeup_date[iyr,2] = date_start.day
                                mask_freezeup[istart] = True
                            else:
                                flag = 1 # This flag indicates that the data time series has started during the freezing season already, so that we cannot use the first date of temp. below freezing point as the freezeup date.
                                continue
                        else:
                            if np.isnan(freezeup_date[iyr-1,0]):
                                freezeup_date[iyr,0] = date_start.year
                                freezeup_date[iyr,1] = date_start.month
                                freezeup_date[iyr,2] = date_start.day
                                mask_freezeup[istart] = True
                            else:
                                if freezeup_date[iyr-1,0] == date_start.year:
                                    # 2012 01 05
                                    # 2012 01 07 NO
                                    # 2012 12 22 YES
                                    if (freezeup_date[iyr-1,1] < 5) & (date_start.month > 10):
                                        freezeup_date[iyr,0] = date_start.year
                                        freezeup_date[iyr,1] = date_start.month
                                        freezeup_date[iyr,2] = date_start.day
                                        mask_freezeup[istart] = True

                                elif date_start.year == freezeup_date[iyr-1,0]+1:
                                    # 2012 12 22
                                    # 2013 01 14 NO
                                    # 2013 12 24 YES

                                    #2014 01 03 (2013 season)
                                    #2015 01 13 (2014 season)
                                    if (date_start.month > 10):
                                        freezeup_date[iyr,0] = date_start.year
                                        freezeup_date[iyr,1] = date_start.month
                                        freezeup_date[iyr,2] = date_start.day
                                        mask_freezeup[istart] = True
                                    elif (date_start.month < 5) & (freezeup_date[iyr-1,1] < 5) :
                                        freezeup_date[iyr,0] = date_start.year
                                        freezeup_date[iyr,1] = date_start.month
                                        freezeup_date[iyr,2] = date_start.day
                                        mask_freezeup[istart] = True

                                elif date_start.year == freezeup_date[iyr-1,0]+2:
                                    if (date_start.month < 5):
                                        freezeup_date[iyr,0] = date_start.year
                                        freezeup_date[iyr,1] = date_start.month
                                        freezeup_date[iyr,2] = date_start.day
                                        mask_freezeup[istart] = True

                                else:
                                    print(iyr)
                                    print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)
                                    # if (date_start.year > freezeup_date[iyr-1,0]+2):
                                    #     freezeup_date[iyr,0] = date_start.year
                                    #     freezeup_date[iyr,1] = date_start.month
                                    #     freezeup_date[iyr,2] = date_start.day
                                    #     mask_freezeup[istart] = True
                                    # else:
                                    #     print(iyr)
                                    #     print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)


        else:
            if mask_tmp[im]:
                sum_m += 1

                if (sum_m >= ndays):
                    # Temperature has been lower than thresh_T
                    # for more than (or equal to) ndays.
                    # Define freezeup date as first date of group

                    date_start = date_ref+dt.timedelta(days=int(time[istart]))
                    doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

                    if doy_start > 300:
                        if (np.where(np.array(years) == date_start.year)[0].size > 0):
                            iyr = np.where(np.array(years) == date_start.year)[0][0]
                    elif doy_start < 60:
                        if (np.where(np.array(years) == date_start.year-1)[0].size > 0):
                            iyr = np.where(np.array(years) == date_start.year-1)[0][0]
                    else:
                        continue

                    if np.isnan(freezeup_date[iyr,0]):

                        if iyr == 0:
                            if (np.sum(np.isnan(Twater_in[istart-ndays-1-30:istart-ndays-1])) < 7) & (flag == 0):
                                freezeup_date[iyr,0] = date_start.year
                                freezeup_date[iyr,1] = date_start.month
                                freezeup_date[iyr,2] = date_start.day
                                mask_freezeup[istart] = True
                            else:
                                flag = 1 # This flag indicates that the data time series has started during the freezing season already, so that we cannot use the first date of temp. below freezing point as the freezeup date.
                                continue
                        else:
                            if np.isnan(freezeup_date[iyr-1,0]):
                                freezeup_date[iyr,0] = date_start.year
                                freezeup_date[iyr,1] = date_start.month
                                freezeup_date[iyr,2] = date_start.day
                                mask_freezeup[istart] = True
                            else:
                                if freezeup_date[iyr-1,0] == date_start.year:
                                    # 2012 01 05
                                    # 2012 01 07 NO
                                    # 2012 12 22 YES
                                    if (freezeup_date[iyr-1,1] < 5) & (date_start.month > 10):
                                        freezeup_date[iyr,0] = date_start.year
                                        freezeup_date[iyr,1] = date_start.month
                                        freezeup_date[iyr,2] = date_start.day
                                        mask_freezeup[istart] = True

                                elif date_start.year == freezeup_date[iyr-1,0]+1:
                                    # 2012 12 22
                                    # 2013 01 14 NO
                                    # 2013 12 24 YES

                                    #2014 01 03 (2013 season)
                                    #2015 01 13 (2014 season)
                                    if (date_start.month > 10):
                                        freezeup_date[iyr,0] = date_start.year
                                        freezeup_date[iyr,1] = date_start.month
                                        freezeup_date[iyr,2] = date_start.day
                                        mask_freezeup[istart] = True
                                    elif (date_start.month < 5) & (freezeup_date[iyr-1,1] < 5) :
                                        freezeup_date[iyr,0] = date_start.year
                                        freezeup_date[iyr,1] = date_start.month
                                        freezeup_date[iyr,2] = date_start.day
                                        mask_freezeup[istart] = True

                                elif date_start.year == freezeup_date[iyr-1,0]+2:
                                    if (date_start.month < 5):
                                        freezeup_date[iyr,0] = date_start.year
                                        freezeup_date[iyr,1] = date_start.month
                                        freezeup_date[iyr,2] = date_start.day
                                        mask_freezeup[istart] = True

                                else:
                                    print(iyr)
                                    print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)
                                    # if (date_start.year > freezeup_date[iyr-1,0]+2):
                                    #     freezeup_date[iyr,0] = date_start.year
                                    #     freezeup_date[iyr,1] = date_start.month
                                    #     freezeup_date[iyr,2] = date_start.day
                                    #     mask_freezeup[istart] = True
                                    # else:
                                    #     print(iyr)
                                    #     print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)


    Twater_out = Twater_in.copy()
    Twater_out[~mask_freezeup] = np.nan

    return freezeup_date, Twater_out, mask_freezeup

###
def find_freezeup_Tw2(def_opt,Twater_in,dTdt_in,d2Tdt2_in,time,year,thresh_T = 2.0,thresh_dTdt = 0.2,thresh_d2Tdt2 = 0.2,ndays = 7,date_ref = dt.date(1900,1,1)):

    def record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref):
        # Temperature has been lower than thresh_T
        # for more than (or equal to) ndays.
        # Define freezeup date as first date of group

        date_start = date_ref+dt.timedelta(days=int(time[istart]))
        doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

        if ((date_start.year > 1992) | ((date_start.year == 1992) & (date_start.month > 10)) ):
            if ( (date_start.year == year) & (doy_start > 319) ) | ((date_start.year == year+1) & (doy_start < 46)):
                    freezeup_date[0] = date_start.year
                    freezeup_date[1] = date_start.month
                    freezeup_date[2] = date_start.day
                    freezeup_Tw = Twater_in[istart]
                    mask_freezeup[istart] = True
            else:
                freezeup_date[0] = np.nan
                freezeup_date[1] = np.nan
                freezeup_date[2] = np.nan
                freezeup_Tw = np.nan
                mask_freezeup[istart] = False
        else:
            freezeup_date[0] = np.nan
            freezeup_date[1] = np.nan
            freezeup_date[2] = np.nan
            freezeup_Tw = np.nan
            mask_freezeup[istart] = False

        return freezeup_date, freezeup_Tw, mask_freezeup


    if def_opt == 1: # T is below thresh_T:
        mask_tmp = Twater_in <= thresh_T

    if (def_opt == 2):# T is below thresh_T and dTdt is below thresh_dTdt:
        zero_dTdt_mask = np.abs(dTdt_in) <= thresh_dTdt
        zero_T_mask = Twater_in <= thresh_T
        # zero_d2Tdt2_mask = np.abs(d2Tdt2_in) <= thresh_d2Tdt2
        # mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
        mask_tmp = zero_T_mask & zero_dTdt_mask

    if (def_opt == 3): # T is below thresh_T and dTdt is below thresh_dTdt and d2Tdt2 is below thresh_d2Tdt2:
        zero_dTdt_mask = np.abs(dTdt_in) <= thresh_dTdt
        zero_T_mask = Twater_in <= thresh_T
        # zero_d2Tdt2_mask = np.abs(d2Tdt2_in) <= thresh_d2Tdt2
        # mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
        mask_tmp = zero_T_mask & zero_dTdt_mask

    mask_freezeup = mask_tmp.copy()
    mask_freezeup[:] = False

    freezeup_Tw = np.nan
    freezeup_date=np.zeros((3))*np.nan

    for im in range(mask_freezeup.size):

        if (im == 0):
            sum_m = 0
            istart = -1 # This ensures that a freeze-up is not detected if the time series started already below the freezing temp.
        else:
            if (np.sum(mask_freezeup) == 0): # Only continue while no prior freeze-up was detected for the sequence
                if (~mask_tmp[im-1]):
                    sum_m = 0
                    if ~mask_tmp[im]:
                        sum_m = 0
                    else:
                        # start new group
                        sum_m +=1
                        istart = im
                        # Below will only occur if ndays is set to 1, e.g. first day of freezing temp.
                        if (sum_m >= ndays):
                            freezeup_date,freezeup_Tw,mask_freezeup = record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref)
                else:
                    if (mask_tmp[im]) & (istart > 0):
                        sum_m += 1
                        if (sum_m >= ndays):
                            freezeup_date,freezeup_Tw,mask_freezeup = record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref)

    Twater_out = Twater_in.copy()
    Twater_out[~mask_freezeup] = np.nan

    return freezeup_date, freezeup_Tw, Twater_out, mask_freezeup


def find_breakup_Tw(Twater_in, time, years, thresh_T = 2.0, ndays = 7, mask_freezeup = None, date_ref = dt.date(1900,1,1)):

    # mask_tmp is True if T is above thresh_T:
    mask_tmp = Twater_in >= thresh_T

    mask_breakup = mask_tmp.copy()
    mask_breakup[:] = False

    breakup_date=np.zeros((len(years),3))*np.nan
    iyr = 0

    if mask_freezeup is not None:

        print('NEED TO WRITE THIS')

    else:

        for iyr in range(len(years)):

            if len(np.where(time == (dt.date(years[iyr],1,1)-date_ref).days)) > 0:
                iJan1 = np.where(time == (dt.date(years[iyr],1,1)-date_ref).days)[0][0]

                for it,im in enumerate(np.arange(iJan1,iJan1+150)):

                    if (it == 0) | (~mask_tmp[im-1]):

                        sum_m = 0
                        if ~mask_tmp[im]:
                            sum_m = 0
                        else:
                            # start new group
                            sum_m +=1
                            istart = im

                    else:
                        if mask_tmp[im]:
                            sum_m += 1

                            if (sum_m >= ndays):
                                # Temperature has been lower than thresh_T
                                # for more than (or equal to) ndays.
                                # Define freezeup date as first date of group

                                date_start = date_ref+dt.timedelta(days=int(time[istart]))
                                doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

                                if np.isnan(breakup_date[iyr,0]):
                                    if date_start.year == years[iyr]:
                                        if (doy_start > 31) & (doy_start < 110):
                                            if np.sum(np.isnan(Twater_in[istart-7:istart])) < 7:
                                                breakup_date[iyr,0] = date_start.year
                                                breakup_date[iyr,1] = date_start.month
                                                breakup_date[iyr,2] = date_start.day
                                                mask_breakup[istart] = True
                                        else:
                                            continue
                                    else:
                                        print('PROBLEM!!!!')


    Twater_out = Twater_in.copy()
    Twater_out[~mask_breakup] = np.nan

    return breakup_date, Twater_out, mask_breakup


#%%
years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

fp = local_path+'slice/data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)



# freezeup_loc_list = ['Lasalle','Candiac','LaPrairie','Longueuil','StLambert','MontrealOldPort','MontrealPort']
# freezeup_type = 'charts'

# chart_fd = np.zeros((len(freezeup_loc_list),len(years),3))*np.nan
# chart_fd_doy = np.zeros((len(freezeup_loc_list),len(years),2))*np.nan

# for iloc,loc in enumerate(freezeup_loc_list):

#     ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type+'/freezeup_'+freezeup_type+'_'+loc+'.npz',allow_pickle='TRUE')
#     freezeup = ice_data['freezeup_ci'][:,0]
#     freezeup_dt = ice_data['freezeup_ci'][:,1]


#     for iyr,year in enumerate(years):

#         date=(dt.date(year,11,1)-date_ref).days
#         i0 = np.where(time==date)[0][0] - 30
#         i1 = i0+120

#         ci_select = freezeup[i0:i1].copy()
#         dt_select = freezeup_dt[i0:i1].copy()

#         if np.sum(~np.isnan(ci_select)) > 0:
#             i_fd = np.where(~np.isnan(ci_select))[0][0]
#             date_chart = date_ref+dt.timedelta(days=int(ci_select[i_fd]))
#             fd_dt = dt_select[i_fd]

#             fd_yy = int(date_chart.year)
#             fd_mm = int(date_chart.month)
#             fd_dd = int(date_chart.day)

#             chart_fd[iloc,iyr,0] = date_chart.year
#             chart_fd[iloc,iyr,1] = date_chart.month
#             chart_fd[iloc,iyr,2] = date_chart.day

#             fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
#             if fd_doy < 60: fd_doy += 365
#             chart_fd_doy[iloc,iyr,0] = fd_doy
#             chart_fd_doy[iloc,iyr,1] = fd_dt



#%%
#PLOT FREEZEUP DATE VS YEARS FOR ALL STATIONS IN ONE FIGURE
show_plot = False

if show_plot:
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5))
    freezeup_plot_list = ['Lasalle','Candiac','LaPrairie','Longueuil','StLambert','MontrealOldPort','MontrealPort']
    # freezeup_plot_list = ['Longueuil','StLambert','MontrealPort']
    # freezeup_plot_list = ['Longueuil','MontrealPort']
    freezeup_plot_list = ['Longueuil','LaPrairie']

    for loc in freezeup_plot_list:
        if loc == 'Lasalle': iloc =0
        if loc == 'Candiac': iloc =1
        if loc == 'LaPrairie': iloc =2
        if loc == 'Longueuil': iloc =3
        if loc == 'StLambert': iloc =4
        if loc == 'MontrealOldPort': iloc =5
        if loc == 'MontrealPort': iloc =6

        lower_error = []
        upper_error = []
        for iyr,year in enumerate(years):
            if ~np.isnan(chart_fd_doy[iloc,iyr,1]):
                lower_error.append(chart_fd_doy[iloc,iyr,1]-1)
                upper_error.append(0)
            else:
                lower_error.append(0)
                upper_error.append(0)


        error = [lower_error, upper_error]
        ax.errorbar(years, chart_fd_doy[iloc,:,0], yerr=error, fmt='*', alpha=0.5, label = loc,color=plt.get_cmap('tab20')(iloc+4))
    ax.grid(axis='y')
    ax.legend()


#%%
# PLOT THE DISTRIBUTION OF FREEZEUP DATES FOR ALL STATIONS
show_plot = False

if show_plot:
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5))
    freezeup_plot_list = ['Lasalle','Candiac','LaPrairie','Longueuil','StLambert','MontrealOldPort','MontrealPort']

    for loc in freezeup_plot_list:
        print(loc)
        # fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5))
        if loc == 'Lasalle': iloc =0
        if loc == 'Candiac': iloc =1
        if loc == 'LaPrairie': iloc =2
        if loc == 'Longueuil': iloc =3
        if loc == 'StLambert': iloc =4
        if loc == 'MontrealOldPort': iloc =5
        if loc == 'MontrealPort': iloc =6

        ax.boxplot(chart_fd_doy[iloc,:,0][~np.isnan(chart_fd_doy[iloc,:,0])], positions = [iloc*0.35],whis=[5, 95],showfliers=True,labels = [loc])

    ax.grid(True)
    ax.set_ylabel('Freezeup DOY')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.24)

#%%
# PLOT THE DIFFERENCE IN FREEZEUP DATE FOR A GIVEN STATION, COMPARED TO ALL
# THE OTHER STATIONS

show_plot = False

if show_plot:

    freezeup_plot_list = ['Lasalle','Candiac','LaPrairie','Longueuil','StLambert','MontrealOldPort','MontrealPort']

    for iloc,loc in enumerate(freezeup_plot_list):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5), sharex=True)

        plt.suptitle(loc)

        fd_main = chart_fd_doy[iloc,:,0]

        for jloc in range(len(freezeup_plot_list)):
            if jloc != iloc:
                fd_diff = fd_main - chart_fd_doy[jloc,:,0]

                ax.boxplot(fd_diff[~np.isnan(fd_diff)], positions = [jloc*0.25],whis=[5, 95],showfliers=True,labels = [freezeup_plot_list[jloc]+'\n(%2i'%np.sum(~np.isnan(fd_diff))+')'],showmeans=True)

                plt.xticks(rotation=45)
                plt.subplots_adjust(bottom=0.24)
                ax.set_ylabel('Diff. (days) in freezeup\n ('+loc+' - other stations)')
                ax.grid(True)
                ax.text(-0.25, -15, 'Earlier',color='blue',rotation=90)
                ax.text(-0.25, 4, 'Later',color='red',rotation=90)
                ax.set_ylim(-44,44)
                ax.set_xlim(-0.35,0.25*7)


#%%
# SCATTER PLOT OF FREEZEUP DATE FOR ONE STATION AGAINST ALL OTHERS
# ONLY POINTS WITH ESTIMATE OF DT

show_plot = False

if show_plot:
    freezeup_plot_list = ['Lasalle','Candiac','LaPrairie','Longueuil','StLambert','MontrealOldPort','MontrealPort']

    for iloc,loc in enumerate(freezeup_plot_list):
        fd_main = chart_fd_doy[iloc,:,0][~np.isnan(chart_fd_doy[iloc,:,1])]
        dt_main = chart_fd_doy[iloc,:,1][~np.isnan(chart_fd_doy[iloc,:,1])]

        fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(8,5), sharex=True, sharey=True)
        iplot = 0

        for jloc in range(len(freezeup_plot_list)):
            if jloc != iloc:
                if iplot == 0: ip = 0; jp=0
                if iplot == 1: ip = 0; jp=1
                if iplot == 2: ip = 0; jp=2
                if iplot == 3: ip = 1; jp=0
                if iplot == 4: ip = 1; jp=1
                if iplot == 5: ip = 1; jp=2

                lower_xerror = []
                upper_xerror = []
                lower_yerror = []
                upper_yerror = []
                x = []
                y = []

                for iyr,year in enumerate(years):

                    fd_main = chart_fd_doy[iloc,iyr,0]
                    dt_main = chart_fd_doy[iloc,iyr,1]

                    fd_comp = chart_fd_doy[jloc,iyr,0]
                    dt_comp = chart_fd_doy[jloc,iyr,1]

                    if ~np.isnan(fd_main) & ~np.isnan(fd_comp):
                        if ~np.isnan(dt_main) &  ~np.isnan(dt_comp):
                            lower_xerror.append(dt_main-1)
                            lower_yerror.append(dt_comp-1)
                            x.append(fd_main)
                            y.append(fd_comp)
                            upper_xerror.append(0)
                            upper_yerror.append(0)
                        else:
                            lower_yerror.append(0)
                            lower_xerror.append(0)
                            x.append(np.nan)
                            y.append(np.nan)
                            upper_xerror.append(0)
                            upper_yerror.append(0)

                xerror = [lower_xerror, upper_xerror]
                yerror = [lower_yerror, upper_yerror]
                ax[ip,jp].errorbar(x,y, xerr=xerror, yerr=yerror, fmt='o')

                ax[ip,jp].plot(np.arange(335,395),np.arange(335,395) ,'-')
                ax[ip,jp].set_ylabel(freezeup_plot_list[jloc])
                ax[ip,jp].set_xlabel(loc)

                ax[ip,jp].grid(True)
                ax[ip,jp].text(380, 335, 'Later',color='red',rotation=45)
                ax[ip,jp].text(335, 380, 'Earlier',color='blue',rotation=45)
                ax[ip,jp].set_ylim(330,400)
                ax[ip,jp].set_xlim(330,400)

                iplot += 1


#%%
# SCATTER PLOT OF FREEZEUP DATE FOR ONE STATION AGAINST ALL OTHERS

show_plot = False

if show_plot:
    freezeup_plot_list = ['Lasalle','Candiac','LaPrairie','Longueuil','StLambert','MontrealOldPort','MontrealPort']

    for iloc,loc in enumerate(freezeup_plot_list):
        fd_main = chart_fd_doy[iloc,:,0]

        fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(8,5), sharex=True, sharey=True)
        iplot = 0

        for jloc in range(len(freezeup_plot_list)):
            if jloc != iloc:
                if iplot == 0: ip = 0; jp=0
                if iplot == 1: ip = 0; jp=1
                if iplot == 2: ip = 0; jp=2
                if iplot == 3: ip = 1; jp=0
                if iplot == 4: ip = 1; jp=1
                if iplot == 5: ip = 1; jp=2

                lower_xerror = []
                upper_xerror = []
                for iyr,year in enumerate(years):
                    if ~np.isnan(chart_fd_doy[iloc,iyr,1]):
                        lower_xerror.append(chart_fd_doy[iloc,iyr,1]-1)
                        upper_xerror.append(0)
                    else:
                        lower_xerror.append(0)
                        upper_xerror.append(0)

                lower_yerror = []
                upper_yerror = []
                for iyr,year in enumerate(years):
                    if ~np.isnan(chart_fd_doy[jloc,iyr,1]):
                        lower_yerror.append(chart_fd_doy[jloc,iyr,1]-1)
                        upper_yerror.append(0)
                    else:
                        lower_yerror.append(0)
                        upper_yerror.append(0)

                xerror = [lower_xerror, upper_xerror]
                yerror = [lower_yerror, upper_yerror]
                ax[ip,jp].errorbar(fd_main,chart_fd_doy[jloc,:,0], xerr=xerror, yerr=yerror, fmt='o')

                ax[ip,jp].plot(fd_main,fd_main ,'-')
                ax[ip,jp].set_ylabel(freezeup_plot_list[jloc])
                ax[ip,jp].set_xlabel(loc)

                ax[ip,jp].grid(True)
                ax[ip,jp].text(380, 335, 'Later',color='red',rotation=45)
                ax[ip,jp].text(335, 380, 'Earlier',color='blue',rotation=45)
                ax[ip,jp].set_ylim(330,400)
                ax[ip,jp].set_xlim(330,400)

                iplot += 1


#%%
# PLOT CHART DATES OF ST-LAMBERT VS DATES FROM SLMSC
show_plot = False
if show_plot:

    chart_loc = 'StLambert'
    ichart = 4
    # chart_loc = 'Longueuil'
    # ichart = 3
    # chart_loc = 'MontrealPort'
    # ichart = 6
    fd_chart = chart_fd_doy[ichart,:,0]

    slsmc_loc = 'SouthShoreCanal'

    ice_data = np.load(fp+'freezeup_dates_SLSMC/freezeup_SLSMC_'+slsmc_loc+'.npz',allow_pickle='TRUE')
    fd_fi_slsmc = ice_data['freezeup_fi']
    fd_si_slsmc = ice_data['freezeup_si']

    slsmc_fi_doy = np.zeros((len(years)))*np.nan
    slsmc_si_doy = np.zeros((len(years)))*np.nan
    for iyr,year in enumerate(years):

        date=(dt.date(year,11,1)-date_ref).days
        i0 = np.where(time==date)[0][0] - 30
        i1 = i0+120

        fi_select = fd_fi_slsmc[i0:i1].copy()
        si_select = fd_si_slsmc[i0:i1].copy()

        if np.sum(~np.isnan(fi_select)) > 0:
            i_fd = np.where(~np.isnan(fi_select))[0][0]
            date_chart = date_ref+dt.timedelta(days=int(fi_select[i_fd]))

            fd_yy = int(date_chart.year)
            fd_mm = int(date_chart.month)
            fd_dd = int(date_chart.day)

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365
            slsmc_fi_doy[iyr] = fd_doy

        if np.sum(~np.isnan(si_select)) > 0:
            i_fd = np.where(~np.isnan(si_select))[0][0]
            date_chart = date_ref+dt.timedelta(days=int(si_select[i_fd]))

            fd_yy = int(date_chart.year)
            fd_mm = int(date_chart.month)
            fd_dd = int(date_chart.day)

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365
            slsmc_si_doy[iyr] = fd_doy


    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8,5), sharex=True, sharey=True)

    ax[0].plot(fd_chart,slsmc_fi_doy,'o')
    ax[0].set_xlabel('Chart DOY')
    ax[0].set_ylabel('SLSMC First ice DOY')
    ax[0].plot(np.arange(325,495),np.arange(325,495),'-')
    ax[0].set_xlim(330,400)
    ax[0].set_ylim(330,400)

    ax[1].plot(fd_chart,slsmc_si_doy,'o')
    ax[1].set_xlabel('Chart DOY')
    ax[1].set_ylabel('SLSMC Stable ice DOY')
    ax[1].plot(np.arange(325,495),np.arange(325,495),'-')
    ax[1].set_xlim(330,400)
    ax[1].set_ylim(330,400)


#%%
# FIND FREEZEUP FROM TWATER AND CHARTS AT THE SAME LOCATIONS

# water_name_list = ['Atwater_cleaned_filled','DesBaillets_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
# station_type = 'cities'

# freezeup_loc_list = ['Lasalle','Lasalle','Longueuil','Candiac']
# freezeup_type = 'charts'

water_name_list = ['Longueuil_updated_cleaned_filled']
station_type = 'cities'

freezeup_loc_list = ['Longueuil']
freezeup_type = 'charts'

chart_fd = np.zeros((len(freezeup_loc_list),len(years),3))*np.nan
chart_fd_doy = np.zeros((len(freezeup_loc_list),len(years),2))*np.nan

# for iloc,loc in enumerate(freezeup_loc_list):

#     ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type+'/freezeup_'+freezeup_type+'_'+loc+'.npz',allow_pickle='TRUE')
#     freezeup = ice_data['freezeup_ci'][:,0]
#     freezeup_dt = ice_data['freezeup_ci'][:,1]

#     for iyr,year in enumerate(years):

#         date=(dt.date(year,11,1)-date_ref).days
#         i0 = np.where(time==date)[0][0] - 30
#         i1 = i0+120

#         ci_select = freezeup[i0:i1].copy()
#         dt_select = freezeup_dt[i0:i1].copy()

#         if np.sum(~np.isnan(ci_select)) > 0:
#             i_fd = np.where(~np.isnan(ci_select))[0][0]
#             date_chart = date_ref+dt.timedelta(days=int(ci_select[i_fd]))
#             fd_dt = dt_select[i_fd]

#             fd_yy = int(date_chart.year)
#             fd_mm = int(date_chart.month)
#             fd_dd = int(date_chart.day)

#             chart_fd[iloc,iyr,0] = date_chart.year
#             chart_fd[iloc,iyr,1] = date_chart.month
#             chart_fd[iloc,iyr,2] = date_chart.day

#             fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
#             if fd_doy < 60: fd_doy += 365
#             chart_fd_doy[iloc,iyr,0] = fd_doy
#             chart_fd_doy[iloc,iyr,1] = fd_dt


# THEN APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST, BEFORE
# FINDING FREEZEUP DATES FROM WATER TEMP.
Twater = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan

for iloc,loc in enumerate(water_name_list):
    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][:,1]

    Twater[:,iloc] = Twater_tmp
    if loc == 'Candiac_cleaned_filled':
        Twater[:,iloc] = Twater_tmp-0.8
    if (loc == 'Atwater_cleaned_filled'):
        Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7
    if (loc == 'Longueuil_updated_cleaned_filled'):
        Twater[14329:,iloc] = Twater_tmp[14329:]-0.78

    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

#%%
# FIND FREEZEUP DATES FROM TWATER TIME SERIES, FROM THRESHOLD ON Tw
freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
breakup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan

# station_labels = ['Atwater','DesBaillets','Longueuil','Candiac']
station_labels = ['Longueuil']

# Original definition:
smooth_T =False; N_smooth = 10; mean_type='centered'
round_T = False; round_type= 'half_unit'
Gauss_filter = False
T_thresh = 0.5
nd = 7

# # New definition
# smooth_T =False; N_smooth = 10; mean_type='centered'
# round_T = True; round_type= 'half_unit'
# Gauss_filter = False
# T_thresh = 0.5
# nd = 1

# Definition equivalent to new def, but without rounding
def_opt = 1
smooth_T =False; N_smooth = 10; mean_type='centered'
round_T = False; round_type= 'unit'
Gauss_filter = False
T_thresh = 0.75
dTdt_thresh = 0.25
d2Tdt2_thresh = 0.25
nd = 1

# def_opt = 1
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = False
# T_thresh = 0.75
# dTdt_thresh = 0.25
# d2Tdt2_thresh = 0.25
# nd = 1

# def_opt = 1
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = False
# T_thresh = 1.0
# dTdt_thresh = 0.25
# d2Tdt2_thresh = 0.25
# nd = 1

# def_opt = 3
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = True
# sig_dog = 3.5
# T_thresh = 3.
# dTdt_thresh = 0.15
# d2Tdt2_thresh = 0.15
# nd = 30

# def_opt = 3
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = True
# sig_dog = 2.5
# T_thresh = 3.
# dTdt_thresh = 0.15
# d2Tdt2_thresh = 0.15
# nd = 30

# def_opt = 3
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = True
# sig_dog = 3.5
# T_thresh = 3.
# dTdt_thresh = 0.15
# d2Tdt2_thresh = 0.15
# nd = 7

# def_opt = 3
# smooth_T =False; N_smooth = 3; mean_type='centered'
# round_T = False; round_type= 'half_unit'
# Gauss_filter = True
# sig_dog =30
# T_thresh = 1.0
# dTdt_thresh = 0.2
# d2Tdt2_thresh = 0.2
# nd = 7

fig_tw,ax_tw = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))

for iloc,loc in enumerate(water_name_list):
    Twater_tmp = Twater[:,iloc].copy()

    if round_T:
        if round_type == 'unit':
            Twater_tmp = np.round(Twater_tmp.copy())
        if round_type == 'half_unit':
            Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.

    if smooth_T:
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

    # if Gauss_filter:
    #    Twater_tmp = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

    if Gauss_filter:
        Twater_DoG1 = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
        Twater_DoG2 = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)
        plt.figure();plt.plot(Twater_tmp)
        plt.figure();plt.plot(Twater_DoG1)
    # fd, T_freezeup, mask_freeze = find_freezeup_Tw(Twater_tmp,time,years,thresh_T = T_thresh, ndays = nd)
    # freezeup_dates[:,:,iloc] = fd

    bd, T_breakup, mask_break = find_breakup_Tw(Twater_tmp,time,years,thresh_T = T_thresh, ndays = nd)
    breakup_dates[:,:,iloc] = bd

    if def_opt == 3:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_DoG1,Twater_DoG2,time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_dates[:,:,iloc] = fd
        # freezeup_temp[:,iloc] = ftw
    else:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_dates[:,:,iloc] = fd
        # freezeup_temp[:,iloc] = ftw

    ax_tw.plot(Twater_tmp,color=plt.get_cmap('tab20')(iloc*2+1))
    ax_tw.plot(T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2))
    ax_tw.plot(T_breakup, 'o',color=plt.get_cmap('tab20')(iloc*2))

    for iyr in range(chart_fd.shape[1]):
        if ~np.isnan(chart_fd[iloc,iyr,0]):
            d = (dt.date(int(chart_fd[iloc,iyr,0]),int(chart_fd[iloc,iyr,1]),int(chart_fd[iloc,iyr,2]))-date_ref).days
            fu_i = np.where(time == d)[0][0]
            ax_tw.plot(fu_i, Twater_tmp[fu_i], '*', color='black')


    mean_winter_temp = np.zeros(len(years))*np.nan
    for iyr,year in enumerate(years[:-1]):
        if ~np.isnan(fd[iyr,0]) & ~np.isnan(bd[iyr+1,0]):

            i_fr = np.where(time == ( dt.date(int(fd[iyr,0]),int(fd[iyr,1]),int(fd[iyr,2]))-date_ref).days)[0][0]
            i_br = np.where(time == ( dt.date(int(bd[iyr+1,0]),int(bd[iyr+1,1]),int(bd[iyr+1,2]))-date_ref).days)[0][0]
            Twater_winter = Twater[i_fr:i_br,iloc]
            mean_winter_temp[iyr] = np.nanmean(Twater_winter)


freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
breakup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
for iloc,loc in enumerate(water_name_list):
    for iyr,year in enumerate(years):
        if ~np.isnan(freezeup_dates[iyr,0,iloc]):
            fd_yy = int(freezeup_dates[iyr,0,iloc])
            fd_mm = int(freezeup_dates[iyr,1,iloc])
            fd_dd = int(freezeup_dates[iyr,2,iloc])

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365

            freezeup_doy[iyr,iloc]=fd_doy

        if ~np.isnan(breakup_dates[iyr,0,iloc]):
            bd_yy = int(breakup_dates[iyr,0,iloc])
            bd_mm = int(breakup_dates[iyr,1,iloc])
            bd_dd = int(breakup_dates[iyr,2,iloc])

            bd_doy = (dt.date(bd_yy,bd_mm,bd_dd)-dt.date(bd_yy,1,1)).days + 1

            breakup_doy[iyr,iloc]=bd_doy



fig_fddoy,ax_fddoy = plt.subplots(nrows=1,ncols=2,figsize=(12,3.5))
for iloc,loc in enumerate(water_name_list):
    ax_fddoy[0].plot(years,freezeup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2+1), label=station_labels[iloc],alpha=0.75)
    # y_plot = freezeup_doy[:,iloc].copy()
    # lincoeff, Rsqr = linear_fit(years[~np.isnan(y_plot)],y_plot[~np.isnan(y_plot)])
    # ax_fddoy.plot(years,lincoeff[0]*years+lincoeff[1],'-',color=plt.get_cmap('tab20')(iloc*2+1) )
    # ax_fddoy.text(2010,413-6*iloc,'%3.2f'%lincoeff[0]+'x '+'%3.2f'%lincoeff[1] +' (R$^{2}$: %3.2f'%(Rsqr)+')',color=plt.get_cmap('tab20')(iloc*2+1) )

    fd_chart_plot = chart_fd_doy[iloc,:,0][~np.isnan(freezeup_doy[:,iloc])]
    dt_chart_plot = chart_fd_doy[iloc,:,1][~np.isnan(freezeup_doy[:,iloc])]
    years_plot = years[~np.isnan(freezeup_doy[:,iloc])]

    lower_yerror = []
    upper_yerror = []
    for iyr,year in enumerate(years_plot):
        if ~np.isnan(dt_chart_plot[iyr]):
            lower_yerror.append(dt_chart_plot[iyr]-1)
            upper_yerror.append(0)
        else:
            lower_yerror.append(0)
            upper_yerror.append(0)

    yerror = [lower_yerror, upper_yerror]
    ax_fddoy[0].errorbar(years_plot,fd_chart_plot, yerr=yerror, fmt='*',color=plt.get_cmap('tab20')(iloc*2),alpha=0.5)

ax_fddoy[0].legend(bbox_to_anchor=(0.0, 0.64, 0.3, 0.3),fontsize=8)
ax_fddoy[0].set_ylim(300,430)
ax_fddoy[0].set_xlabel('Years')
ax_fddoy[0].set_ylabel('Freezeup DOY')


#ADD FREEZEUP OBSERVED FROM SLSMC
# freezeup_loc_slsmc = ['SouthShoreCanal']
# freezeup_type2 = 'SLSMC'
# slsmc_fi = np.zeros((len(freezeup_loc_slsmc),len(years),3))*np.nan
# slsmc_fi_doy = np.zeros((len(freezeup_loc_slsmc),len(years),2))*np.nan
# slsmc_si = np.zeros((len(freezeup_loc_slsmc),len(years),3))*np.nan
# slsmc_si_doy = np.zeros((len(freezeup_loc_slsmc),len(years),2))*np.nan

# for iloc,loc in enumerate(freezeup_loc_slsmc):

#     ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type2+'/freezeup_'+freezeup_type2+'_'+loc+'.npz',allow_pickle='TRUE')
#     freezeup_fi = ice_data['freezeup_fi'][:,0]
#     freezeup_si = ice_data['freezeup_si'][:,0]

#     for iyr,year in enumerate(years):

#         date=(dt.date(year,11,1)-date_ref).days
#         i0 = np.where(time==date)[0][0] - 30
#         i1 = i0+120

#         fi_select = freezeup_fi[i0:i1].copy()
#         si_select = freezeup_si[i0:i1].copy()

#         if np.sum(~np.isnan(fi_select)) > 0:
#             i_fd = np.where(~np.isnan(fi_select))[0][0]
#             date_chart = date_ref+dt.timedelta(days=int(fi_select[i_fd]))

#             fd_yy = int(date_chart.year)
#             fd_mm = int(date_chart.month)
#             fd_dd = int(date_chart.day)

#             slsmc_fi[iloc,iyr,0] = date_chart.year
#             slsmc_fi[iloc,iyr,1] = date_chart.month
#             slsmc_fi[iloc,iyr,2] = date_chart.day

#             fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
#             if fd_doy < 60: fd_doy += 365
#             slsmc_fi_doy[iloc,iyr,0] = fd_doy

#         if np.sum(~np.isnan(si_select)) > 0:
#             i_fd = np.where(~np.isnan(si_select))[0][0]
#             date_chart = date_ref+dt.timedelta(days=int(si_select[i_fd]))

#             fd_yy = int(date_chart.year)
#             fd_mm = int(date_chart.month)
#             fd_dd = int(date_chart.day)

#             slsmc_si[iloc,iyr,0] = date_chart.year
#             slsmc_si[iloc,iyr,1] = date_chart.month
#             slsmc_si[iloc,iyr,2] = date_chart.day

#             fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
#             if fd_doy < 60: fd_doy += 365
#             slsmc_si_doy[iloc,iyr,0] = fd_doy



# for iloc,loc in enumerate(freezeup_loc_slsmc):
#     fi_plot = slsmc_fi_doy[iloc,:,0]
#     si_plot = slsmc_si_doy[iloc,:,0]
#     years_plot = years

#     ax_fddoy[0].plot(years_plot,fi_plot, '*',color='gray',alpha=0.5)
#     ax_fddoy[0].plot(years_plot,si_plot, '*',color='black',alpha=0.5)

# for iloc,loc in enumerate(water_name_list):
#     mask_water = ~np.isnan(freezeup_doy[:,iloc])
#     mask_chart = ~np.isnan(chart_fd_doy[iloc,:,0])
#     mask = mask_water & mask_chart
#     ax_fddoy[1].boxplot(freezeup_doy[:,iloc][mask], positions = [iloc],whis=[5, 95],showfliers=True,labels = [station_labels[iloc]])
#     ax_fddoy[1].boxplot(chart_fd_doy[iloc,:,0][mask], positions = [iloc+0.3],whis=[5, 95],showfliers=True,labels = [freezeup_loc_list[iloc]])
#     print(np.nanmedian(freezeup_doy[:,iloc][mask]),np.nanmedian(chart_fd_doy[iloc,:,0][mask]))
#     print(np.nanmean(np.abs(freezeup_doy[:,iloc][mask]-chart_fd_doy[iloc,:,0][mask])))

ax_fddoy[1].grid(True)
ax_fddoy[1].set_ylabel('Freezeup DOY')
plt.xticks(rotation=45)
ax_fddoy[1].set_ylim(336,395)

plt.subplots_adjust(bottom=0.24)

posorig = ax_fddoy[0].get_position()
posnew = [posorig.x0-0.05, posorig.y0, posorig.width*1.5, posorig.height]
ax_fddoy[0].set_position(posnew)
ax_fddoy[0].grid(axis='y')

posorig = ax_fddoy[1].get_position()
posnew = [posorig.x0+0.12, posorig.y0, posorig.width*0.85, posorig.height]
ax_fddoy[1].set_position(posnew)




# AND PLOT DIFFERENCES BETWEEN STATIONS
show_plot = False

if show_plot:
    for iloc,loc in enumerate(water_name_list):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5), sharex=True)

        plt.suptitle(loc)

        fd_main = freezeup_doy[:,iloc]

        for jloc in range(len(water_name_list)):
            if jloc != iloc:
                fd_diff = fd_main - freezeup_doy[:,jloc]

                ax.boxplot(fd_diff[~np.isnan(fd_diff)], positions = [jloc*0.25],whis=[5, 95],showfliers=True,labels = [station_labels[jloc]+'\n(%2i'%np.sum(~np.isnan(fd_diff))+')'])

                plt.xticks(rotation=45)
                plt.subplots_adjust(bottom=0.24)
                ax.set_ylabel('Diff. (days) in freezeup\n ('+loc+' - other stations)')
                ax.grid(True)
                ax.text(-0.25, -15, 'Earlier',color='blue',rotation=90)
                ax.text(-0.25, 4, 'Later',color='red',rotation=90)
                ax.set_ylim(-44,44)
                ax.set_xlim(-0.35,0.25*7)



#%%
# ADD FREEZEUP FROM TW FROM ECCC
water_name_list = ['LaPrairie_cleaned_filled']
station_type = 'ECCC'
station_labels = ['LaPrairie ECCC']

freezeup_loc_list = ['LaPrairie']
freezeup_type = 'charts'

# FIRST FIND FREEZEUP FROM CHARTS AT THE DESIRED LOCATIONS TO MATCH WATER DATA
chart_fd = np.zeros((len(freezeup_loc_list),len(years),3))*np.nan
chart_fd_doy = np.zeros((len(freezeup_loc_list),len(years),2))*np.nan

# for iloc,loc in enumerate(freezeup_loc_list):

#     ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type+'/freezeup_'+freezeup_type+'_'+loc+'.npz',allow_pickle='TRUE')
#     freezeup = ice_data['freezeup_ci'][:,0]
#     freezeup_dt = ice_data['freezeup_ci'][:,1]

#     for iyr,year in enumerate(years):

#         date=(dt.date(year,11,1)-date_ref).days
#         i0 = np.where(time==date)[0][0] - 30
#         i1 = i0+120

#         ci_select = freezeup[i0:i1].copy()
#         dt_select = freezeup_dt[i0:i1].copy()

#         if np.sum(~np.isnan(ci_select)) > 0:
#             i_fd = np.where(~np.isnan(ci_select))[0][0]
#             date_chart = date_ref+dt.timedelta(days=int(ci_select[i_fd]))
#             fd_dt = dt_select[i_fd]

#             fd_yy = int(date_chart.year)
#             fd_mm = int(date_chart.month)
#             fd_dd = int(date_chart.day)

#             chart_fd[iloc,iyr,0] = date_chart.year
#             chart_fd[iloc,iyr,1] = date_chart.month
#             chart_fd[iloc,iyr,2] = date_chart.day

#             fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
#             if fd_doy < 60: fd_doy += 365
#             chart_fd_doy[iloc,iyr,0] = fd_doy
#             chart_fd_doy[iloc,iyr,1] = fd_dt


# THEN APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST, BEFORE
# FINDING FREEZEUP DATES FROM WATER TEMP.
Twater = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan

for iloc,loc in enumerate(water_name_list):
    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][:,1]

    Twater[:,iloc] = Twater_tmp
    if loc == 'Candiac_cleaned_filled':
        Twater[:,iloc] = Twater_tmp-0.8
    if (loc == 'Atwater_cleaned_filled'):
        Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7

    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)


# FIND FREEZEUP DATES FROM TWATER TIME SERIES
freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
breakup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan


for iloc,loc in enumerate(water_name_list):
    Twater_tmp = Twater[:,iloc].copy()

    if round_T:
        if round_type == 'unit':
            Twater_tmp = np.round(Twater_tmp.copy())
        if round_type == 'half_unit':
            Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.

    if smooth_T:
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

    # if Gauss_filter:
    #    Twater_tmp = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=1,order=2)

    if Gauss_filter:
        Twater_DoG1 = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
        Twater_DoG2 = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

    if def_opt == 3:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_DoG1,Twater_DoG2,time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_dates[:,:,iloc] = fd
        # freezeup_temp[:,iloc] = ftw
    else:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_dates[:,:,iloc] = fd
        # freezeup_temp[:,iloc] = ftw


    # fd, T_freezeup, mask_freeze = find_freezeup_Tw(Twater_tmp,time,years,thresh_T = T_thresh, ndays = nd)
    # freezeup_dates[:,:,iloc] = fd

    bd, T_breakup, mask_break = find_breakup_Tw(Twater_tmp,time,years,thresh_T = T_thresh, ndays = nd)
    breakup_dates[:,:,iloc] = bd

    ax_tw.plot(Twater_tmp,color=plt.get_cmap('tab20')(iloc*2+1+8))
    ax_tw.plot(T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2+8))
    ax_tw.plot(T_breakup, 'o',color=plt.get_cmap('tab20')(iloc*2+8))

    for iyr in range(chart_fd.shape[1]):
        if ~np.isnan(chart_fd[iloc,iyr,0]):
            d = (dt.date(int(chart_fd[iloc,iyr,0]),int(chart_fd[iloc,iyr,1]),int(chart_fd[iloc,iyr,2]))-date_ref).days
            fu_i = np.where(time == d)[0][0]
            ax_tw.plot(fu_i, Twater_tmp[fu_i], '*', color='black')


    mean_winter_temp = np.zeros(len(years))*np.nan
    for iyr,year in enumerate(years[:-1]):
        if ~np.isnan(fd[iyr,0]) & ~np.isnan(bd[iyr+1,0]):

            i_fr = np.where(time == ( dt.date(int(fd[iyr,0]),int(fd[iyr,1]),int(fd[iyr,2]))-date_ref).days)[0][0]
            i_br = np.where(time == ( dt.date(int(bd[iyr+1,0]),int(bd[iyr+1,1]),int(bd[iyr+1,2]))-date_ref).days)[0][0]
            Twater_winter = Twater[i_fr:i_br,iloc]
            mean_winter_temp[iyr] = np.nanmean(Twater_winter)


freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
breakup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
for iloc,loc in enumerate(water_name_list):
    for iyr,year in enumerate(years):
        if ~np.isnan(freezeup_dates[iyr,0,iloc]):
            fd_yy = int(freezeup_dates[iyr,0,iloc])
            fd_mm = int(freezeup_dates[iyr,1,iloc])
            fd_dd = int(freezeup_dates[iyr,2,iloc])

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365

            freezeup_doy[iyr,iloc]=fd_doy

        if ~np.isnan(breakup_dates[iyr,0,iloc]):
            bd_yy = int(breakup_dates[iyr,0,iloc])
            bd_mm = int(breakup_dates[iyr,1,iloc])
            bd_dd = int(breakup_dates[iyr,2,iloc])

            bd_doy = (dt.date(bd_yy,bd_mm,bd_dd)-dt.date(bd_yy,1,1)).days + 1

            breakup_doy[iyr,iloc]=bd_doy



for iloc,loc in enumerate(water_name_list):
    ax_fddoy[0].plot(years,freezeup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2+1+2), label=station_labels[iloc],alpha=0.65)
    fd_chart_plot = chart_fd_doy[iloc,:,0][~np.isnan(freezeup_doy[:,iloc])]
    dt_chart_plot = chart_fd_doy[iloc,:,1][~np.isnan(freezeup_doy[:,iloc])]
    years_plot = years[~np.isnan(freezeup_doy[:,iloc])]

    lower_yerror = []
    upper_yerror = []
    for iyr,year in enumerate(years_plot):
        if ~np.isnan(dt_chart_plot[iyr]):
            lower_yerror.append(dt_chart_plot[iyr]-1)
            upper_yerror.append(0)
        else:
            lower_yerror.append(0)
            upper_yerror.append(0)

    yerror = [lower_yerror, upper_yerror]
    ax_fddoy[0].errorbar(years_plot,fd_chart_plot, yerr=yerror, fmt='*',color=plt.get_cmap('tab20')(iloc*2+8),alpha=0.5)

ax_fddoy[0].legend(bbox_to_anchor=(0.0, 0.64, 0.3, 0.3),fontsize=8)
ax_fddoy[0].set_ylim(300,430)
ax_fddoy[0].set_xlabel('Years')
ax_fddoy[0].set_ylabel('Freezeup DOY')



for iloc,loc in enumerate(water_name_list):
    mask_water = ~np.isnan(freezeup_doy[:,iloc])
    mask_chart = ~np.isnan(chart_fd_doy[iloc,:,0])
    mask = mask_water & mask_chart
    ax_fddoy[1].boxplot(freezeup_doy[:,iloc][mask], positions = [iloc],whis=[5, 95],showfliers=True,labels = [station_labels[iloc]])
    ax_fddoy[1].boxplot(chart_fd_doy[iloc,:,0][mask], positions = [iloc+0.3],whis=[5, 95],showfliers=True,labels = [freezeup_loc_list[iloc]])

#%%

# # # ADD FREEZEUP FROM TW FROM SLSMC
# water_name_list = ['StLambert_cleaned_filled']
# station_type = 'SLSMC'

# freezeup_loc_list = ['StLambert']
# freezeup_type = 'charts'

# # FIRST FIND FREEZEUP FROM CHARTS AT THE DESIRED LOCATIONS TO MATCH WATER DATA
# chart_fd = np.zeros((len(freezeup_loc_list),len(years),3))*np.nan
# chart_fd_doy = np.zeros((len(freezeup_loc_list),len(years),2))*np.nan

# # for iloc,loc in enumerate(freezeup_loc_list):

# #     ice_data = np.load(fp+'freezeup_dates_'+ freezeup_type+'/freezeup_'+freezeup_type+'_'+loc+'.npz',allow_pickle='TRUE')
# #     freezeup = ice_data['freezeup_ci'][:,0]
# #     freezeup_dt = ice_data['freezeup_ci'][:,1]

# #     for iyr,year in enumerate(years):

# #         date=(dt.date(year,11,1)-date_ref).days
# #         i0 = np.where(time==date)[0][0] - 30
# #         i1 = i0+120

# #         ci_select = freezeup[i0:i1].copy()
# #         dt_select = freezeup_dt[i0:i1].copy()

# #         if np.sum(~np.isnan(ci_select)) > 0:
# #             i_fd = np.where(~np.isnan(ci_select))[0][0]
# #             date_chart = date_ref+dt.timedelta(days=int(ci_select[i_fd]))
# #             fd_dt = dt_select[i_fd]

# #             fd_yy = int(date_chart.year)
# #             fd_mm = int(date_chart.month)
# #             fd_dd = int(date_chart.day)

# #             chart_fd[iloc,iyr,0] = date_chart.year
# #             chart_fd[iloc,iyr,1] = date_chart.month
# #             chart_fd[iloc,iyr,2] = date_chart.day

# #             fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
# #             if fd_doy < 60: fd_doy += 365
# #             chart_fd_doy[iloc,iyr,0] = fd_doy
# #             chart_fd_doy[iloc,iyr,1] = fd_dt


# # THEN APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST, BEFORE
# # FINDING FREEZEUP DATES FROM WATER TEMP.
# Twater = np.zeros((len(time),len(water_name_list)))*np.nan
# Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
# Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan

# for iloc,loc in enumerate(water_name_list):
#     loc_water_loc = water_name_list[iloc]
#     water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
#     Twater_tmp = water_loc_data['Twater'][:,1]

#     Twater[:,iloc] = Twater_tmp
#     if loc == 'Candiac_cleaned_filled':
#         Twater[:,iloc] = Twater_tmp-0.8
#     if (loc == 'Atwater_cleaned_filled'):
#         Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7

#     dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

#     dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
#     dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
#     dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

#     Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
#     Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)


# # FIND FREEZEUP DATES FROM TWATER TIME SERIES
# freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
# breakup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan

# station_labels = ['St-Lambert SLSMC']


# for iloc,loc in enumerate(water_name_list):
#     Twater_tmp = Twater[:,iloc].copy()

#     if round_T:
#         if round_type == 'unit':
#             Twater_tmp = np.round(Twater_tmp.copy())
#         if round_type == 'half_unit':
#             Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.

#     if smooth_T:
#         Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

#     # if Gauss_filter:
#     #    Twater_tmp = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=1,order=2)

#     if Gauss_filter:
#         Twater_DoG1 = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
#         Twater_DoG2 = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

#     if def_opt == 3:
#         fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_DoG1,Twater_DoG2,time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
#         freezeup_dates[:,:,iloc] = fd
#         # freezeup_temp[:,iloc] = ftw
#     else:
#         fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
#         freezeup_dates[:,:,iloc] = fd
#         # freezeup_temp[:,iloc] = ftw

#     # fd, T_freezeup, mask_freeze = find_freezeup_Tw(Twater_tmp,time,years,thresh_T = T_thresh, ndays = nd)
#     # freezeup_dates[:,:,iloc] = fd

#     bd, T_breakup, mask_break = find_breakup_Tw(Twater_tmp,time,years,thresh_T = T_thresh, ndays = nd)
#     breakup_dates[:,:,iloc] = bd

#     ax_tw.plot(Twater_tmp,color=plt.get_cmap('tab20')(iloc*2+1+16))
#     ax_tw.plot(T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2+16))
#     ax_tw.plot(T_breakup, 'o',color=plt.get_cmap('tab20')(iloc*2+16))

#     for iyr in range(chart_fd.shape[1]):
#         if ~np.isnan(chart_fd[iloc,iyr,0]):
#             d = (dt.date(int(chart_fd[iloc,iyr,0]),int(chart_fd[iloc,iyr,1]),int(chart_fd[iloc,iyr,2]))-date_ref).days
#             fu_i = np.where(time == d)[0][0]
#             ax_tw.plot(fu_i, Twater_tmp[fu_i], '*', color='black')


#     mean_winter_temp = np.zeros(len(years))*np.nan
#     for iyr,year in enumerate(years[:-1]):
#         if ~np.isnan(fd[iyr,0]) & ~np.isnan(bd[iyr+1,0]):

#             i_fr = np.where(time == ( dt.date(int(fd[iyr,0]),int(fd[iyr,1]),int(fd[iyr,2]))-date_ref).days)[0][0]
#             i_br = np.where(time == ( dt.date(int(bd[iyr+1,0]),int(bd[iyr+1,1]),int(bd[iyr+1,2]))-date_ref).days)[0][0]
#             Twater_winter = Twater[i_fr:i_br,iloc]
#             mean_winter_temp[iyr] = np.nanmean(Twater_winter)


# freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
# breakup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
# for iloc,loc in enumerate(water_name_list):
#     for iyr,year in enumerate(years):
#         if ~np.isnan(freezeup_dates[iyr,0,iloc]):
#             fd_yy = int(freezeup_dates[iyr,0,iloc])
#             fd_mm = int(freezeup_dates[iyr,1,iloc])
#             fd_dd = int(freezeup_dates[iyr,2,iloc])

#             fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
#             if fd_doy < 60: fd_doy += 365

#             freezeup_doy[iyr,iloc]=fd_doy

#         if ~np.isnan(breakup_dates[iyr,0,iloc]):
#             bd_yy = int(breakup_dates[iyr,0,iloc])
#             bd_mm = int(breakup_dates[iyr,1,iloc])
#             bd_dd = int(breakup_dates[iyr,2,iloc])

#             bd_doy = (dt.date(bd_yy,bd_mm,bd_dd)-dt.date(bd_yy,1,1)).days + 1

#             breakup_doy[iyr,iloc]=bd_doy



# for iloc,loc in enumerate(water_name_list):
#     ax_fddoy[0].plot(years,freezeup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2+1+12), label=station_labels[iloc],alpha=0.65)
#     fd_chart_plot = chart_fd_doy[iloc,:,0][~np.isnan(freezeup_doy[:,iloc])]
#     dt_chart_plot = chart_fd_doy[iloc,:,1][~np.isnan(freezeup_doy[:,iloc])]
#     years_plot = years[~np.isnan(freezeup_doy[:,iloc])]

#     lower_yerror = []
#     upper_yerror = []
#     for iyr,year in enumerate(years_plot):
#         if ~np.isnan(dt_chart_plot[iyr]):
#             lower_yerror.append(dt_chart_plot[iyr]-1)
#             upper_yerror.append(0)
#         else:
#             lower_yerror.append(0)
#             upper_yerror.append(0)

#     yerror = [lower_yerror, upper_yerror]
#     ax_fddoy[0].errorbar(years_plot,fd_chart_plot, yerr=yerror, fmt='*',color=plt.get_cmap('tab20')(iloc*2+16),alpha=0.5)

# ax_fddoy[0].legend(bbox_to_anchor=(0.0, 0.64, 0.3, 0.3),fontsize=8)
# ax_fddoy[0].set_ylim(300,430)
# ax_fddoy[0].set_xlabel('Years')
# ax_fddoy[0].set_ylabel('Freezeup DOY')



# for iloc,loc in enumerate(water_name_list):
#     mask_water = ~np.isnan(freezeup_doy[:,iloc])
#     mask_chart = ~np.isnan(chart_fd_doy[iloc,:,0])
#     mask = mask_water & mask_chart
#     ax_fddoy[1].boxplot(freezeup_doy[:,iloc][mask], positions = [iloc],whis=[5, 95],showfliers=True,labels = [station_labels[iloc]])
#     ax_fddoy[1].boxplot(chart_fd_doy[iloc,:,0][mask], positions = [iloc+0.3],whis=[5, 95],showfliers=True,labels = [freezeup_loc_list[iloc]])








#%%
ax_fddoy[0].set_yticks(np.arange(300,400,2))





