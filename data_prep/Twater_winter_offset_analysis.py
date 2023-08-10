#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:13:14 2021

@author: Amelie
"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'

#%%
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import scipy.stats as sp

import datetime as dt
import calendar

import matplotlib.pyplot as plt

#%%
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


def mask_freezing(Twater_in, dTdt_in, d2Tdt2_in, thresh_T = 4.0, thresh_dTdt = 0.02, thresh_d2Tdt2 = 0.02, ndays = 7):

    zero_dTdt_mask = np.abs(dTdt_in) < thresh_dTdt
    zero_d2Tdt2_mask = np.abs(d2Tdt2_in) < thresh_d2Tdt2
    zero_T_mask = np.abs(Twater_in) < thresh_T

    # T is below thresh_T and dTdt is below thresh_dTdt and d2Tdt2 is below thresh_d2Tdt2:
    mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
    mask_winter = mask_tmp.copy()
    mask_winter[:] = False

    T_freeze = []
    for im in range(1,mask_winter.size):

        if (im == 1) | (~mask_tmp[im-1]):
            # start new group
            sum_m = 0
            if ~mask_tmp[im]:
                sum_m = 0
            else:
                sum_m +=1
                istart_freeze = im

        else:
            if mask_tmp[im]:
                sum_m += 1
            else:
                # This is the end of the group of constant dTdt,
                # so count the total number of points in group,
                # and remove whole group if total is larger than
                # ndays
                iend_freeze = im
                if sum_m >= ndays:
                    mask_winter[istart_freeze:iend_freeze] = True
                    T_freeze.append(np.nanmean(Twater_in[istart_freeze:iend_freeze]))
                sum_m = 0 # Put back sum to zero

    Twater_out= Twater_in.copy()
    Twater_out[mask_winter] = np.nan

    return Twater_out, T_freeze, mask_winter


def running_nanmean(x, N=3):
    xmean = np.ones(x.shape[0])*np.nan
    temp = np.vstack([x[i:-(N-i)] for i in range(N)]) # stacks vertically the strided arrays
    temp = np.nanmean(temp, axis=0)

    # mean is taken from values before:
    # xmean[N-1:N-1+temp.shape[0]] = temp

    # mean is centered:
    xmean[int((N-1)/2):int((N-1)/2)+temp.shape[0]] = temp

    return xmean


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
# # plot = True
# plot = False

# fill = True
# fill_type = 'linear'

# # save = True
# save = False

# years = [1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005]
# years = [2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020]

years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

# water_name_list = ['DesBaillets_cleaned_filled']
# water_name_list = ['Longueuil_cleaned_filled']
water_name_list = ['Longueuil_updated_cleaned_filled']
# water_name_list = ['Atwater_cleaned_filled']
# water_name_list = ['Candiac_cleaned_filled']
water_name_list = ['Longueuil_updated_cleaned_filled','Atwater_cleaned_filled','DesBaillets_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
station_type = 'cities'

# water_name_list = ['Lasalle_cleaned_filled']
# water_name_list = ['Lasalle_cleaned_filled','LaPrairie_cleaned_filled']
# station_type = 'ECCC'

# water_name_list = ['StLambert_cleaned_filled']
# station_type = 'SLSMC'

fp = local_path+'slice/data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

yr_start = 0 # 1992
yr_end = len(years) -1 #2021

istart = np.where(time == (dt.date(years[yr_start],1,1)-date_ref).days)[0][0]
iend = np.where(time == (dt.date(years[yr_end],12,31)-date_ref).days)[0][0]
iend += 1
if istart < 5000: istart = 0

time_select = time[istart:iend]
years = np.array(years)

#%%
# DETECT PERIODS OF NO TEMPRATURE VARIATION DURING EACH WINTER (DEFINED BY
# T< 4C, DTDt < 0.02 AND D2TDt2 <0.02, FOR AT LEAST 7 DAYS) AND FIND THE
# AVERAGE TEMPERATURE DURING THESE MOMENTS. IF IT IS WITHIN -0.5 TO 0.5 DEG.
# (IE UNCERTAINTY ON TEMPERATURE MEASUREMENTS) THEN THE DATA IS FINE AND
# DOES NOT NEED AN OFFSET SINCE THE WATER IS ALREADY WITHIN THE FREEZING
# TEMPERATURE RANGE. IF IT IS LARGER THAN 0.5 DEG, THEN CHECK IF IT IS
# SIMILAR EVERY YEAR SO THAT WE CAN JUST OFFSET BY A SINGLE NUMBER FOR
# ALL THE TIME SERIES OR IF JUST A CERTAIN PERIOD NEEDS TO BE OFFSET.

# Find winter offset for data sets:
Twater = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt_f = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt_b = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan


for iloc,loc in enumerate(water_name_list):
    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][istart:iend,1]

    Twater[:,iloc] = Twater_tmp

    # THESE OFFSET WERE ALREADY FOUND AND THIS IS THE ADJUSTEMENT NEEDED:
    # if loc == 'Candiac_cleaned_filled':
    #     Twater[:,iloc] = Twater_tmp-0.8
    # if (loc == 'Atwater_cleaned_filled'):
    #     Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7
    # if (loc == 'Longueuil_updated_cleaned_filled'):
    #     Twater[14329:,iloc] = Twater[14329:,iloc]- 0.78

    # FOR THE THRESHOLD TO WORK TO DETECT PERIOD OF NO VARIATION, WE NEED TO
    # FIRST SMOOTH THE TEMPERATURE TIME SERIES
    smooth_T = True
    if smooth_T:
        N = 7
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N)

    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_dTdt_f[:,iloc] = dTdt_tmp[:,0]
    Twater_dTdt_b[:,iloc] = dTdt_tmp[:,1]

    Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

    Twater_winter, Twater_freezeup, winter_mask = mask_freezing(Twater[:,iloc], Twater_dTdt[:,iloc], Twater_d2Tdt2[:,iloc], thresh_T = 4.0, thresh_dTdt = 0.02, thresh_d2Tdt2 = 0.02, ndays = 7)

    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(6,5))
    ax[0].plot(Twater[:,iloc])
    ax[0].plot(Twater_winter)
    plt.suptitle(loc)
    ax[0].plot(np.arange(Twater[:,iloc].size),np.ones(Twater[:,iloc].size)*0,color='black')
    ax[0].plot(np.arange(Twater[:,iloc].size),np.ones(Twater[:,iloc].size)*-0.5,color='gray')
    ax[0].plot(np.arange(Twater[:,iloc].size),np.ones(Twater[:,iloc].size)*0.5,color='gray')
    ax[1].plot(Twater_freezeup)
    print(loc, np.nanmean(np.array(Twater_freezeup)[0:3])) # Atwater offset
    print(loc, np.nanmean(np.array(Twater_freezeup)[:-1])) # Candiac offset
    print(loc, np.nanmean(np.array(Twater_freezeup)[:-10])) # Candiac offset
    print(loc, np.nanmean(np.array(Twater_freezeup)[:])) # Candiac offset


# PLOT ADJUSTED WATER TEMPERATURE TIME SERIES ON TOP OF EACH OTHER FOR ALL
# STATIONS TO COMPARE.
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
for iloc,loc in enumerate(water_name_list):
    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][istart:iend,1]

    smooth_T = True
    if smooth_T:
        N = 7
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N)

    Twater[:,iloc] = Twater_tmp

    if loc == 'Candiac_cleaned_filled':
        Twater[:,iloc] = Twater[:,iloc]-0.8

    if (loc == 'Atwater_cleaned_filled'):
        Twater[0:12490,iloc] = Twater[0:12490,iloc]-0.7

    if (loc == 'Longueuil_updated_cleaned_filled'):
        Twater[14329:,iloc] = Twater[14329:,iloc]- 0.78
    ax.plot(Twater[:,iloc],label=loc)

ax.legend()

#%%
# PLOT AVERAGE ANNUAL WATER TEMPRATURE VS YEARS
fig_clim,ax_clim = plt.subplots(nrows=1,ncols=1,figsize=(6,5))

for iloc,loc in enumerate(water_name_list):

    Twater_mean = np.zeros(len(years))*np.nan
    for iyr,year in enumerate(years):
        istart_year = np.where(time == ((dt.date(year,1,1)-date_ref).days) )
        if len(istart_year[0]) > 0:
            istart_year = np.max([0,istart_year[0][0]])
        else:
            istart_year = 0
        iend_year = np.where(time == ((dt.date(year+1,1,1)-date_ref).days) )
        if len(iend_year[0]) > 0:
            iend_year = np.min([time.size,iend_year[0][0]])
        else:
            iend_year = time.size

        if (np.sum(~np.isnan(Twater[istart_year:iend_year,iloc])) > 330):
            Twater_mean[iyr] = np.nanmean(Twater[istart_year:iend_year,iloc])

    ax_clim.plot(years,Twater_mean,'o',label=loc, alpha=0.5)
    lincoeff, Rsqr = linear_fit(years[~np.isnan(Twater_mean)],Twater_mean[~np.isnan(Twater_mean)])
    ax_clim.plot(years,lincoeff[0]*years+lincoeff[1],'-',color=plt.get_cmap('tab20')(iloc*2+1) )
    ax_clim.text(1992,13-0.3*iloc,'%3.2f'%lincoeff[0]+'x '+'%3.2f'%lincoeff[1] +' (R$^{2}$: %3.2f'%(Rsqr)+')',color=plt.get_cmap('tab20')(iloc*2+1) )


ax_clim.legend()
ax_clim.set_ylim(7,15)

ax_clim.set_xlabel('Years')
ax_clim.set_ylabel('Annual Avg. Water Temp. (deg. C)')


#%%%
# freezeup_loc = 'SouthShoreCanal'
# # freezeup_loc = 'MontrealPort'

# doy_ci_list_tot = []
# temp_list_tot = []
# for iloc,loc in enumerate(water_name_list):
#     date_ci_list_tot = []

#     ice_data = np.load(fp+'freezeup_dates_SLSMC/freezeup_SLSMC_'+freezeup_loc+'.npz',allow_pickle='TRUE')
#     # freezeup_ci = ice_data['freezeup_fi']
#     # label_obs = 'SLSMC first ice'
#     # freezeup_ci = ice_data['freezeup_si']
#     # label_obs = 'SLSMC stable ice'
#     freezeup_ci = ice_data['freezeup_ci']
#     label_obs = 'Ice Charts'

#     doy_ci_list = []
#     temp_list = []

#     for year in years:

#         date=(dt.date(year,11,1)-date_ref).days
#         i0 = np.where(time==date)[0][0] - 30
#         i1 = i0+120

#         time_select_Tw = time[i0:i1].copy()
#         Tw_select = Twater[i0:i1,iloc].copy()
#         ci_select = freezeup_ci[i0:i1].copy()

#         if np.sum(~np.isnan(ci_select)) > 0:
#             date_tmp = date_ref+dt.timedelta(days=int(ci_select[np.where(~np.isnan(ci_select))[0][0]][0]))
#             doy_ci = (date_tmp - dt.date(int(year),1,1)).days

#             doy_ci_list.append(doy_ci)
#             temp_list.append(Tw_select[np.where(~np.isnan(ci_select))[0][0]])

#             doy_ci_list_tot.append(doy_ci)
#             date_ci_list_tot.append((date_tmp-date_ref).days)
#             temp_list_tot.append(Tw_select[np.where(~np.isnan(ci_select))[0][0]])

#             ci_select[np.where(~np.isnan(ci_select))[0][0]] = Tw_select[np.where(~np.isnan(ci_select))[0][0]]
#         else:
#             doy_ci_list.append(np.nan)
#             temp_list.append(np.nan)

#             doy_ci_list_tot.append(np.nan)
#             date_ci_list_tot.append(np.nan)
#             temp_list_tot.append(np.nan)



# chart_freezeup_date = np.zeros((len(years),3))*np.nan
# chart_fd_doy = np.zeros((len(years)))*np.nan
# for iyr,d in enumerate(date_ci_list_tot):
#     if ~np.isnan(d):
#         date_chart = date_ref+dt.timedelta(days=int(d))
#         chart_freezeup_date[iyr,0] = date_chart.year
#         chart_freezeup_date[iyr,1] = date_chart.month
#         chart_freezeup_date[iyr,2] = date_chart.day

#         fd_yy = int(date_chart.year)
#         fd_mm = int(date_chart.month)
#         fd_dd = int(date_chart.day)

#         fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
#         if fd_doy < 60: fd_doy += 365
#         chart_fd_doy[iyr] = fd_doy


# freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
# breakup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
# plt.figure()
# for iloc,loc in enumerate(water_name_list):
#     Twater_tmp = Twater[:,iloc]
#     plt.plot(Twater_tmp,color=plt.get_cmap('tab20')(iloc*2+1))

#     fd, T_freezeup, mask_freeze = find_freezeup_Tw(Twater_tmp,time,years,thresh_T = 0.25, ndays = 7)
#     plt.plot(T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2))
#     freezeup_dates[:,:,iloc] = fd

#     bd, T_breakup, mask_break = find_breakup_Tw(Twater_tmp,time,years,thresh_T = 0.25, ndays = 7)
#     plt.plot(T_breakup, 'o',color=plt.get_cmap('tab20')(iloc*2))
#     breakup_dates[:,:,iloc] = bd


#     for d in date_ci_list_tot:
#         if ~np.isnan(d):
#             fu_i = np.where(time == d)[0][0]
#             plt.plot(fu_i, Twater_tmp[fu_i], '*', color='black')


#     mean_winter_temp = np.zeros(len(years))*np.nan
#     mean_freezing_temp = np.zeros(len(years))*np.nan
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


# fig_fddoy,ax_fddoy = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
# for iloc,loc in enumerate(water_name_list):
#     ax_fddoy.plot(years,freezeup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2), label=loc,alpha=0.5)
#     # y_plot = freezeup_doy[:,iloc].copy()
#     # lincoeff, Rsqr = linear_fit(years[~np.isnan(y_plot)],y_plot[~np.isnan(y_plot)])
#     # ax_fddoy.plot(years,lincoeff[0]*years+lincoeff[1],'-',color=plt.get_cmap('tab20')(iloc*2+1) )
#     # ax_fddoy.text(2010,413-6*iloc,'%3.2f'%lincoeff[0]+'x '+'%3.2f'%lincoeff[1] +' (R$^{2}$: %3.2f'%(Rsqr)+')',color=plt.get_cmap('tab20')(iloc*2+1) )

# ax_fddoy.plot(years,chart_fd_doy,'*',color='black',label=label_obs)
# ax_fddoy.legend()
# ax_fddoy.set_ylim(300,430)
# fig_fddoy.suptitle('Freezeup DOY')
# ax_fddoy.set_xlabel('Years')
# ax_fddoy.set_ylabel('Freezeup DOY')



# fig_bddoy,ax_bddoy = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
# for iloc,loc in enumerate(water_name_list):
#     ax_bddoy.plot(years,breakup_doy[:,iloc],'o',color=plt.get_cmap('tab20')(iloc*2), label=loc,alpha=0.5)
#     y_plot = breakup_doy[:,iloc].copy()
#     lincoeff, Rsqr = linear_fit(years[~np.isnan(y_plot)],y_plot[~np.isnan(y_plot)])
#     ax_bddoy.plot(years,lincoeff[0]*years+lincoeff[1],'-',color=plt.get_cmap('tab20')(iloc*2+1) )
#     ax_bddoy.text(1992,143-6*iloc,'%3.2f'%lincoeff[0]+'x '+'%3.2f'%lincoeff[1] +' (R$^{2}$: %3.2f'%(Rsqr)+')',color=plt.get_cmap('tab20')(iloc*2+1) )

# ax_bddoy.legend()
# ax_bddoy.set_ylim(0,160)
# fig_bddoy.suptitle('Breakup DOY')
# ax_bddoy.set_xlabel('Years')
# ax_bddoy.set_ylabel('Breakup DOY')


#%%

# for iiloc in range(len(water_name_list)):
#     for jiloc in range(len(water_name_list)):

#         if iiloc != jiloc:
#             fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
#             ax.plot(freezeup_doy[:,iiloc],freezeup_doy[:,jiloc],'o')
#             plt.xlabel(water_name_list[iiloc])
#             plt.ylabel(water_name_list[jiloc])
#             ax.set_xlim(340,390)
#             ax.set_ylim(340,390)
#             ax.grid()

#%%
# for iiloc in range(len(water_name_list)):
#     for jiloc in range(len(water_name_list)):

#         if iiloc != jiloc:
#             fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
#             plt.hist(freezeup_doy[:,iiloc]-freezeup_doy[:,jiloc])
#             plt.suptitle(water_name_list[iiloc]+' - ' + water_name_list[jiloc])


# import pandas as pd
# from pandas.plotting import scatter_matrix
# df = pd.DataFrame(freezeup_doy, columns = water_name_list)
# axS = scatter_matrix(df,figsize = (6, 6))

# for i in range(4):
#     for j in range(4):
#         if i != j:
#             axS[i,j].set_xlim(340,390)
#             axS[i,j].set_xticks([360,380])
#             axS[i,j].set_ylim(340,390)
#             axS[i,j].set_yticks([360,380])
#             axS[i,j].grid(axis='both')




