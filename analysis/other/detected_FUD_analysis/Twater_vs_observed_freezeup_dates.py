#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:33:40 2020

@author: Amelie
"""
import numpy as np

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import running_nanmean,running_mean


#%%

years = [1991,1992,1993,1994,
         1995,1996,1997,1998,1999,
         2000,2001,2002,2003,2004,
         2005,2006,2007,2008,2009,
         2010,2011,2012,2013,2014,
         2015,2016,2017,2018,2019,2020]

water_cities_list = []
water_SLSMC_list = []
water_ECCC_list = []

# water_cities_list = ['Longueuil','Atwater']
# water_cities_list = ['DesBaillets_cleaned_filled']
# water_cities_list = ['Atwater']
water_cities_list = ['DesBaillets_cleaned_filled','Atwater_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
water_ECCC_list = ['Lasalle_cleaned_filled', 'LaPrairie_cleaned_filled']
water_SLSMC_list = ['StLambert_cleaned_filled']



weather_loc = 'MontrealDorval'
# freezeup_loc = 'SouthShoreCanal'
freezeup_loc = 'MontrealPort'


date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

fp = '../../data/processed/'

show_series = False
mean_opt = 'centered'
N = 5

doy_ci_list_tot = []
temp_list_tot = []

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5),sharex=True)
fig3,ax3 = plt.subplots(nrows=1,ncols=1,figsize=(8,5),sharex=True)

# for w in range(1):
# for i in range(1,2):
# for i in range(2,3):
for w in range(3):

    if w == 0:
        water_cat = 'cities'
        water_name_list = water_cities_list
    if w == 1:
        water_cat = 'ECCC'
        water_name_list = water_ECCC_list
    if w == 2:
        water_cat = 'SLSMC'
        water_name_list = water_SLSMC_list

    if len(water_name_list) > 0:

        for i in range(len(water_name_list)):
            date_ci_list_tot = []

            loc_water = water_name_list[i]
            if water_cat == 'cities':
                water_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water+'.npz',allow_pickle='TRUE')
                mrk ='.'
            if water_cat == 'ECCC':
                water_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+loc_water+'.npz',allow_pickle='TRUE')
                mrk ='+'
            if water_cat == 'SLSMC':
                water_data = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_water+'.npz',allow_pickle='TRUE')
                mrk ='+'

            Twater = water_data['Twater']

            file_data = np.load(fp+'weather_NCEI/weather_NCEI_'+weather_loc+'.npz',allow_pickle='TRUE')
            weather_data = file_data['weather_data']

            ice_data = np.load(fp+'freezeup_dates_SLSMC/freezeup_SLSMC_'+freezeup_loc+'.npz',allow_pickle='TRUE')
            freezeup_ci = ice_data['freezeup_ci']
            delay = 0 # this controls if we want to consider
                      # the chart ice date (delay = 0) or some
                      # time before, to better correspond with
                      # first ice or stable ice from SLSMC

            avg_temp = weather_data[:,[0,3]]

            doy_ci_list = []
            temp_list = []

            for year in years:

                date=(dt.date(year,11,1)-date_ref).days
                i0 = np.where(time==date)[0][0] - 30
                i1 = i0+120

                time_select_Tw = Twater[i0:i1,0].copy()
                time_select_Ta = avg_temp[i0:i1,0].copy()

                Tw_select = Twater[i0:i1,1].copy()
                Ta_select = avg_temp[i0:i1,1].copy()
                ci_select = freezeup_ci[i0:i1].copy()

                Ta_select = (Ta_select - 32) * (5/9.) # From F to degrees C.


                if np.sum(~np.isnan(ci_select)) > 0:
                    date_tmp = date_ref+dt.timedelta(days=int(ci_select[np.where(~np.isnan(ci_select))[0][0]][0])-delay)
                    doy_ci = (date_tmp - dt.date(int(year),1,1)).days

                    doy_ci_list.append(doy_ci)
                    temp_list.append(Tw_select[np.where(~np.isnan(ci_select))[0][0]-delay])

                    doy_ci_list_tot.append(doy_ci)
                    date_ci_list_tot.append((date_tmp-date_ref).days)
                    temp_list_tot.append(Tw_select[np.where(~np.isnan(ci_select))[0][0]-delay])

                    ci_select[np.where(~np.isnan(ci_select))[0][0]] = Tw_select[np.where(~np.isnan(ci_select))[0][0]-delay]
                else:
                    doy_ci_list.append(np.nan)
                    temp_list.append(np.nan)

                    doy_ci_list_tot.append(np.nan)
                    date_ci_list_tot.append(np.nan)
                    temp_list_tot.append(np.nan)


                # Tw_select[Tw_select <= 0.2] = np.nan

                Ta_smooth = running_mean(Ta_select,N,mean_opt)

                mask_Tw = ~np.isnan(Tw_select)
                mask_Ta = ~np.isnan(Ta_smooth)
                mask = mask_Tw & mask_Ta

                x1 = Ta_select[mask]
                x2 = Ta_smooth[mask]
                x3 = Tw_select[mask]
                x4 = ci_select[mask]


                if show_series:
                    fig2,ax2 = plt.subplots(nrows=3,ncols=1,figsize=(7,8),sharex=True)

                    ax2[0].plot(np.arange(len(x1))+305,x1,'.-',color=plt.get_cmap('tab20')(1))
                    ax2[0].plot(np.arange(len(x2))+305,x2,'-',color=plt.get_cmap('tab20')(0))

                    ax2[1].plot(np.arange(len(x3))+305,x3,'.-',color=plt.get_cmap('tab20')(4))
                    ax2[1].plot(np.arange(len(x3))+305,x4,'*',markerfacecolor=plt.get_cmap('tab20')(5),markeredgecolor=[0, 0, 0],markersize=15)

                    ax2[2].plot(np.arange(len(x1))+305,x3-x1,'.-',color=plt.get_cmap('tab20')(2))

                    if np.sum(~np.isnan(x4)) < 1:
                        ax2[1].text(360,10,'No freezeup data')

                    ax2[1].set_xlim([305,400])
                    ax2[1].set_ylim([-2,15])

                    ax2[0].yaxis.grid(color=(0.9, 0.9, 0.9))
                    ax2[1].yaxis.grid(color=(0.9, 0.9, 0.9))
                    ax2[2].yaxis.grid(color=(0.9, 0.9, 0.9))

                    ax2[0].set_xlabel('Time (DOY)')
                    ax2[0].set_ylabel('Air Temp. ($^{\circ}$F)')
                    ax2[1].set_ylabel('Water Temp. ($^{\circ}$C)')
                    ax2[2].set_ylabel(r'T$_w$-T$_a$ [C]')
                    plt.suptitle(loc_water+'/'+weather_loc)




            ax.plot(np.arange(22),np.zeros(22),'-',color = [0.9,0.9,0.9],linewidth=0.5)
            ax.plot(np.arange(len(temp_list)),temp_list,marker=mrk,linestyle='',label=loc_water)
            ax.set_xlim([0,len(years)])


            # Twater at freezeup vs DOY
            xarr = np.array(doy_ci_list)
            yarr = np.array(temp_list)
            ax3.plot(np.arange(330,390),np.arange(330,390)*0,':',color=[0.9,0.9,0.9])
            ax3.plot(xarr[~np.isnan(xarr)],yarr[~np.isnan(xarr)],marker=mrk,linestyle='',label=loc_water)
            ax3.set_ylabel('Water temperature at observed freezeup date ($^{\circ}$C)')
            ax3.set_xlabel('Observed freezeup day-of-year')
            ax3.set_ylim([-2,6.6])
            ax3.set_xlim([330,390])
            ax3.legend()



# Twater at freezeup vs years
temp_list_tot = np.array(temp_list_tot)
# ax.boxplot(temp_list_tot[~np.isnan(temp_list_tot)],
#             positions=[4],
#             widths= 0.5,whis='range')
ax.plot(np.arange(30),np.ones(30)*np.nanmean(temp_list_tot),':',color=[0.5,0.5,0.5])
ax.fill_between(np.arange(30), np.ones(30)*np.nanmean(temp_list_tot)+np.nanstd(temp_list_tot),np.ones(30)*np.nanmean(temp_list_tot)-np.nanstd(temp_list_tot),facecolor=[0.9,0.9,0.9], interpolate=True, alpha=0.65)

ax.set_ylim([-2,6.6])
ax.set_ylabel('Water temperature at observed freezeup date ($^{\circ}$C)')
ax.set_xlabel('Year')
ax.set_xticks(np.arange(len(years)))
ax.set_xticklabels([str(years[i])[-2:] for i in range(len(years))],fontsize=10)
ax.legend()


#%%
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


#%%
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



water_cat = 'ECCC'
water_name_list = water_ECCC_list

if len(water_name_list) > 0:
    for i in range(len(water_name_list)):
        loc_water = water_name_list[i]
        water_data = np.load(fp+'Twater_'+water_cat+'/Twater_'+water_cat+'_'+loc_water+'.npz',allow_pickle='TRUE')
        mrk ='.'
        Twater = water_data['Twater'][:,1]

        plt.figure();
        plt.title(loc_water)
        plt.plot(Twater)

        for d in date_ci_list_tot:
            if ~np.isnan(d):
                fu_i = np.where(time == d)[0][0]
                plt.plot(fu_i, Twater[fu_i], '*', color='black')

        freezeup_date, T_freezeup, mask_freeze  = find_freezeup_Tw(Twater,time,years,thresh_T = 0.5, ndays = 5)
        plt.plot(T_freezeup, '*')
        # diff_days=[]
        # for iyr in range(len(years)):
        #     if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
        #         diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        # print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))

        breakup_date, T_breakup, mask_break  = find_breakup_Tw(Twater,time,years,thresh_T = 0.5, ndays = 5)
        plt.plot(T_breakup, '*')


#%%



water_cat = 'cities'
water_name_list = ['Atwater_cleaned_filled']
# water_name_list = ['Candiac_cleaned_filled','DesBaillets_cleaned_filled','Atwater_cleaned_filled','Longueuil_cleaned_filled']
plt.figure();
if len(water_name_list) > 0:
    for i in range(len(water_name_list)):
        loc_water = water_name_list[i]
        water_data = np.load(fp+'Twater_'+water_cat+'/Twater_'+water_cat+'_'+loc_water+'.npz',allow_pickle='TRUE')
        mrk ='.'
        Twater = water_data['Twater'][:,1]

        round_T = False
        if round_T:
            Twater = np.round(Twater.copy() * 2) / 2.

        plt.plot(Twater,label=loc_water)

        print(loc_water)
        for d in date_ci_list_tot:
            if ~np.isnan(d):
                fu_i = np.where(time == d)[0][0]
                plt.plot(fu_i, Twater[fu_i], '*', color='black')

        freezeup_date, T_freezeup, mask_freeze  = find_freezeup_Tw(Twater,time,years,thresh_T = 0.5, ndays = 7)
        plt.plot(T_freezeup, '*')


water_cat = 'eccc'
water_name_list = ['Lasalle_cleaned_filled','LaPrairie_cleaned_filled']
if len(water_name_list) > 0:
    for i in range(len(water_name_list)):
        loc_water = water_name_list[i]
        water_data = np.load(fp+'Twater_'+water_cat+'/Twater_'+water_cat+'_'+loc_water+'.npz',allow_pickle='TRUE')
        mrk ='.'
        Twater = water_data['Twater'][:,1]

        round_T = False
        if round_T:
            Twater = np.round(Twater.copy() * 2) / 2.

        plt.plot(Twater,label=loc_water)

        print(loc_water)
        for d in date_ci_list_tot:
            if ~np.isnan(d):
                fu_i = np.where(time == d)[0][0]
                plt.plot(fu_i, Twater[fu_i], '*', color='black')

        freezeup_date, T_freezeup, mask_freeze  = find_freezeup_Tw(Twater,time,years,thresh_T = 0.5, ndays = 7)
        plt.plot(T_freezeup, '*')


plt.legend()

#%%

chart_freezeup_date = np.zeros((len(years),3))*np.nan
for iyr,d in enumerate(date_ci_list_tot):
    if ~np.isnan(d):
        date_chart = date_ref+dt.timedelta(days=int(d))
        chart_freezeup_date[iyr,0] = date_chart.year
        chart_freezeup_date[iyr,1] = date_chart.month
        chart_freezeup_date[iyr,2] = date_chart.day


water_cat = 'cities'
water_name_list = ['Longueuil_cleaned_filled']
# water_name_list = ['Candiac_cleaned_filled','DesBaillets_cleaned_filled','Atwater_cleaned_filled','Longueuil_cleaned_filled']

if len(water_name_list) > 0:
    for i in range(len(water_name_list)):
        loc_water = water_name_list[i]
        water_data = np.load(fp+'Twater_'+water_cat+'/Twater_'+water_cat+'_'+loc_water+'.npz',allow_pickle='TRUE')
        mrk ='.'
        Twater = water_data['Twater'][:,1]

        round_T = True
        if round_T:
            Twater = np.round(Twater.copy() * 2) / 2.


        plt.figure();
        plt.title(loc_water)
        plt.plot(Twater)

        print(loc_water)
        for d in date_ci_list_tot:
            if ~np.isnan(d):
                fu_i = np.where(time == d)[0][0]
                plt.plot(fu_i, Twater[fu_i], '*', color='black')

        freezeup_date, T_freezeup, mask_freeze  = find_freezeup_Tw(Twater,time,years,thresh_T = 1.5, ndays = 7)
        plt.plot(T_freezeup, '*')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))

        freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater,time,years,thresh_T = 1.0, ndays = 7)
        plt.plot(T_freezeup, 'x')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))

        freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater,time,years,thresh_T = 0.5, ndays = 7)
        plt.plot(T_freezeup, '+')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))

#%%
water_cat = 'cities'
water_name_list = ['Longueuil_cleaned_filled']
# water_name_list = ['Candiac_cleaned_filled','DesBaillets_cleaned_filled','Atwater_cleaned_filled','Longueuil_cleaned_filled']
plt.figure();
if len(water_name_list) > 0:
    for i in range(len(water_name_list)):
        loc_water = water_name_list[i]
        water_data = np.load(fp+'Twater_'+water_cat+'/Twater_'+water_cat+'_'+loc_water+'.npz',allow_pickle='TRUE')
        mrk ='.'
        Twater = water_data['Twater'][:,1]


        for d in date_ci_list_tot:
            if ~np.isnan(d):
                fu_i = np.where(time == d)[0][0]
                plt.plot(fu_i, Twater[fu_i], '*', color='black')

        round_T = False
        if round_T:
            Twater = np.round(Twater.copy() * 2) / 2.

        for N in [1,3,5,7,15,31]:
            print(N)
            Twater_smooth = running_nanmean(Twater.copy(),N,mean_opt)
            freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater_smooth,time,years,thresh_T = 0.5, ndays = 7)
            plt.plot(T_freezeup, '+')
            diff_days=[]
            for iyr in range(len(years)):
                if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                    diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
            # print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))

            plt.plot(Twater_smooth,label=loc_water)

plt.legend()


#%%

water_cat = 'ECCC'
water_name_list = water_ECCC_list

if len(water_name_list) > 0:
    for i in range(len(water_name_list)):
        loc_water = water_name_list[i]
        water_data = np.load(fp+'Twater_'+water_cat+'/Twater_'+water_cat+'_'+loc_water+'.npz',allow_pickle='TRUE')
        mrk ='.'
        Twater = water_data['Twater'][:,1]

        plt.figure();
        plt.title(loc_water)
        plt.plot(Twater)

        for d in date_ci_list_tot:
            if ~np.isnan(d):
                fu_i = np.where(time == d)[0][0]
                plt.plot(fu_i, Twater[fu_i], '*', color='black')

        freezeup_date, T_freezeup, mask_freeze  = find_freezeup_Tw(Twater,time,years,thresh_T = 1.5, ndays = 5)
        plt.plot(T_freezeup, '*')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))


        freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater,time,years,thresh_T = 1.0, ndays = 5)
        plt.plot(T_freezeup, 'x')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))


        freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater,time,years,thresh_T = 0.5, ndays = 5)
        plt.plot(T_freezeup, '*')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))


        freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater,time,years,thresh_T = 0.1, ndays = 5)
        plt.plot(T_freezeup, '+')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))

#%%
# water_cat = 'ECCC'
# water_name_list = water_ECCC_list
# plt.figure();
# if len(water_name_list) > 0:
#     for i in range(len(water_name_list)):
#         loc_water = water_name_list[i]
#         water_data = np.load(fp+'Twater_'+water_cat+'/Twater_'+water_cat+'_'+loc_water+'.npz',allow_pickle='TRUE')
#         mrk ='.'
#         Twater = water_data['Twater'][:,1]

#         plt.title(loc_water)
#         plt.plot(Twater)

#%%

water_cat = 'SLSMC'
water_name_list = water_SLSMC_list

if len(water_name_list) > 0:
    for i in range(len(water_name_list)):
        loc_water = water_name_list[i]
        water_data = np.load(fp+'Twater_'+water_cat+'/Twater_'+water_cat+'_'+loc_water+'.npz',allow_pickle='TRUE')
        mrk ='.'
        Twater = water_data['Twater'][:,1]

        plt.figure();
        plt.title(loc_water)
        plt.plot(Twater)

        for d in date_ci_list_tot:
            if ~np.isnan(d):
                fu_i = np.where(time == d)[0][0]
                plt.plot(fu_i, Twater[fu_i], '*', color='black')

        freezeup_date, T_freezeup, mask_freeze  = find_freezeup_Tw(Twater,time,years,thresh_T = 1.5, ndays = 7)
        plt.plot(T_freezeup, '*')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))


        freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater,time,years,thresh_T = 1.0, ndays = 7)
        plt.plot(T_freezeup, 'x')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))


        freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater,time,years,thresh_T = 0.5, ndays = 7)
        plt.plot(T_freezeup, '+')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))

        freezeup_date, T_freezeup, mask_freeze = find_freezeup_Tw(Twater,time,years,thresh_T = 0.1, ndays = 7)
        plt.plot(T_freezeup, '+')
        diff_days=[]
        for iyr in range(len(years)):
            if ~np.isnan(chart_freezeup_date[iyr,0]) & ~np.isnan(freezeup_date[iyr,0]):
                diff_days.append((dt.date(int(chart_freezeup_date[iyr,0]),int(chart_freezeup_date[iyr,1]),int(chart_freezeup_date[iyr,2]))-dt.date(int(freezeup_date[iyr,0]),int(freezeup_date[iyr,1]),int(freezeup_date[iyr,2]))).days)
        print(np.mean(diff_days),np.min(diff_days),np.max(diff_days))








#%%
# def find_freezeup_Tw_ini(Twater_in, thresh_T = 2.0, ndays = 7):

#     # mask_tmp is True if T is below thresh_T:
#     mask_tmp = Twater_in <= thresh_T

#     mask_freezeup = mask_tmp.copy()
#     mask_freezeup[:] = False

#     for im in range(1,mask_freezeup.size):

#         if (im == 1) | (~mask_tmp[im-1]):
#             # start new group
#             sum_m = 0
#             if ~mask_tmp[im]:
#                 sum_m = 0
#             else:
#                 sum_m +=1
#                 istart = im

#         else:
#             if mask_tmp[im]:
#                 sum_m += 1

#                 if sum_m >= ndays:
#                     # Temperature has been lower than thresh_T
#                     # for more than (or equal to) ndays.
#                     # Define freezeup date as first date of group
#                     mask_freezeup[istart] = True

#                     sum_m = 0 # Put back sum to zero

#     Twater_out = Twater_in.copy()
#     Twater_out[~mask_freezeup] = np.nan


#     return Twater_out, mask_freezeup

# water_cat = 'cities'
# water_name_list = ['DesBaillets_cleaned_filled']

# if len(water_name_list) > 0:
#     for i in range(len(water_name_list)):
#         loc_water = water_name_list[i]
#         water_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water+'.npz',allow_pickle='TRUE')
#         mrk ='.'
#         Twater = water_data['Twater'][:,1]


# plt.figure();
# plt.plot(Twater)

# T_freezeup, mask_freeze = find_freezeup_Tw_ini(Twater,thresh_T = 1.5, ndays = 7)
# plt.plot(T_freezeup, '*')

# T_freezeup, mask_freeze = find_freezeup_Tw_ini(Twater,thresh_T = 1.0, ndays = 7)
# plt.plot(T_freezeup, '*')

# T_freezeup, mask_freeze = find_freezeup_Tw_ini(Twater,thresh_T = 0.5, ndays = 7)
# plt.plot(T_freezeup, '*')


# for d in date_ci_list_tot:
#     if ~np.isnan(d):
#         fu_i = np.where(time == d)[0][0]
#         plt.plot(fu_i, Twater[fu_i], '*', color='black')


#%%

# def find_freezeup_Tw(Twater_in, time, years, thresh_T = 2.0, ndays = 7, date_ref = dt.date(1900,1,1)):

#     # mask_tmp is True if T is below thresh_T:
#     mask_tmp = Twater_in <= thresh_T

#     mask_freezeup = mask_tmp.copy()
#     mask_freezeup[:] = False

#     freezeup_date=np.zeros((len(years),3))*np.nan
#     iyr = 0

#     for im in range(1,mask_freezeup.size):

#         # date_im = date_ref+dt.timedelta(days=int(time[im]))
#         # doy_im = (date_im - dt.date(int(date_im.year),1,1)).days

#         # if iyr > 0 & iyr < len(years):
#         #     if doy_im > 300:
#         #         # print (iyr,im,date_im.year)
#         #         iyr = np.where(np.array(years) == date_im.year)[0][0]
#         #     elif doy_im < 60:
#         #         # print (iyr,im,date_im.year)
#         #         iyr = np.where(np.array(years) == date_im.year-1)[0][0]

#         # iyr = np.where(np.array(years) == date_im.year)[0][0]
#         # if (date_im.year == years[iyr]) & (date_im.month > 10):
#         # if (date_im.year == years[iyr]+1) & (date_im.month < 5):
#         # if time[im] < (dt.date(years[0],11,1)-date_ref).days:
#         #     iyr = 0
#         # elif (time[im] > (dt.date(years[iyr],11,1)-date_ref).days) & (time[im] < (dt.date(years[iyr]+1,3,1)-date_ref).days):
#         #     iyr = iyr
#         # it0 = (dt.date(years[iyr],11,1)-date_ref).days
#         # it1 = (dt.date(years[iyr]+1,3,1)-date_ref).days

#         # 33907:34027 iyr = 0
#         # 34272:34392 iyr = 1
#         # 34637:34757 iyr = 2
#         # 35002:35123 iyr = 3


#         # 1992-11-01 : 1993-03-01 = 1992, iyr = 0
#         # 1993-11-01 : 1994-03-01 = 1993, iyr = 1
#         # 1994-11-01 : 1995-03-01 = 1994, iyr = 2
#         # 1995-11-01 : 1996-03-01 = 1995, iyr = 3

#         if (im == 1) | (~mask_tmp[im-1]):

#             sum_m = 0
#             if ~mask_tmp[im]:
#                 sum_m = 0
#             else:
#                 # start new group
#                 if sum_m ==0: iend= 0
#                 sum_m +=1
#                 istart = im
#                 # print("START:",im,sum_m,istart,iend)


#         else:
#             if mask_tmp[im]:
#                 sum_m += 1

#                 if (sum_m >= ndays) & (iend == 0):
#                     # Temperature has been lower than thresh_T
#                     # for more than (or equal to) ndays.
#                     # Define freezeup date as first date of group
#                     date_tmp = date_ref+dt.timedelta(days=int(time[istart]))

#                     if iyr == 0:
#                         freezeup_date[iyr,0] = date_tmp.year
#                         freezeup_date[iyr,1] = date_tmp.month
#                         freezeup_date[iyr,2] = date_tmp.day
#                         iyr += 1
#                         mask_freezeup[istart] = True
#                         iend = 1
#                     else:
#                         if freezeup_date[iyr-1,0] == date_tmp.year:
#                             # 2012 01 05
#                             # 2012 01 07 NO
#                             # 2012 12 22 YES

#                             if (freezeup_date[iyr-1,1] < 5) & (date_tmp.month > 10):
#                                 freezeup_date[iyr,0] = date_tmp.year
#                                 freezeup_date[iyr,1] = date_tmp.month
#                                 freezeup_date[iyr,2] = date_tmp.day
#                                 iyr += 1
#                                 mask_freezeup[istart] = True
#                                 iend = 1

#                         elif date_tmp.year == freezeup_date[iyr-1,0]+1:
#                             # 2012 12 22
#                             # 2013 01 14 NO
#                             # 2013 12 24 YES

#                             #2014 01 03 (2013 season)
#                             #2015 01 13 (2014 season)

#                             if (date_tmp.month > 10):
#                                 freezeup_date[iyr,0] = date_tmp.year
#                                 freezeup_date[iyr,1] = date_tmp.month
#                                 freezeup_date[iyr,2] = date_tmp.day
#                                 iyr += 1
#                                 mask_freezeup[istart] = True
#                                 iend = 1
#                             elif (date_tmp.month < 5) & (freezeup_date[iyr-1,1] < 5) :
#                                 freezeup_date[iyr,0] = date_tmp.year
#                                 freezeup_date[iyr,1] = date_tmp.month
#                                 freezeup_date[iyr,2] = date_tmp.day
#                                 iyr += 1
#                                 mask_freezeup[istart] = True
#                                 iend = 1


#                         elif date_tmp.year == freezeup_date[iyr-1,0]+2:
#                             if (date_tmp.month < 5):
#                                 freezeup_date[iyr,0] = date_tmp.year
#                                 freezeup_date[iyr,1] = date_tmp.month
#                                 freezeup_date[iyr,2] = date_tmp.day
#                                 iyr += 1
#                                 mask_freezeup[istart] = True
#                                 iend = 1

#                         else:
#                             if (date_tmp.year > freezeup_date[iyr-1,0]+2):
#                                 freezeup_date[iyr,0] = date_tmp.year
#                                 freezeup_date[iyr,1] = date_tmp.month
#                                 freezeup_date[iyr,2] = date_tmp.day
#                                 iyr += 1
#                                 mask_freezeup[istart] = True
#                                 iend = 1
#                             else:
#                                 print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_tmp.year,date_tmp.month,date_tmp.day)


#     Twater_out = Twater_in.copy()
#     Twater_out[~mask_freezeup] = np.nan

#     return freezeup_date, Twater_out, mask_freezeup
