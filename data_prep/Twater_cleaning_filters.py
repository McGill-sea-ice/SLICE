#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:58:03 2021

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

def rolling_clim(var_in, clim_years, t, Nwindow = 31):
    # NOTE: Only odd window size are possible
    # t = time[istart:iend]
    # clim_years = years[yr_start:yr_end+1]
    # var_in = Twater[:,iloc]

    date_ref = dt.date(1900,1,1)

    # First re-arrange data to have each window, for each DOY, each year
    data = np.zeros((Nwindow,366,len(clim_years)))*np.nan
    var_tmp = var_in.copy()

    for it in range(var_tmp.shape[0]):
        iw0 = np.max([0,it-int((Nwindow-1)/2)])
        iw1 = np.min([it+int((Nwindow-1)/2)+1,len(var_tmp)])

        var_window = np.zeros(Nwindow)*np.nan
        var_window[0:len(var_tmp[iw0:iw1])] = var_tmp[iw0:iw1]

        date_mid = date_ref+dt.timedelta(days=int(t[it]))
        year_mid = date_mid.year
        month_mid = date_mid.month
        day_mid = date_mid.day

        if len(np.where(clim_years == year_mid)[0]) > 0:
            iyear = np.where(clim_years == year_mid)[0][0]
            doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days

            data[:,doy,iyear] = var_window

            if not calendar.isleap(year_mid) and (doy == 364):
                if it == (var_tmp.shape[0]-1):
                    data[:,365,iyear] = np.ones(Nwindow)*np.nan
                else:
                    imid = int((Nwindow-1)/2)

                    var_window_366 = np.zeros((Nwindow))*np.nan
                    var_window_366[imid] = np.array(np.nanmean([var_tmp[it],var_tmp[it+1]]))
                    var_window_366[0:imid] = var_tmp[int(it+1-((Nwindow-1)/2)):it+1]
                    var_window_366[imid+1:Nwindow] = var_tmp[it+1:int(it+1+((Nwindow-1)/2))]
                    data[:,365,iyear] = var_window_366

    # Then, find the window climatological mean and std for each DOY
    clim_mean = np.zeros((len(t)))*np.nan
    clim_std = np.zeros((len(t)))*np.nan

    clim_mean_tmp = np.nanmean(data,axis=(0,2))
    clim_std_tmp = np.nanstd(data,axis=(0,2))

    for iyr,year in enumerate(clim_years):

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(t == date)[0][0]
        i1 = i0+365+calendar.isleap(year)

        clim_mean[i0:i1] = clim_mean_tmp[0:365+int(calendar.isleap(year))]
        clim_std[i0:i1] = clim_std_tmp[0:365+int(calendar.isleap(year))]


    return clim_mean, clim_std



def mask_lines_and_steps(Twater_in, d2Tdt2_in, thresh_T = 2.0, thresh_d2Tdt2 = 0.04, ndays = 7):

    # THIS WILL MASK PERIODS OF 7 DAYS + OF CONSTANT DT/DT
    # INDICATIING EITHER THAT T IS CONSTANT FOR 7 DAYS OR MORE
    # WHICH WOULD INDICATE A PROBLEM WITH THE TEMPERATURE PROBE
    # OR IT WILL ALSO IDENTIFY LINEAR SEGMENTS SUCH AS IN 2010.
    # THE THRESHOLD OF 0.04 MIGHT NEED TO BE ADJUSTED FOR OTHER
    # DATA SETS...

    # FOR ATWATER, THIS FILTER MIGHT ALSO BE PROBLEMATIC BECAUSE
    # WHEN T IS ROUNDED TO THE UNIT ONLY, THERE ARE PERIODS WITH
    # SEVERAL DAYS MARKED AS THE SAME TEMPERATURE...

    # *** ALSO THE NUMBER OF DAYS (7 DAYS OR MORE) SHOULD BE
    # CHOSEN TO BE CONSISTENT WITH THE LARGEST INTERVAL I CHOOSE
    # TO PATCH WITH LINEAR INTERPOLATION.

    zero_d2Tdt2_mask = np.abs(d2Tdt2_in) < thresh_d2Tdt2
    zero_T_mask = np.abs(Twater_in) < thresh_T

    # T is above thresh_T and d2Tdt2 is below thresh_d2Tdt2:
    mask_tmp = ~zero_T_mask & zero_d2Tdt2_mask
    mask_line = mask_tmp.copy()
    mask_line[:] = False

    for im in range(1,mask_line.size):

        if (im == 1) | (~mask_tmp[im-1]):
            # start new group
            sum_m = 0
            if ~mask_tmp[im]:
                sum_m = 0
            else:
                sum_m +=1
                istart = im

        else:
            if mask_tmp[im]:
                sum_m += 1
            else:
                # This is the end of the group of constant dTdt,
                # so count the total number of points in group,
                # and remove whole group if total is larger than
                # ndays
                iend = im
                if sum_m >= ndays:
                    mask_line[istart:iend] = True

                sum_m = 0 # Put back sum to zero

    Twater_out = Twater_in.copy()
    Twater_out[mask_line] = np.nan

    return Twater_out, mask_line



def fill_gaps(Twater_in, ndays = 7, fill_type = 'linear'):

    # mask_tmp is true if there is no data (i.e. Tw is nan):
    mask_tmp = np.isnan(Twater_in)

    mask_gap = mask_tmp.copy()
    mask_gap[:] = False

    Twater_out = Twater_in.copy()

    for im in range(1,mask_gap.size):

        if (im == 1) | (~mask_tmp[im-1]):
            # start new group
            sum_m = 0
            if ~mask_tmp[im]:
                sum_m = 0
            else:
                sum_m +=1
                istart = im

        else:
            if mask_tmp[im]:
                sum_m += 1
            else:
                # This is the end of the group of constant dTdt,
                # so count the total number of points in group,
                # and remove whole group if total is larger than
                # ndays
                iend = im
                if sum_m < ndays:
                    mask_gap[istart:iend] = True

                    if fill_type == 'linear':
                        # Fill small gap with linear interpolation
                        slope = (Twater_out[iend]-Twater_out[istart-1])/(sum_m+1)
                        Twater_out[istart:iend] = Twater_out[istart-1] + slope*(np.arange(sum_m)+1)

                    else:
                        print('Problem! ''fill_type'' not defined...')

                sum_m = 0 # Put back sum to zero


    return Twater_out, mask_gap


#%%
plot = True
# plot = False

fill = True
fill_type = 'linear'

# save = True
save = False

# years = [1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005]
# years = [2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020]

years = [1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

# water_name_list = ['DesBaillets']
water_name_list = ['Longueuil_updated']
# water_name_list = ['Atwater']
# water_name_list = ['Candiac']
# water_name_list = ['Atwater','Longueuil','DesBaillets','Candiac']
# water_name_list = ['Atwater','Longueuil','Candiac']
station_type = 'cities'

# water_name_list = ['Lasalle']
# water_name_list = ['LaPrairie']
#water_name_list = ['Lasalle','LaPrairie']
# station_type = 'ECCC'

# water_name_list = ['StLambert','StLouisBridge']
# station_type = 'SLSMC'

fp = local_path+'slice/data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

yr_start = 0 # 1992
yr_end = len(years) -1 #2020

istart = np.where(time == (dt.date(years[yr_start],1,1)-date_ref).days)[0][0]
iend = np.where(time == (dt.date(years[yr_end],12,31)-date_ref).days)[0][0]
iend += 1
if istart < 5000: istart = 0

time_select = time[istart:iend]
years = np.array(years)

#%%
# Compute dTwater/dt and d2Twater/dt2
Twater = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt_f = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt_b = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan

for iloc,loc in enumerate(water_name_list):
    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][istart:iend,1]

    if loc == 'Longueuil':
        Twater_tmp[13574:] = np.nan # Remove all the singular records at the end
    if loc == 'Atwater':
        Twater_tmp[:11000] = np.nan # Remove years prior to 2010, since the resolution is +/- one degree and the distribution is different
    Twater[:,iloc] = Twater_tmp
    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_dTdt_f[:,iloc] = dTdt_tmp[:,0]
    Twater_dTdt_b[:,iloc] = dTdt_tmp[:,1]

    # This is the Laplacian:
    Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)



# 1) apply filter to remove steps and lines
Twater_clean = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt_clean = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt_f_clean = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_dTdt_b_clean = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
Twater_d2Tdt2_clean = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan

for iloc,loc in enumerate(water_name_list):
    Twater_mask_lines, mask_lines = mask_lines_and_steps(Twater[:,iloc], Twater_d2Tdt2[:,iloc])

    dTdt_mask_lines = Twater_dTdt[:,iloc].copy()
    dTdt_f_mask_lines = Twater_dTdt_f[:,iloc].copy()
    dTdt_b_mask_lines = Twater_dTdt_b[:,iloc].copy()
    d2Tdt2_mask_lines = Twater_d2Tdt2[:,iloc].copy()

    dTdt_mask_lines[mask_lines] = np.nan
    dTdt_f_mask_lines[mask_lines] = np.nan
    dTdt_b_mask_lines[mask_lines] = np.nan
    d2Tdt2_mask_lines[mask_lines] = np.nan

    if plot:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
        plt.title(loc)
        ax.plot(Twater[:,iloc], color=plt.get_cmap('tab20')(iloc*2),label=water_name_list[iloc])
        ax.plot(Twater_mask_lines, '.-', markersize=3,  color=plt.get_cmap('tab20')(iloc*2+1))

    Twater_clean[:,iloc] = Twater_mask_lines
    Twater_dTdt_clean[:,iloc] = dTdt_mask_lines
    Twater_dTdt_f_clean[:,iloc] = dTdt_f_mask_lines
    Twater_dTdt_b_clean[:,iloc] = dTdt_b_mask_lines
    Twater_d2Tdt2_clean[:,iloc] = d2Tdt2_mask_lines



# 2) Remove points with d2T/dt2 larger than
#    mean(d2Tdt2) +/- t0*std(d2Tdt2)
#    where mean() and std() are taken from whole dataset
t0 = 3.0
thresh_pos_d2Tdt2 = np.zeros(Twater_d2Tdt2_clean.shape)*np.nan
thresh_neg_d2Tdt2 = np.zeros(Twater_d2Tdt2_clean.shape)*np.nan
thresh_pos_dTdt = np.zeros(Twater_dTdt_clean.shape)*np.nan
thresh_neg_dTdt = np.zeros(Twater_dTdt_clean.shape)*np.nan

for iloc,loc in enumerate(water_name_list):
    Tw_mask_d2Tdt2     = Twater_clean[:,iloc].copy()
    dTdt_mask_d2Tdt2   = Twater_dTdt_clean[:,iloc].copy()
    dTdt_f_mask_d2Tdt2 = Twater_dTdt_f_clean[:,iloc].copy()
    dTdt_b_mask_d2Tdt2 = Twater_dTdt_b_clean[:,iloc].copy()
    d2Tdt2_mask_d2Tdt2 = Twater_d2Tdt2_clean[:,iloc].copy()

    mask_d2Tdt2_2 = np.abs(d2Tdt2_mask_d2Tdt2-np.nanmean(d2Tdt2_mask_d2Tdt2)) > t0*np.nanstd(d2Tdt2_mask_d2Tdt2)
    thresh_pos_d2Tdt2[:,iloc] = np.nanmean(d2Tdt2_mask_d2Tdt2) + t0*np.nanstd(d2Tdt2_mask_d2Tdt2)
    thresh_neg_d2Tdt2[:,iloc] = np.nanmean(d2Tdt2_mask_d2Tdt2) - t0*np.nanstd(d2Tdt2_mask_d2Tdt2)

    Tw_mask_d2Tdt2[mask_d2Tdt2_2] = np.nan
    dTdt_mask_d2Tdt2[mask_d2Tdt2_2] = np.nan
    dTdt_f_mask_d2Tdt2[mask_d2Tdt2_2] = np.nan
    dTdt_b_mask_d2Tdt2[mask_d2Tdt2_2] = np.nan
    d2Tdt2_mask_d2Tdt2[mask_d2Tdt2_2] = np.nan

    Twater_clean[:,iloc] = Tw_mask_d2Tdt2
    Twater_d2Tdt2_clean[:,iloc] = d2Tdt2_mask_d2Tdt2
    Twater_dTdt_clean[:,iloc] = dTdt_mask_d2Tdt2
    Twater_dTdt_f_clean[:,iloc] = dTdt_f_mask_d2Tdt2
    Twater_dTdt_b_clean[:,iloc] = dTdt_b_mask_d2Tdt2



# 3) Now filter points if T is larger than climatology:
#    i.e. abs(Tw -Tw_mean) > t0*Tw_std

# First, compute climatology:
clim_mean_Tw = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan
clim_std_Tw = np.zeros((len(time[istart:iend]),len(water_name_list)))*np.nan

for iloc,loc in enumerate(water_name_list):
    clim_mean_Tw[:,iloc], clim_std_Tw[:,iloc] = rolling_clim(Twater_clean[:,iloc], years[yr_start:yr_end+1], time[istart:iend])

# Then apply filter:
t0 = 3.5
for iloc,loc in enumerate(water_name_list):
    Tw_mask_climT     = Twater_clean[:,iloc].copy()
    dTdt_mask_climT   = Twater_dTdt_clean[:,iloc].copy()
    dTdt_f_mask_climT = Twater_dTdt_f_clean[:,iloc].copy()
    dTdt_b_mask_climT = Twater_dTdt_b_clean[:,iloc].copy()
    d2Tdt2_mask_climT = Twater_d2Tdt2_clean[:,iloc].copy()

    mask_Tw_clim = np.abs(Tw_mask_climT-clim_mean_Tw[:,iloc]) > t0*clim_std_Tw[:,iloc]

    Tw_mask_climT[mask_Tw_clim] = np.nan
    dTdt_mask_climT[mask_Tw_clim] = np.nan
    dTdt_f_mask_climT[mask_Tw_clim] = np.nan
    dTdt_b_mask_climT[mask_Tw_clim] = np.nan
    d2Tdt2_mask_climT[mask_Tw_clim] = np.nan

    Twater_clean[:,iloc] = Tw_mask_climT
    Twater_d2Tdt2_clean[:,iloc] = d2Tdt2_mask_climT
    Twater_dTdt_clean[:,iloc] = dTdt_mask_climT
    Twater_dTdt_f_clean[:,iloc] = dTdt_f_mask_climT
    Twater_dTdt_b_clean[:,iloc] = dTdt_b_mask_climT



# 4) Fill gaps shorter than 7 days
if fill:
    for iloc,loc in enumerate(water_name_list):

        T_line, mask_gaps = fill_gaps(Twater_clean[:,iloc],fill_type=fill_type)

        # fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(6,5),sharex=True)
        # ax[0].plot(T_line, color=plt.get_cmap('tab20')(iloc*2))
        # ax[0].plot(Twater_clean[:,iloc], color=plt.get_cmap('tab20')(iloc*2+1))

        # loc_water_loc = water_name_list[iloc]
        # water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
        # Twater = water_loc_data['Twater'][:,1]
        # ax[1].plot(Twater,color=plt.get_cmap('tab20')(iloc*2))

        Twater_clean[:,iloc] = T_line



if plot:
    for iyear,year in enumerate(years[yr_start:yr_end+1]):
        fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
        plt.title(years[yr_start:yr_end+1][iyear])

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365+calendar.isleap(year)

        i0_clean = np.where(time_select==date)[0][0]
        i1_clean = i0_clean+365+calendar.isleap(year)

        for iloc,loc in enumerate(water_name_list):

            ax[0].plot(Twater[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2),label=water_name_list[iloc])
            ax[0].plot(Twater_clean[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2+1))

            ax[1].plot(Twater_d2Tdt2[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2),label=water_name_list[iloc])
            ax[1].plot(Twater_d2Tdt2_clean[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2+1))
            ax[1].plot(thresh_pos_d2Tdt2[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2+1))
            ax[1].plot(thresh_neg_d2Tdt2[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2+1))

            ax[2].plot(Twater_dTdt[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2),label=water_name_list[iloc])
            ax[2].plot(Twater_dTdt_clean[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2+1))
            ax[2].plot(thresh_pos_dTdt[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2+1))
            ax[2].plot(thresh_neg_dTdt[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2+1))

        ax[0].legend(bbox_to_anchor=(0.3, 0.91))
        ax[0].set_ylabel('T$_w$')
        ax[0].set_xlabel('DOY')

#%%

for iyear,year in enumerate(years[25:26]):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
    # plt.title('2017')

    date=(dt.date(year,1,1)-date_ref).days
    i0 = np.where(time==date)[0][0]
    i1 = i0+365+calendar.isleap(year)

    i0_clean = np.where(time_select==date)[0][0]
    i1_clean = i0_clean+365+calendar.isleap(year)

    for iloc,loc in enumerate(water_name_list):
        Twater_plot = Twater[:,iloc].copy()
        Twater_clean_plot = Twater_clean[:,iloc].copy()
        if loc == 'Candiac':
            Twater_plot = Twater[:,iloc].copy()-0.8
            Twater_clean_plot = Twater_clean[:,iloc].copy()-0.8
        if (loc == 'Atwater'):
            Twater_plot = Twater[:,iloc].copy()
            Twater_plot[0:12490] = Twater_plot[0:12490]-0.7
            Twater_clean_plot = Twater_clean[:,iloc].copy()
            Twater_clean_plot[0:12490] = Twater_clean_plot[0:12490]-0.7

    ax.legend(bbox_to_anchor=(0.3, 0.91))
    ax.set_ylabel('T$_w$')
    ax.set_xlabel('DOY')

plt.subplots_adjust(bottom=0.2)


#%%
for iyear,year in enumerate(years[13:14]):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
    # plt.title('2005')

    date=(dt.date(year,1,1)-date_ref).days
    i0 = np.where(time==date)[0][0]
    i1 = i0+365+calendar.isleap(year)

    i0_clean = np.where(time_select==date)[0][0]
    i1_clean = i0_clean+365+calendar.isleap(year)

    for iloc,loc in enumerate(water_name_list):

        ax.plot(Twater[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2),label=water_name_list[iloc]+' - Original')
        ax.plot(Twater_clean[i0_clean:i1_clean,iloc], color=plt.get_cmap('tab20')(iloc*2+1),label=water_name_list[iloc]+' - Filtered')

    ax.legend(bbox_to_anchor=(0.45, 0.71))
    ax.set_ylabel('T$_w$')
    ax.set_xlabel('DOY')

plt.subplots_adjust(bottom=0.2)
#%%
for iyear,year in enumerate(years):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
    # plt.title('2005')

    date=(dt.date(year,1,1)-date_ref).days
    i0 = np.where(time==date)[0][0]
    i1 = i0+365+calendar.isleap(year)

    i0_clean = np.where(time_select==date)[0][0]
    i1_clean = i0_clean+365+calendar.isleap(year)

    for iloc,loc in enumerate(water_name_list):
        Twater_plot = Twater[:,iloc].copy()
        Twater_clean_plot = Twater_clean[:,iloc].copy()
        if loc == 'Candiac':
            Twater_plot = Twater[:,iloc].copy()-0.8
            Twater_clean_plot = Twater_clean[:,iloc].copy()-0.8
        if (loc == 'Atwater'):
            Twater_plot = Twater[:,iloc].copy()
            Twater_plot[0:12490] = Twater_plot[0:12490]-0.7
            Twater_clean_plot = Twater_clean[:,iloc].copy()
            Twater_clean_plot[0:12490] = Twater_clean_plot[0:12490]-0.7

        ax.plot(Twater_plot[i0_clean:i1_clean], color=plt.get_cmap('tab20')(iloc*2),label=water_name_list[iloc]+' - Original')
        ax.plot(Twater_clean_plot[i0_clean:i1_clean], color=plt.get_cmap('tab20')(iloc*2+1),label=water_name_list[iloc]+' - Filtered')

    ax.legend(bbox_to_anchor=(0.45, 0.71))
    ax.set_ylabel('T$_w$')
    ax.set_xlabel('DOY')

plt.subplots_adjust(bottom=0.2)

#%%
# 5) Identify winter season and bring all winters back to zero C.



# 6) Filter negative temperatures




# 7) save cleaned data set in format [ ndays, [time,Tw] ]:
# if save:
#     for iloc,loc in enumerate(water_name_list):
#         Twater = np.zeros((Twater_clean.shape[0],2))

#         Twater[:,0] = time
#         Twater[:,1] = Twater_clean[:,iloc]

#         np.savez(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc+'_cleaned'+fill*'_filled'+'.npz',
#                   Twater=Twater,date_ref=date_ref)



#%%

Twater_all = np.zeros((Twater_clean.shape[0]*len(water_name_list),4))*np.nan

# 0: Twater
# 1: plant
# 2: year
# 3: season

save_loc = ''
for iloc in range(len(water_name_list)):
    loc = water_name_list[iloc]
    save_loc += loc
    year_tmp = np.zeros((Twater_clean.shape[0]))*np.nan
    season_tmp = np.zeros((Twater_clean.shape[0]))*np.nan

    for it in range(Twater_clean.shape[0]):
        date_it = date_ref+dt.timedelta(days=int(time_select[it]))
        year_tmp[it] = int(date_it.year)

        if (((date_it - dt.date(int(date_it.year),3,1)).days > 0) &
            ((date_it - dt.date(int(date_it.year),6,1)).days <= 0) ):
                season_tmp[it] = 0 # Spring: March to May

        if (((date_it - dt.date(int(date_it.year),6,1)).days > 0) &
            ((date_it - dt.date(int(date_it.year),9,1)).days <= 0) ):
                season_tmp[it] = 1 # Summer: June to August

        if (((date_it - dt.date(int(date_it.year),9,1)).days > 0) &
            ((date_it - dt.date(int(date_it.year),12,1)).days <= 0) ):
                season_tmp[it] = 2 # Fall: September to november

        if (((date_it - dt.date(int(date_it.year),12,1)).days > 0)):
              season_tmp[it] = 3 # Winter: December 1st to December 31st

        if (((date_it - dt.date(int(date_it.year),3,1)).days <= 0)):
              season_tmp[it] = 3 # Winter: January 1st to March


    Twater_tmp = Twater_clean[:,iloc].copy()

    # APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST
    if loc == 'Candiac':
        Twater_tmp = Twater_tmp-0.8
    if (loc == 'Atwater'):
        Twater_tmp[0:12490] = Twater_tmp[0:12490]-0.7
    if (loc == 'Longueuil_updated'):
        Twater_tmp[14329:] = Twater_tmp[14329:]- 0.78


    # Remove data prior to end of 2010 to compare
    # only the same period (i.e. 2010-2020)
    # Twater_tmp[:11220] = np.nan

    # Remove data prior to end of 2004 to compare
    # only the same period (i.e. 2004-2020)
    # Twater_tmp[:8800] = np.nan

    # Twater_tmp[:12500] = np.nan
    # Twater_tmp[13600:] = np.nan
    # Twater_tmp[:12670] = np.nan
    # Twater_tmp[14676:] = np.nan

    Twater_all[iloc*Twater_clean.shape[0]:Twater_clean.shape[0]*(iloc+1),0] = Twater_tmp
    Twater_all[iloc*Twater_clean.shape[0]:Twater_clean.shape[0]*(iloc+1),1] = iloc
    Twater_all[iloc*Twater_clean.shape[0]:Twater_clean.shape[0]*(iloc+1),2] = year_tmp
    Twater_all[iloc*Twater_clean.shape[0]:Twater_clean.shape[0]*(iloc+1),3] = season_tmp

# Twater_all = Twater_all[~np.isnan(Twater_all[:,0])]

if save:
    np.savez(fp+'Twater_'+station_type+'/Twater_'+station_type+'_all_'+save_loc+'_cleaned'+fill*'_filled',
            Twater_all=Twater_all,date_ref=date_ref)














