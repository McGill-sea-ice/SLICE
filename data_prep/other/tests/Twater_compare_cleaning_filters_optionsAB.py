#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:58:03 2021

@author: Amelie
"""

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
    # var_in = Twater[:,icity]

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


#%%
plot = True

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
          2018,2019,2020]

water_cities_name_list = ['Candiac','Longueuil','Atwater','DesBaillets']

loc_weather = 'MontrealDorvalMontrealPETMontrealMcTavishmerged'

fp = '../../data/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
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
Twater = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_dTdt = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_dTdt_f = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_dTdt_b = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_city_data['Twater'][istart:iend,1]
    Twater[:,icity] = Twater_tmp
    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,icity] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_dTdt_f[:,icity] = dTdt_tmp[:,0]
    Twater_dTdt_b[:,icity] = dTdt_tmp[:,1]

    Twater_d2Tdt2[:,icity] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)


#%%
# 1) apply filter to remove steps and lines
Twater_clean = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_dTdt_clean = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_dTdt_f_clean = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_dTdt_b_clean = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_d2Tdt2_clean = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    Tw     = Twater[:,icity]
    dTdt   = Twater_dTdt[:,icity]
    dTdt_f = Twater_dTdt_f[:,icity]
    dTdt_b = Twater_dTdt_b[:,icity]
    d2Tdt2 = Twater_d2Tdt2[:,icity]

    Twater_mask_lines, mask_lines = mask_lines_and_steps(Tw, d2Tdt2)

    dTdt_mask_lines = dTdt.copy()
    dTdt_f_mask_lines = dTdt_f.copy()
    dTdt_b_mask_lines = dTdt_b.copy()
    d2Tdt2_mask_lines = d2Tdt2.copy()
    dTdt_mask_lines[mask_lines] = np.nan
    dTdt_f_mask_lines[mask_lines] = np.nan
    dTdt_b_mask_lines[mask_lines] = np.nan
    d2Tdt2_mask_lines[mask_lines] = np.nan

    if plot:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
        plt.title(city)
        ax.plot(Tw, color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
        ax.plot(Twater_mask_lines, '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

    Twater_clean[:,icity] = Twater_mask_lines
    Twater_dTdt_clean[:,icity] = dTdt_mask_lines
    Twater_dTdt_f_clean[:,icity] = dTdt_f_mask_lines
    Twater_dTdt_b_clean[:,icity] = dTdt_b_mask_lines
    Twater_d2Tdt2_clean[:,icity] = d2Tdt2_mask_lines


# copy data now to compare with other filter after
Twater_clean_line = Twater_clean.copy()
Twater_dTdt_clean_line = Twater_dTdt_clean.copy()
Twater_dTdt_f_clean_line = Twater_dTdt_f_clean.copy()
Twater_dTdt_b_clean_line = Twater_dTdt_b_clean.copy()
Twater_d2Tdt2_clean_line = Twater_d2Tdt2_clean.copy()


#%%
# 2) Option A: Remove points with very large d2T/dt2 or dT/dt
#   that otherwise pollute the climatology for dT/dt and d2T/dt2

d2Tdt2_plot0 = Twater_d2Tdt2_clean.copy()
dTdt_plot0 = Twater_dTdt_clean.copy()

t0 = 5.0
for icity,city in enumerate(water_cities_name_list):
    Tw     = Twater_clean[:,icity]
    dTdt   = Twater_dTdt_clean[:,icity]
    dTdt_f = Twater_dTdt_f_clean[:,icity]
    dTdt_b = Twater_dTdt_b_clean[:,icity]
    d2Tdt2 = Twater_d2Tdt2_clean[:,icity]

    mask_dTdt = np.abs(dTdt-np.nanmean(dTdt)) > t0*np.nanstd(dTdt)
    mask_d2Tdt2 = np.abs(d2Tdt2-np.nanmean(d2Tdt2)) > t0*np.nanstd(d2Tdt2)

    Tw[mask_dTdt | mask_d2Tdt2] = np.nan
    dTdt[mask_dTdt | mask_d2Tdt2] = np.nan
    dTdt_f[mask_dTdt | mask_d2Tdt2] = np.nan
    dTdt_b[mask_dTdt | mask_d2Tdt2] = np.nan
    d2Tdt2[mask_dTdt | mask_d2Tdt2] = np.nan

    # d2Tdt2_plot0[:,icity] = d2Tdt2


# Then compute climatology for T, dTdt, and d2T/dt2
clim_mean_Tw = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
clim_mean_dTdt = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
clim_mean_dTdt_f = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
clim_mean_dTdt_b = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
clim_mean_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

clim_std_Tw = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
clim_std_dTdt = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
clim_std_dTdt_f = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
clim_std_dTdt_b = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
clim_std_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    clim_mean_Tw[:,icity], clim_std_Tw[:,icity] = rolling_clim(Twater_clean[:,icity], years[yr_start:yr_end+1], time[istart:iend])
    clim_mean_dTdt[:,icity], clim_std_dTdt[:,icity] = rolling_clim(Twater_dTdt_clean[:,icity], years[yr_start:yr_end+1], time[istart:iend])
    clim_mean_dTdt_f[:,icity], clim_std_dTdt_f[:,icity] = rolling_clim(Twater_dTdt_f_clean[:,icity], years[yr_start:yr_end+1], time[istart:iend])
    clim_mean_dTdt_b[:,icity], clim_std_dTdt_b[:,icity] = rolling_clim(Twater_dTdt_b_clean[:,icity], years[yr_start:yr_end+1], time[istart:iend])
    clim_mean_d2Tdt2[:,icity], clim_std_d2Tdt2[:,icity] = rolling_clim(Twater_d2Tdt2_clean[:,icity], years[yr_start:yr_end+1], time[istart:iend])


# and remove points for which:
# abs(dTdt - clim_mean_dTdt)> t0*clim_std_dTdt
# abs(d2Tdt2 - clim_mean_d2Tdt2)> t0*clim_std_d2Tdt2
t0 = 3.5

thresh_pos_d2Tdt2_plot = np.zeros(Twater_d2Tdt2_clean.shape)*np.nan
thresh_neg_d2Tdt2_plot = np.zeros(Twater_d2Tdt2_clean.shape)*np.nan
thresh_pos_dTdt_plot = np.zeros(Twater_dTdt_clean.shape)*np.nan
thresh_neg_dTdt_plot = np.zeros(Twater_dTdt_clean.shape)*np.nan
d2Tdt2_plot = Twater_d2Tdt2_clean.copy()
dTdt_plot = Twater_dTdt_clean.copy()

for icity,city in enumerate(water_cities_name_list):
    Tw     = Twater_clean[:,icity]
    dTdt   = Twater_dTdt_clean[:,icity]
    dTdt_f = Twater_dTdt_f_clean[:,icity]
    dTdt_b = Twater_dTdt_b_clean[:,icity]
    d2Tdt2 = Twater_d2Tdt2_clean[:,icity]

    mask_dTdt_clim = np.abs(dTdt-clim_mean_dTdt[:,icity]) > t0*clim_std_dTdt[:,icity]
    mask_d2Tdt2_clim = np.abs(d2Tdt2-clim_mean_d2Tdt2[:,icity]) > t0*clim_std_d2Tdt2[:,icity]

    thresh_pos_d2Tdt2_plot[:,icity] = clim_mean_d2Tdt2[:,icity] + t0*clim_std_d2Tdt2[:,icity]
    thresh_neg_d2Tdt2_plot[:,icity] = clim_mean_d2Tdt2[:,icity] - t0*clim_std_d2Tdt2[:,icity]
    thresh_pos_dTdt_plot[:,icity] = clim_mean_dTdt[:,icity] + t0*clim_std_dTdt[:,icity]
    thresh_neg_dTdt_plot[:,icity] = clim_mean_dTdt[:,icity] - t0*clim_std_dTdt[:,icity]

    Tw[mask_dTdt_clim | mask_d2Tdt2_clim] = np.nan
    dTdt[mask_dTdt_clim | mask_d2Tdt2_clim] = np.nan
    dTdt_f[mask_dTdt_clim | mask_d2Tdt2_clim] = np.nan
    dTdt_b[mask_dTdt_clim | mask_d2Tdt2_clim] = np.nan
    d2Tdt2[mask_dTdt_clim | mask_d2Tdt2_clim] = np.nan

    d2Tdt2_plot[:,icity] = d2Tdt2
    dTdt_plot[:,icity] = dTdt

# 2) Option B, only use 3*std(d2Tdt2) from whole dataset
#    (Note: I tried using also 3*std(dTdt) from whole
#     dataset but the best choice seems to be between this
#     version and Option A).
t0 = 3.0

thresh_pos_d2Tdt2_plot_2 = np.zeros(Twater_d2Tdt2_clean_line.shape)*np.nan
thresh_neg_d2Tdt2_plot_2 = np.zeros(Twater_d2Tdt2_clean_line.shape)*np.nan
thresh_pos_dTdt_plot_2 = np.zeros(Twater_dTdt_clean_line.shape)*np.nan
thresh_neg_dTdt_plot_2 = np.zeros(Twater_dTdt_clean_line.shape)*np.nan

Twater_clean2 = np.zeros(Twater_clean_line.shape)
d2Tdt2_plot_2 = np.zeros(Twater_d2Tdt2_clean_line.shape)
dTdt_plot_2 = np.zeros(Twater_dTdt_clean_line.shape)

for icity,city in enumerate(water_cities_name_list):
    Tw     = Twater_clean_line[:,icity].copy()
    dTdt   = Twater_dTdt_clean_line[:,icity].copy()
    dTdt_f = Twater_dTdt_f_clean_line[:,icity].copy()
    dTdt_b = Twater_dTdt_b_clean_line[:,icity].copy()
    d2Tdt2 = Twater_d2Tdt2_clean_line[:,icity].copy()

    mask_d2Tdt2_2 = np.abs(d2Tdt2-np.nanmean(d2Tdt2)) > t0*np.nanstd(d2Tdt2)
    thresh_pos_d2Tdt2_plot_2[:,icity] = np.nanmean(d2Tdt2) + t0*np.nanstd(d2Tdt2)
    thresh_neg_d2Tdt2_plot_2[:,icity] = np.nanmean(d2Tdt2) - t0*np.nanstd(d2Tdt2)

    Tw[mask_d2Tdt2_2] = np.nan
    dTdt[mask_d2Tdt2_2] = np.nan
    dTdt_f[mask_d2Tdt2_2] = np.nan
    dTdt_b[mask_d2Tdt2_2] = np.nan
    d2Tdt2[mask_d2Tdt2_2] = np.nan

    Twater_clean2[:,icity] = Tw
    d2Tdt2_plot_2[:,icity] = d2Tdt2
    dTdt_plot_2[:,icity] = dTdt



#%%
if plot:
    for iyear,year in enumerate(years[yr_start:yr_end+1]):
        fig,ax = plt.subplots(nrows=3,ncols=2,figsize=(12,10),sharex=True)
        plt.title(years[yr_start:yr_end+1][iyear])

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365+calendar.isleap(year)

        i0_clean = np.where(time_select==date)[0][0]
        i1_clean = i0_clean+365+calendar.isleap(year)

        for icity,city in enumerate(water_cities_name_list):

            ax[0,0].plot(Twater[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[0,0].plot(Twater_clean[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))

            ax[0,1].plot(Twater[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[0,1].plot(Twater_clean2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))

            ax[1,0].plot(Twater_d2Tdt2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[1,0].plot(d2Tdt2_plot[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))
            ax[1,0].plot(thresh_pos_d2Tdt2_plot[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))
            ax[1,0].plot(thresh_neg_d2Tdt2_plot[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))

            ax[1,1].plot(Twater_d2Tdt2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[1,1].plot(d2Tdt2_plot_2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))
            ax[1,1].plot(thresh_pos_d2Tdt2_plot_2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))
            ax[1,1].plot(thresh_neg_d2Tdt2_plot_2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))

            ax[2,0].plot(Twater_dTdt[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[2,0].plot(dTdt_plot[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))
            ax[2,0].plot(thresh_pos_dTdt_plot[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))
            ax[2,0].plot(thresh_neg_dTdt_plot[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))

            ax[2,1].plot(Twater_dTdt[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[2,1].plot(dTdt_plot_2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))
            ax[2,1].plot(thresh_pos_dTdt_plot_2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))
            ax[2,1].plot(thresh_neg_dTdt_plot_2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(icity*2+1))



        ax[0,0].legend(bbox_to_anchor=(0.01, 1))



#%%

# if plot:
#     for icity,city in enumerate(water_cities_name_list):
#         fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
#         plt.title(city)
#         ax[0].plot(clim_mean_Tw[:,icity], color='black')
#         ax[0].plot(clim_mean_Tw[:,icity]+3.5*clim_std_Tw[:,icity], '-', color='gray')
#         ax[0].plot(clim_mean_Tw[:,icity]-3.5*clim_std_Tw[:,icity], '-', color='gray')
#         ax[0].plot(clim_mean_Tw[:,icity]+4*clim_std_Tw[:,icity], '-', color='red')
#         ax[0].plot(clim_mean_Tw[:,icity]-4*clim_std_Tw[:,icity], '-', color='red')
#         ax[0].plot(clim_mean_Tw[:,icity]+4.5*clim_std_Tw[:,icity], '-', color='gray')
#         ax[0].plot(clim_mean_Tw[:,icity]-4.5*clim_std_Tw[:,icity], '-', color='gray')
#         ax[0].plot(Twater[:,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#         ax[0].plot(Twater_clean[:,icity], color=plt.get_cmap('tab20')(icity*2+1),label=water_cities_name_list[icity])

#         ax[1].plot(clim_mean_dTdt[:,icity], color='black')
#         ax[1].plot(clim_mean_dTdt[:,icity]+3.5*clim_std_dTdt[:,icity], '-', color='gray')
#         ax[1].plot(clim_mean_dTdt[:,icity]-3.5*clim_std_dTdt[:,icity], '-', color='gray')
#         ax[1].plot(Twater_dTdt[:,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#         ax[1].plot(Twater_dTdt_clean[:,icity], color=plt.get_cmap('tab20')(icity*2+1),label=water_cities_name_list[icity])
#         ax[1].plot(np.ones(Twater_dTdt[:,icity].shape)*(np.nanmean(Twater_dTdt_clean[:,icity])+3*np.nanstd(Twater_dTdt_clean[:,icity])),'gray')
#         ax[1].plot(np.ones(Twater_dTdt[:,icity].shape)*(np.nanmean(Twater_dTdt_clean[:,icity])-3*np.nanstd(Twater_dTdt_clean[:,icity])),'gray')
#         ax[1].plot(np.ones(Twater_dTdt[:,icity].shape)*(np.nanmean(Twater_dTdt_clean[:,icity])+4*np.nanstd(Twater_dTdt_clean[:,icity])),'red')
#         ax[1].plot(np.ones(Twater_dTdt[:,icity].shape)*(np.nanmean(Twater_dTdt_clean[:,icity])-4*np.nanstd(Twater_dTdt_clean[:,icity])),'red')

#         ax[2].plot(clim_mean_d2Tdt2[:,icity], color='black')
#         ax[2].plot(clim_mean_d2Tdt2[:,icity]+3.5*clim_std_d2Tdt2[:,icity], '-', color='gray')
#         ax[2].plot(clim_mean_d2Tdt2[:,icity]-3.5*clim_std_d2Tdt2[:,icity], '-', color='gray')
#         ax[2].plot(Twater_d2Tdt2[:,icity], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#         ax[2].plot(Twater_d2Tdt2_clean[:,icity], color=plt.get_cmap('tab20')(icity*2+1),label=water_cities_name_list[icity])
#         ax[2].plot(np.ones(Twater_d2Tdt2[:,icity].shape)*(np.nanmean(Twater_d2Tdt2_clean[:,icity])+3*np.nanstd(Twater_d2Tdt2_clean[:,icity])),'gray')
#         ax[2].plot(np.ones(Twater_d2Tdt2[:,icity].shape)*(np.nanmean(Twater_d2Tdt2_clean[:,icity])-3*np.nanstd(Twater_d2Tdt2_clean[:,icity])),'gray')
#         ax[2].plot(np.ones(Twater_d2Tdt2[:,icity].shape)*(np.nanmean(Twater_d2Tdt2_clean[:,icity])+4*np.nanstd(Twater_d2Tdt2_clean[:,icity])),'red')
#         ax[2].plot(np.ones(Twater_d2Tdt2[:,icity].shape)*(np.nanmean(Twater_d2Tdt2_clean[:,icity])-4*np.nanstd(Twater_d2Tdt2_clean[:,icity])),'red')


#%%
# # and remove points with abs(Tw -Tw_mean) > t0*Tw_std:
# # This eliminates the very large outliers that otherwise pollute the climatology for dT/dt and d2T/dt2
# t0 = 4.0
# for icity,city in enumerate(water_cities_name_list):
#     Tw     = Twater_clean[:,icity]
#     dTdt   = Twater_dTdt_clean[:,icity]
#     dTdt_f = Twater_dTdt_f_clean[:,icity]
#     dTdt_b = Twater_dTdt_b_clean[:,icity]
#     d2Tdt2 = Twater_d2Tdt2_clean[:,icity]

#     mask_Tw_clim = np.abs(Tw-clim_mean_Tw[:,icity]) > t0*clim_std_Tw[:,icity]

#     Tw[mask_Tw_clim] = np.nan
#     dTdt[mask_Tw_clim] = np.nan
#     dTdt_f[mask_Tw_clim] = np.nan
#     dTdt_b[mask_Tw_clim] = np.nan
#     d2Tdt2[mask_Tw_clim] = np.nan

#%%



