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

# ==========================================================================
def running_mean(x, N, mode):
    cumsum = np.cumsum(np.insert(x, 0, 0))

    xmean_tmp = (cumsum[N:] - cumsum[:-N]) / float(N)
    if mode == 'centered':
        xmean = np.insert(xmean_tmp,0,np.zeros(int((N-1)/2.))*np.nan)
        xmean = np.insert(xmean,xmean.size, np.zeros(int((N-1)/2.))*np.nan)
    if mode == 'before':
        xmean = np.insert(xmean_tmp,0,np.zeros(int((N-1)))*np.nan)

    return xmean

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

# ==========================================================================
# SLSMC_water_name_list = ['Iroquois']
# NOAA_water_name_list = ['OBGN6']
# weather_name_list = ['Ogdensburg']
# years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

SLSMC_water_name_list = ['Kingston']
NOAA_water_name_list = ['ALXN6']
weather_name_list = ['Kingston']
# years = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
# years = [2012,2013,2014,2015,2016,2017,2018]
# years = [2019]

date_ref = dt.date(1900,1,1)
fp = '../../data/processed/'

show_series = True
mean_opt = 'centered'
N = 5
lag_list = [0]
river_vec = np.zeros((len(NOAA_water_name_list),len(lag_list)))*np.nan


for i in range(len(NOAA_water_name_list)):
    loc_weather = weather_name_list[i]
    loc_water_SLSMC = SLSMC_water_name_list[i]
    loc_water_NOAA = NOAA_water_name_list[i]

    SLSMC_water_data = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_water_SLSMC+'.npz',allow_pickle='TRUE')
    Twater_SLSMC = SLSMC_water_data['Twater']

    NOAA_water_data = np.load(fp+'Twater_NOAA/Twater_NOAA_'+loc_water_NOAA+'.npz',allow_pickle='TRUE')
    Twater_NOAA = NOAA_water_data['Twater']

    NOAA_air_data = np.load(fp+'Twater_NOAA/Tair_NOAA_'+loc_water_NOAA+'.npz',allow_pickle='TRUE')
    Tair_NOAA = NOAA_air_data['Tair']

    file_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
    weather_data = file_data['weather_data']

    Tair = weather_data[:,[0,3]]
    Tair[:,1] = (Tair[:,1] - 32) * (5/9.) # From F to degrees C.
    Twater = Twater_NOAA

    Tair2 = Tair_NOAA
    Twater2 = Twater_SLSMC


    for l,lag in enumerate(lag_list):
    # for l,N in enumerate(N_list):

        R_mean=[]

        for year in years:
            date0=(dt.date(year,1,1)-date_ref).days
            i0 = np.where(Twater[:,0]==date0)[0][0]

            date1=(dt.date(year,12,31)-date_ref).days
            i1 = np.where(Twater[:,0]==date1)[0][0]+1

            time_select_Tw = Twater[i0:i1,0].copy()
            time_select_Ta = Tair[i0-lag:i1-lag,0].copy()

            date_list = [date_ref+dt.timedelta(days=time_select_Tw[d]) for d in range(len(time_select_Tw))]
            year_list = [date_list[d].year for d in range(len(time_select_Tw))]
            month_list = [date_list[d].month  for d in range(len(time_select_Tw))]
            day_list = [date_list[d].day for d in range(len(time_select_Tw))]
            DOY = [(date_list[d] - dt.date(year_list[d],1,1)).days+1 for d in range(len(time_select_Tw))]

            Tw_select = Twater[i0:i1,1].copy()
            Tw_select2 = Twater2[i0:i1,1].copy()
            Ta_select = Tair[i0-lag:i1-lag,1].copy()
            Ta_select2 = Tair2[i0-lag:i1-lag,1].copy()

            # Tw_select[Tw_select <= 2] = np.nan

            Ta_smooth = running_mean(Ta_select,N,mean_opt)
            Ta_smooth2 = running_mean(Ta_select2,N,mean_opt)

            mask_Tw = ~np.isnan(Tw_select)
            mask_Ta = ~np.isnan(Ta_smooth)
            mask_Tw2 = ~np.isnan(Tw_select2)
            mask_Ta2 = ~np.isnan(Ta_smooth2)
            mask = mask_Tw & mask_Ta & mask_Tw2 & mask_Ta2


            x1 = Ta_select[mask]
            x2 = Ta_smooth[mask]
            x3 = Tw_select[mask]
            time_select = time_select_Tw[mask]

            xi = x3
            xj = x1

            if show_series:
                if l == 0:
                    # fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7,4),sharex=True)
                    # ax[0].plot(np.arange(len(x3))+305,x3,'.-',color=plt.get_cmap('tab20')(4))
                    # ax[1].plot(np.arange(len(x1))+305,x1,'.-',color=plt.get_cmap('tab20')(1))
                    # ax[1].plot(np.arange(len(x2))+305,x2,'-',color=plt.get_cmap('tab20')(0))
                    # # ax[0].set_xlim([305,385])
                    # ax[1].set_xlabel('Time (DOY)')
                    # ax[1].set_ylabel('Air Temp. [C]')
                    # ax[0].set_ylabel('Water Temp. [C]')
                    # plt.suptitle(loc_water_NOAA+'/'+loc_weather)


                    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(7,8),sharex=True)
                    ax[0].plot(DOY,Ta_select,'.-',color=plt.get_cmap('tab20')(1))
                    ax[0].plot(DOY,Ta_smooth,'-',color=plt.get_cmap('tab20')(0))
                    ax[0].plot(DOY,np.ones(len(DOY))*np.nanmean(Ta_select),':',color=plt.get_cmap('tab20')(0))
                    ax[0].plot(DOY,Ta_select2,'.-',color=[0.9,0.9,0.9])
                    ax[0].plot(DOY,Ta_smooth2,'-',color=[0.5,0.5,0.5])

                    ax[1].plot(DOY,Tw_select,'.-',color=plt.get_cmap('tab20')(4))
                    ax[1].plot(DOY,Tw_select2,'.-',color=plt.get_cmap('tab20')(5))

                    ax[2].plot(DOY,Tw_select-Ta_select,'.-',color=plt.get_cmap('tab20')(2))

                    ax[0].set_ylabel('Air Temp. [C]')
                    ax[1].set_ylabel('Water Temp. [C]')
                    ax[2].set_ylabel(r'T$_w$-T$_a$ [C]')
                    ax[1].set_xlabel('Time (DOY)')

                    ax[0].yaxis.grid(color=(0.9, 0.9, 0.9))
                    ax[1].yaxis.grid(color=(0.9, 0.9, 0.9))
                    ax[2].yaxis.grid(color=(0.9, 0.9, 0.9))

                    plt.suptitle(loc_water_NOAA+'/'+loc_weather)

                    ax[0].set_ylim([-25,30])
                    ax[1].set_ylim([-2,30])
                    ax[2].set_ylim([-12,25])


            R = np.corrcoef(xi,xj,rowvar=False)[0,1]
            R_mean.append(R)


        R_mean = np.nanmean(R_mean)
        river_vec[i,l]=R_mean

# plt.figure()
# R_sqr =np.flipud(np.transpose(river_vec*river_vec))
# plt.imshow(R_sqr)
# # plt.set_cmap('binary_r')
# plt.set_cmap('inferno')
# plt.colorbar()
# plt.clim(0.3,0.55)

# for i in range(len(NOAA_water_name_list)):
#     jmax=R_sqr[:,i].argmax()
#     highlight_cell(i,jmax, color="black", linewidth=3)

# plt.xticks([0,1,2,3,4],['Kingston','Iroquois','Cornwall','St-Louis','St-Lambert'],rotation=25,fontsize=8)
# plt.xlabel('Downstream '+r'$\rightarrow$')

# plt.yticks([0,1,2,3,4,5],['4','4','3','2','1','0'],fontsize=8)
# plt.ylabel('Time Lag [days]')


# plt.title('R$^{2}$ ($T_{water}$,$T_{air}$)')




#%%


# np.savez('./ML_timeseries_5years_ALXN6_Kingston',
#           Ta=Ta_5yrs,Tw=Tw_5yrs,time=time_5yrs)


# np.savez('./ML_timeseries_3years_ALXN6_Kingston',
#           Ta=Ta_3yrs,Tw=Tw_3yrs,time=time_3yrs)







