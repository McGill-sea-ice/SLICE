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

from functions import running_mean, highlight_cell

#%%==========================================================================
station_type = 'SLSMC'
water_name_list = ['Kingston',  'Iroquois','Cornwall','StLouisBridge','StLambert']
weather_name_list = ['Kingston','Massena','Massena','MontrealDorval', 'MontrealDorval']
years = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

# station_type = 'SLSMC'
# water_name_list = ['StLambert']
# weather_name_list = ['MontrealDorval']
# years = [2010]

# station_type = 'cities'
# water_name_list = ['Longueuil_cleaned_filled']
# weather_name_list = ['MontrealDorvalMontrealPETMontrealMcTavishmerged']
# years = [1991,1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005,2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020]

date_ref = dt.date(1900,1,1)
fp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'

show_series = False
mean_opt = 'centered'
N = 5
lag_list = [0,1,2,3,4,5]
river_vec = np.zeros((len(water_name_list),len(lag_list)))*np.nan


for i in range(len(water_name_list)):
    loc_weather = weather_name_list[i]
    loc_water = water_name_list[i]

    # water_data = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_water+'.npz',allow_pickle='TRUE')
    # Twater = water_data['Twater']

    water_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water+'.npz',allow_pickle='TRUE')
    Twater = water_data['Twater']

    file_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
    weather_data = file_data['weather_data']

    avg_temp = weather_data[:,[0,3]]

    for l,lag in enumerate(lag_list):
    # for l,N in enumerate(N_list):

        R_mean=[]

        for year in years:
            if station_type =='SLSMC':
                date=(dt.date(year,11,1)-date_ref).days
                i0 = np.where(Twater[:,0]==date)[0][0]
                i1 = i0+150

            else:
                date=(dt.date(year,1,1)-date_ref).days
                i0 = np.where(Twater[:,0]==date)[0][0]
                i1 = i0+366

            time_select_Tw = Twater[i0:i1,0].copy()
            time_select_Ta = avg_temp[i0-lag:i1-lag,0].copy()

            Tw_select = Twater[i0:i1,1].copy()
            Ta_select = avg_temp[i0-lag:i1-lag,1].copy()

            # Tw_select[Tw_select <= 2] = np.nan


            Ta_smooth = running_mean(Ta_select,N,mean_opt)

            x1 = Ta_select.copy()
            x2 = Ta_smooth.copy()
            x3 = Tw_select.copy()

            x3[x3 <= 2] = np.nan

            x1 = x1[~np.isnan(x3)]
            x2 = x2[~np.isnan(x3)]
            x3 = x3[~np.isnan(x3)]

            x1 = x1[~np.isnan(x2)]
            x3 = x3[~np.isnan(x2)]
            x2 = x2[~np.isnan(x2)]

            xi = x3.copy()
            xj = x1.copy()

            if show_series:
                if l == 0:
                    # fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7,4),sharex=True)
                    # ax[0].plot(np.arange(len(x3))+305,x3,'-',color=plt.get_cmap('tab20')(4))
                    # ax[1].plot(np.arange(len(x1))+305,x1,'.-',color=plt.get_cmap('tab20')(1))
                    # ax[1].plot(np.arange(len(x2))+305,x2,'-',color=plt.get_cmap('tab20')(0))
                    # ax[0].set_xlim([305,385])
                    # ax[1].set_xlabel('Time (DOY)')
                    # ax[1].set_ylabel('Air Temp. [C]')
                    # ax[0].set_ylabel('Water Temp. [C]')
                    # plt.suptitle(str(year) + '\n'+loc_water+'/'+loc_weather)


                    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7,4),sharex=True)
                    ax[0].plot(time_select_Tw,Tw_select,'-',color=plt.get_cmap('tab20')(4))
                    ax[1].plot(time_select_Tw,Ta_select,'.-',color=plt.get_cmap('tab20')(1))
                    ax[1].plot(time_select_Tw,Ta_smooth,'-',color=plt.get_cmap('tab20')(0))

                    ax[1].set_xlabel('Time (DOY)')
                    ax[1].set_ylabel('Air Temp. [C]')
                    ax[0].set_ylabel('Water Temp. [C]')
                    plt.suptitle(str(year) + '\n'+loc_water+'/'+loc_weather)
                    # ax[0].set_xlim([avg_temp[i0-lag,0],avg_temp[i1-lag,0]])
                    # ax[1].set_xlim([avg_temp[i0-lag,0],avg_temp[i1-lag,0]])


            R = np.corrcoef(xi,xj,rowvar=False)[0,1]
            R_mean.append(R)

            # corr_arr = np.correlate(xi-np.mean(xi),
            #                         xj-np.mean(xj),
            #                         mode='full')
            # corr_arr = corr_arr/(np.sqrt(np.cov(xi)*np.cov(xj))*(len(xi)-1))

            # lag2 = corr_arr.argmax() - (len(xi) - 1)
            # print(R,corr_arr[corr_arr.argmax()])
            # print(lag2)

        R_mean = np.nanmean(R_mean)
        # if l== 0: print(R_mean*R_mean)
        river_vec[i,l]=R_mean

plt.figure()
R_sqr =np.flipud(np.transpose(river_vec*river_vec))
plt.imshow(R_sqr)
# plt.set_cmap('binary_r')
plt.set_cmap('inferno')
plt.colorbar()
# plt.clim(0.56,0.75)
plt.clim(0.3,0.6)

for i in range(len(water_name_list)):
    jmax=R_sqr[:,i].argmax()
    highlight_cell(i,jmax, color="black", linewidth=3)

plt.xticks([0,1,2,3,4],['Kingston','Iroquois','Cornwall','St-Louis','St-Lambert'],rotation=25,fontsize=8)
plt.xlabel('Downstream '+r'$\rightarrow$')

plt.yticks([0,1,2,3,4,5],['4','4','3','2','1','0'],fontsize=8)
plt.ylabel('Time Lag [days]')


plt.title('R$^{2}$ ($T_{water}$,$T_{air}$)')