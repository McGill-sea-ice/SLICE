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
water_name_list = ['Kingston', 'Iroquois','Cornwall','StLouisBridge','StLambert']
# years = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

# water_name_list = ['StLambert']
# weather_name_list = ['MontrealDorval']
years = [2010]

date_ref = dt.date(1900,1,1)
fp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'


show_series = True
# mean_opt = 'centered'
# N = 5
# lag_list = [0,2,4,6,8,10,12,20,30,40]
lag_list = [0]
# river_vec = np.zeros((len(water_name_list),len(lag_list)))*np.nan

# # for i in range(len(water_name_list)):
# for i in range(1):
#     loc_upstream = 'Kingston'
#     down_stations = water_name_list[i+1:]

#     water_upstream = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_upstream+'.npz',allow_pickle='TRUE')
#     Twater_upstream = water_upstream['Twater']

#     river_vec = np.zeros((len(water_name_list),len(lag_list)))*np.nan

#     for j in range(len(down_stations)):
#         loc_downstream = down_stations[j]

#         water_downstream = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_downstream+'.npz',allow_pickle='TRUE')
#         Twater_downstream = water_downstream['Twater']

#         for l,lag in enumerate(lag_list):

#             R_mean=[]

#             for year in years:
#                 date=(dt.date(year,11,1)-date_ref).days
#                 i0 = np.where(Twater_upstream[:,0]==date)[0][0] - 10
#                 i1 = i0+120

#                 time_select = Twater_upstream[i0:i1,0].copy()

#                 Tw_upstream_select = Twater_upstream[i0:i1,1].copy()
#                 Tw_downstream_select = Twater_downstream[i0+lag:i1+lag,1].copy()

#                 # Tw_upstream_select[Tw_upstream_select <= 0.2] = np.nan
#                 # Tw_downstream_select[Tw_downstream_select <= 0.2] = np.nan

#                 mask1 = ~np.isnan(Tw_upstream_select)
#                 mask2 = ~np.isnan(Tw_downstream_select)
#                 mask = mask1 & mask2

#                 x1 = Tw_upstream_select[mask]
#                 x2 = Tw_downstream_select[mask]

#                 # xi = (x2 -np.mean(x2))/np.std(x2)
#                 # xj = (x1 -np.mean(x1))/np.std(x1)

#                 xi = x2
#                 xj = x1

#                 if show_series:
#                     if (l ==9) & (i+j ==0) :
#                         fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7,4),sharex=True)
#                         ax[0].plot(np.arange(len(x1))+305,x1,'-',color=plt.get_cmap('tab20')(4))
#                         ax[1].plot(np.arange(len(x2))+305,x2,'-',color=plt.get_cmap('tab20')(0))
#                         ax[0].set_xlim([305,385])
#                         ax[0].set_ylim([-1,15])
#                         ax[1].set_ylim([-1,15])
#                         ax[1].set_xlabel('Time (DOY)')
#                         ax[1].set_ylabel('$T_w$ Downstream [C]')
#                         ax[0].set_ylabel('$T_w$ Upstream [C]')
#                         plt.suptitle('Upstream: '+loc_upstream+'\nDownstream: '+loc_downstream)
#                         ax[0].yaxis.grid(color=(0.9, 0.9, 0.9))
#                         ax[1].yaxis.grid(color=(0.9, 0.9, 0.9))

#                 R = np.corrcoef(xi,xj,rowvar=False)[0,1]
#                 R_mean.append(R)


#             R_mean = np.nanmean(R_mean)
#             river_vec[j+i,l]=R_mean
#             # print(i+j,loc_upstream,loc_downstream)
#
#
    # plt.figure()
    # R_sqr =np.flipud(np.transpose(river_vec*river_vec))
    # plt.imshow(R_sqr)
    # # plt.set_cmap('binary_r')
    # plt.set_cmap('inferno')
    # plt.colorbar()
    # # plt.clim(0.3,0.55)

    # # for i in range(len(water_name_list)):
    # #     jmax=R_sqr[:,i].argmax()
    # #     highlight_cell(i,jmax, color="black", linewidth=3)

    # # plt.xticks([0,1,2,3,4],['Kingston','Iroquois','Cornwall','St-Louis','St-Lambert'],rotation=25,fontsize=8)
    # # plt.xlabel('Downstream '+r'$\rightarrow$')

    # plt.title('R$^{2}$ ($T_{water}$,$T_{air}$)')



# for i in range(len(water_name_list)):
for i in range(1):
    loc_upstream = 'StLouisBridge'
    water_upstream = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_upstream+'.npz',allow_pickle='TRUE')
    Twater_upstream = water_upstream['Twater']

    # river_vec = np.zeros((len(water_name_list),len(lag_list)))*np.nan

    loc_downstream = water_name_list[i+4]
    water_downstream = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_downstream+'.npz',allow_pickle='TRUE')
    Twater_downstream = water_downstream['Twater']


    R_mean=[]

    for year in years:
        date=(dt.date(year,11,1)-date_ref).days
        i0 = np.where(Twater_upstream[:,0]==date)[0][0] - 10
        i1 = i0+120

        time_select = Twater_upstream[i0:i1,0].copy()

        Tw_upstream_select = Twater_upstream[i0:i1,1].copy()
        Tw_downstream_select = Twater_downstream[i0:i1,1].copy()

        # Tw_upstream_select[Tw_upstream_select <= 0.2] = np.nan
        # Tw_downstream_select[Tw_downstream_select <= 0.2] = np.nan

        mask1 = ~np.isnan(Tw_upstream_select)
        mask2 = ~np.isnan(Tw_downstream_select)
        mask = mask1 & mask2

        x1 = Tw_upstream_select[mask]
        x2 = Tw_downstream_select[mask]

        xi = (x2 -np.mean(x2))/np.std(x2)
        xj = (x1 -np.mean(x1))/np.std(x1)

        # xi = x2
        # xj = x1

        if show_series:
            fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7,4),sharex=True)
            ax[0].plot(np.arange(len(x1))+305,x1,'-',color=plt.get_cmap('tab20')(4))
            ax[1].plot(np.arange(len(x2))+305,x2,'-',color=plt.get_cmap('tab20')(0))
            ax[0].set_xlim([305,385])
            ax[0].set_ylim([-1,15])
            ax[1].set_ylim([-1,15])
            ax[1].set_xlabel('Time (DOY)')
            ax[1].set_ylabel('$T_w$ Downstream [C]')
            ax[0].set_ylabel('$T_w$ Upstream [C]')
            plt.suptitle('Upstream: '+loc_upstream+'\nDownstream: '+loc_downstream)
            ax[0].yaxis.grid(color=(0.9, 0.9, 0.9))
            ax[1].yaxis.grid(color=(0.9, 0.9, 0.9))

        R = np.corrcoef(xi,xj,rowvar=False)[0,1]
        R_mean.append(R)


    R_mean_tot = np.nanmean(R_mean)
    # river_vec[j+i,l]=R_mean
    print(R_mean_tot,loc_upstream,loc_downstream)


