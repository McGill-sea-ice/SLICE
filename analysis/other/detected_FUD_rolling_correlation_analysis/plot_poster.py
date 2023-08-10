#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:12:41 2021

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



nyears = 30
pc = 0.05

rc_m1, rc_p1 = r_confidence_interval(0,pc,nyears,tailed='one')
rc_m2, rc_p2 = r_confidence_interval(0,pc,nyears,tailed='two')
plot_colors = [plt.get_cmap('tab20b')(0),plt.get_cmap('tab20b')(4),plt.get_cmap('tab20b')(8),plt.get_cmap('tab20b')(12)]


iend = 0
enddate_str = 'Dec. 1st'

ivar_arr = [2,4,7]
iloc = 0
ip = 0

nrows = 1
ncols = 1
fig,ax = plt.subplots(nrows,figsize=(5,1*(10/5.)),sharex=True,sharey=True,squeeze=False)
if (nrows == 1) | (ncols == 1) :
    ax = ax.reshape(-1)

window_arr = np.array([ 30,  60,  90, 120, 150, 180, 210, 240])




r_Ta_mean = np.array([ 0.48185464, -0.15613735, -0.01936799, -0.1773115 ,  0.05232436,
       -0.04040609,  0.33149468,  0.34869198])
r_FDD = np.array([-0.41736059, -0.17402369,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -0.54129173])
r_SLP = np.array([ 0.53933727, -0.26995156, -0.31650328, -0.08841195, -0.25110975,
       -0.14599478, -0.01014978, -0.17213532])
r_windspeed = np.array([ 0.03678076,  0.45642817,  0.11080172,  0.03563754,  0.20442332,
        0.07753032, -0.01015988,  0.08720983])
r_snow = np.array([-0.62181679, -0.2245711 ,  0.07636528,  0.        ,  0.        ,
        0.        , -0.07603755, -0.42668526])
r_clouds = np.array([-0.34593172,  0.23942407,  0.42465537,  0.18858938, -0.18245648,
        0.20752837,  0.02488345, -0.20860139])
r_SH = np.array([ 0.47743994, -0.21605111, -0.01915353, -0.18801891,  0.05747183,
       -0.07544896,  0.32215516,  0.28334944])
r_RH = np.array([-0.2249827 , -0.23994657,  0.49642031,  0.1627024 , -0.02848038,
        0.217345  ,  0.1291833 , -0.04626647])
r_NAO = np.array([ 0.35391285, -0.1325903 , -0.51696073, -0.40054619,  0.00854227,
        0.11188057, -0.13407522,  0.30843509])

r_mean_plot = r_Ta_mean
ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plt.get_cmap('tab20')(0))

r_mean_plot = r_FDD
ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plt.get_cmap('tab20')(2))

r_mean_plot = r_snow
ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plt.get_cmap('tab20')(4))

r_mean_plot = r_RH
ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plt.get_cmap('tab20')(6))

ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_p2,':', color='gray')
ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_m2,':', color='gray')

plt.subplots_adjust(left=0.2,right=0.9,bottom=0.23)
ax[iloc].set_xlim(0,np.nanmax(window_arr)+np.nanmax(window_arr)/10.)
ax[iloc].set_ylim(-1,1)
# ax[iloc].grid()

ax[iloc].set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

ax[iloc].set_xticks(window_arr)
ax[iloc].set_xticklabels(labels_list)




nrows = 1
ncols = 1
fig,ax = plt.subplots(nrows,figsize=(5,1*(10/5.)),sharex=True,sharey=True,squeeze=False)
if (nrows == 1) | (ncols == 1) :
    ax = ax.reshape(-1)

r_mean_plot = r_SLP
ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plt.get_cmap('tab20')(8))

r_mean_plot = r_NAO
ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plt.get_cmap('tab20')(10))

r_mean_plot = r_windspeed
ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plt.get_cmap('tab20')(12))

r_mean_plot = r_clouds
ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plt.get_cmap('tab20')(14))

ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_p2,':', color='gray')
ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_m2,':', color='gray')

plt.subplots_adjust(left=0.2,right=0.9,bottom=0.23)
ax[iloc].set_xlim(0,np.nanmax(window_arr)+np.nanmax(window_arr)/10.)
ax[iloc].set_ylim(-1,1)
# ax[iloc].grid()

ax[iloc].set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

ax[iloc].set_xticks(window_arr)
ax[iloc].set_xticklabels(labels_list)





