#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:44:10 2021

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
import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import read_csv,clean_csv

# ==========================================================================

fp = local_path+'slice/data/raw/water_levels_discharge_ECCC/'
# fp = local_path+'slice/data/raw/QL_ECCC/'

loc_name_list = ['Lasalle', 'LaPrairie','JeteeNo1','QuaiFrontenac']
loc_nb_list = ['02OA016','02OA041','02OA046','02OA047']

# loc_name_list = ['Lasalle']
# loc_nb_list = ['02OA016']

# loc_name_list = ['SteCatherine','AmontSteCatherine', 'AmontStLambert']
# loc_nb_list = ['02OA024','02OA043','02OA044']

# loc_name_list = ['PointeClaire']
# loc_nb_list = ['02OA039']

# loc_name_list = ['JeteeNo1']
# loc_nb_list = ['02OA046']

# loc_name_list = ['SteAnnedeBellevue']
# loc_nb_list = ['02OA013']


date_ref = dt.date(1900,1,1)
date_start = dt.date(1900,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)


# # Name of the month as a three letter abbreviation
# dateFormatted = sampleDate.strftime("%b")
# print("Name of the month as a three letter abbreviation: {}".format(dateFormatted))

# # Name of the month as a three letter abbreviation
# dateFormatted = sampleDate.strftime("%b")
# print("Month name being printed as a Three Letter Abbreviation: {}".format(dateFormatted))



for iloc,loc in enumerate(loc_name_list):
    csv = read_csv(fp+loc_nb_list[iloc]+'_Level_'+loc+'.csv',skip=2)

    csv = clean_csv(csv,[1,3,4,5,6])

    discharge_csv = csv[csv[:,0] == 1][:,1:]
    level_csv = csv[csv[:,0] == 2][:,1:]

    level = np.zeros((ndays,2))*np.nan
    discharge = np.zeros((ndays,2))*np.nan

    for it in range(level_csv.shape[0]):
        date=(dt.date(int(level_csv[it,0]),int(level_csv[it,1]),int(level_csv[it,2]))-date_ref).days
        indx = np.where(time == date)[0]
        level[indx,0] = date
        level[indx,1] = level_csv[it,3]

    for it in range(discharge_csv.shape[0]):
        date=(dt.date(int(discharge_csv[it,0]),int(discharge_csv[it,1]),int(discharge_csv[it,2]))-date_ref).days
        indx = np.where(time == date)[0]
        discharge[indx,0] = date
        discharge[indx,1] = discharge_csv[it,3]



    # ==========================================================================
    np.savez(local_path+'slice/data/processed/water_levels_discharge_ECCC/water_levels_discharge_'+loc,
              level=level,discharge=discharge,date_ref=date_ref)

#%%
def annual_average_from_daily_ts(var_in,years,time,date_ref = dt.date(1900,1,1)):

    var_out = np.zeros((len(years)))*np.nan

    for iyr,year in enumerate(years):
        if ((dt.date(int(year),1,1)-date_ref).days in time) & ((dt.date(int(year+1),1,1)-date_ref).days in time):
            it_start = np.where(time == (dt.date(int(year),1,1)-date_ref).days )[0][0]
            # it_end = np.where(time == (dt.date(int(year+1),1,1)-date_ref).days )[0][0]
            it_end = np.where(time == (dt.date(int(year),12,1)-date_ref).days )[0][0]

            var_yr = var_in[it_start:it_end]
            var_out[iyr] = np.nanmean(var_yr)

    return var_out

years = np.arange(1900,2022)
discharge_annual = annual_average_from_daily_ts(discharge[:,1],years,time,date_ref = dt.date(1900,1,1))
plt.figure()
plt.plot(years,discharge_annual,'o-')

level_annual = annual_average_from_daily_ts(level[:,1],years,time,date_ref = dt.date(1900,1,1))
plt.figure()
plt.plot(years,level_annual,'o-',color=plt.get_cmap('tab20')(2))

#%%
loc_name_list = ['Lasalle', 'LaPrairie','JeteeNo1','QuaiFrontenac']

loc_name_list = ['Lasalle', 'LaPrairie','SteCatherine','AmontSteCatherine', 'AmontStLambert']

loc_name_list = ['SteAnnedeBellevue','Pointeclaire','Lasalle', 'LaPrairie','SteCatherine','AmontSteCatherine']

loc_name_list = ['AmontSteCatherine','SteAnnedeBellevue','Lasalle','PointeClaire', 'AmontStLambert']


fig_l,ax_l = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
fig_l.suptitle('Level')
fig_d,ax_d = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
fig_d.suptitle('Discharge')

for iloc,loc in enumerate(loc_name_list):
    data = np.load(local_path+'slice/data/processed/water_levels_discharge_ECCC/water_levels_discharge_'+loc+'.npz',allow_pickle='TRUE')
    level = data['level'][:,1]
    discharge = data['discharge'][:,1]

    # level_norm = (level-np.nanmin(level))/(np.nanmax(level)-np.nanmin(level))
    level_norm = (level-np.nanmean(level))/(np.nanstd(level))
    discharge_norm = (discharge-np.nanmean(discharge))/(np.nanstd(discharge))

    ax_l.plot(level_norm,label=loc)
    ax_d.plot(discharge_norm,label=loc)
    # ax_l.plot(level,label=loc)
    # ax_d.plot(discharge,label=loc)


ax_l.legend()
ax_d.legend()

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


# loc1 = 'QuaiFrontenac'
# loc2 = 'JeteeNo1'

# loc1 = 'LaPrairie'
# loc2 = 'Lasalle'

# loc1 = 'LaPrairie'
# loc2 = 'JeteeNo1'

loc1 = 'Lasalle'
loc2 = 'JeteeNo1'


loc1 = 'PointeClaire'
loc2 = 'AmontSteCatherine'

data1 = np.load(local_path+'slice/data/processed/water_levels_discharge_ECCC/water_levels_discharge_'+loc1+'.npz',allow_pickle='TRUE')
data2 = np.load(local_path+'slice/data/processed/water_levels_discharge_ECCC/water_levels_discharge_'+loc2+'.npz',allow_pickle='TRUE')
# level_1 = data1['level'][:,1]
level_1 = data1['level'][:6000,1]
mask1 = ~np.isnan(level_1)
# level_2 = data2['level'][:,1]
level_2 = data2['level'][:6000,1]
mask2 = ~np.isnan(level_2)
mask = mask1 & mask2
level_1 = level_1[mask]
level_2 = level_2[mask]

level1_norm = (level_1-np.nanmean(level_1))/(np.nanstd(level_1))
level2_norm = (level_2-np.nanmean(level_2))/(np.nanstd(level_2))

# plt.figure()
# plt.plot(level_1,level_2,'.')
# lincoeff, Rsqr = linear_fit(level_1,level_2)
# print(lincoeff, Rsqr)
# x_fit = np.arange(np.min(level_1)-0.2,np.max(level_1)+0.2)
# plt.plot(x_fit,x_fit*lincoeff[0]+lincoeff[1],'--')
# plt.plot(x_fit,x_fit+lincoeff[1],'--', color='gray')

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5))

ax.plot(level1_norm,level2_norm,'.')
lincoeff, Rsqr = linear_fit(level1_norm,level2_norm)
print(lincoeff, Rsqr)
x_fit = np.arange(0,1,0.05)
ax.plot(x_fit,x_fit*lincoeff[0]+lincoeff[1],'--')
ax.plot(x_fit,x_fit,'--',color='red')
ax.set_xlabel(loc1)
ax.set_ylabel(loc2)

#%%
loc_discharge = 'Lasalle'
loc_level = 'LaSalle'

fig_l,ax_l = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
fig_l.suptitle('Level')
fig_d,ax_d = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
fig_d.suptitle('Discharge')

data = np.load(local_path+'slice/data/processed/water_levels_discharge_ECCC/water_levels_discharge_'+loc_level+'.npz',allow_pickle='TRUE')
level = data['level'][:,1]

data = np.load(local_path+'slice/data/processed/water_levels_discharge_ECCC/water_levels_discharge_'+loc_discharge+'.npz',allow_pickle='TRUE')
discharge = data['discharge'][:,1]

level_norm = (level-np.nanmean(level))/(np.nanstd(level))
discharge_norm = (discharge-np.nanmean(discharge))/(np.nanstd(discharge))


ax_l.plot(level_norm,discharge_norm,'.',label=loc)

mask1 = ~np.isnan(level_norm)
mask2 = ~np.isnan(discharge_norm)
mask = mask1 & mask2
level_norm = level_norm[mask]
discharge_norm = discharge_norm[mask]
lincoeff, Rsqr = linear_fit(level_norm,discharge_norm)
print(lincoeff, Rsqr)
x_fit = np.arange(-3,3,0.05)
ax_l.plot(x_fit,x_fit*lincoeff[0]+lincoeff[1],'--')

ax_d.plot(level,discharge,'.',label=loc)


ax_l.legend()
ax_d.legend()




