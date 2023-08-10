#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:33:40 2020

@author: Amelie
"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'
#%%
import numpy as np

import datetime as dt
import calendar

import matplotlib.pyplot as plt

# ==========================================================================
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
years = [#2006,2007,
          # 2008,2009,2010,2011,
          2012,
          2013,2014,2015,2016,2017,
          2018,2019,2020]
# years = [2011,2012]
# years = [2004,2005,2006,2007,
#          2008,2009,2010,2011]
years = [1992,1993,1994,1995,
         1996,1997,1998,1999,
         2000,2001,2002,2003,
         2004,2005,2006,2007,
         2008,2009,2010,2011,
         2012,2013,2014,2015,
         2016,2017,2018,2019,2020]


# water_cities_name_list = ['Candiac','DesBailletsclean','Atwater']
# water_cities_name_list = ['Atwater','Candiac']
water_cities_name_list = ['Longueuil_updated_cleaned_filled']
# water_cities_name_list = ['DesBailletsclean']
water_SLSMC_name_list = ['StLambert']
# water_SLSMC_name_list = ['StLouisBridge']
# water_ECCC_name_list = ['LaSalle']
water_ECCC_name_list = ['LaSalle','LaPrairie']
weather_name_list = ['MontrealDorval']
freezeup_name_list = ['SouthShoreCanal']

fp = local_path+'slice/data/processed/'

show_linear_model = False
show_series = True
mean_opt = 'centered'
N = 5

date_ci_list_tot = []
temp_list_tot = []

mrk = ['.','+','*']


date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,11,1)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)


loc_weather = weather_name_list[0]
loc_water_SLSMC = water_SLSMC_name_list[0]
loc_water_ECCC = water_ECCC_name_list[0]
loc_ice = freezeup_name_list[0]

label_SLSMC = loc_water_SLSMC
label_ECCC = loc_water_ECCC
label_weather = loc_weather


#%%
water_SLSMC_data = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_water_SLSMC+'.npz',allow_pickle='TRUE')
Twater_SLSMC = water_SLSMC_data['Twater']

plt.figure();plt.plot(Twater_SLSMC[:,1])
xl=[]
xt=[]
for i in np.arange(1980,2020,2):
    ndays = (dt.date(i+1,1,1)-date_ref).days
    if (len(np.where(Twater_SLSMC[:,0]==ndays)[0]) != 0):
        xl.append(np.where(Twater_SLSMC[:,0]==ndays)[0][0])
        xt.append(str(i+1))
plt.xticks(xl,xt)
plt.xlabel('Year')
plt.ylabel('$T_w$ ($^{\circ}$C)')
#%%
loc_water_ECCC = 'Lasalle'
loc_water_ECCC = 'LaPrairie'
water_ECCC_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+loc_water_ECCC+'.npz',allow_pickle='TRUE')
Twater = water_ECCC_data['Twater']

plt.figure();plt.plot(Twater[:,1])
xl=[]
xt=[]
for i in np.arange(1980,2020,2):
    ndays = (dt.date(i+1,1,1)-date_ref).days
    if (len(np.where(Twater[:,0]==ndays)[0]) != 0):
        xl.append(np.where(Twater[:,0]==ndays)[0][0])
        xt.append(str(i+1))
plt.xticks(xl,xt)
plt.xlabel('Year')
plt.ylabel('$T_w$ ($^{\circ}$C)')
#%%
# water_cities_name_list = ['Candiac','DesBailletsclean','Atwater']
loc_water_city = 'Atwater'
water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
Twater = water_city_data['Twater']

plt.figure();plt.plot(Twater[:,1])
xl=[]
xt=[]
for i in np.arange(1980,2020,2):
    ndays = (dt.date(i+1,1,1)-date_ref).days
    if (len(np.where(Twater[:,0]==ndays)[0]) != 0):
        xl.append(np.where(Twater[:,0]==ndays)[0][0])
        xt.append(str(i+1))
plt.xticks(xl,xt)
plt.xlabel('Year')
plt.ylabel('$T_w$ ($^{\circ}$C)')
#%%

# loc_weather = 'MontrealDorval'
loc_weather = 'MontrealMirabel'
# loc_weather = 'StHubert'
# loc_weather = 'MontrealPET'
# loc_weather = 'MontrealMctavish'
file_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
weather_data = file_data['weather_data']
avg_temp = weather_data[:,[0,3]]


plt.figure()
plt.plot(avg_temp[:,1])
xl=[]
xt=[]
for i in np.arange(1980,2020,2):
    ndays = (dt.date(i+1,1,1)-date_ref).days
    if (len(np.where(avg_temp[:,0]==ndays)[0]) != 0):
        xl.append(np.where(avg_temp[:,0]==ndays)[0][0])
        xt.append(str(i+1))
plt.xticks(xl,xt)
plt.xlabel('Year')
plt.ylabel('$T_a$ ($^{\circ}$C)')

#%%
# Dorval_temp = avg_temp
# PET_temp = avg_temp2

# plt.figure();
# plt.scatter(Dorval_temp[9400:,1],PET_temp[9400:,1])


# plt.figure();
# plt.scatter(Dorval_temp[6575:,1],PET_temp[6575:,1])


# plt.figure();
# plt.scatter(Dorval_temp[:6575,1],PET_temp[:6575,1])

#%%

# water_ECCC_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+loc_water_ECCC+'.npz',allow_pickle='TRUE')
# Twater_ECCC = water_ECCC_data['Twater']

file_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
weather_data = file_data['weather_data']

ice_data = np.load(fp+'freezeup_dates_SLSMC/freezeup_SLSMC_'+loc_ice+'.npz',allow_pickle='TRUE')
freezeup_ci = ice_data['freezeup_ci']

avg_temp = weather_data[:,[0,3]]

#%%
# Ta_yrs = np.zeros((366,len(years)))*np.nan

# for iyear,year in enumerate(years):

#     date=(dt.date(year,1,1)-date_ref).days
#     i0 = np.where(time==date)[0][0]
#     i1 = i0+365+calendar.isleap(year)

#     Ta_yrs[0:365+calendar.isleap(year),iyear] = avg_temp[i0:i1,1].copy()
#     if not calendar.isleap(year):
#         Ta_yrs[-1,iyear] = np.nanmean([avg_temp[i0:i1,1][-1],avg_temp[i1,1]])

# Ta_clim_mean= np.nanmean(Ta_yrs,1)
# Ta_clim_std = np.nanstd(Ta_yrs,1)

# plt.figure()
# for i in range(9):
#     Ta_smooth = running_mean(Ta_yrs[:,i],N,mean_opt)
#     plt.plot(Ta_smooth)
# plt.plot(Ta_clim_mean,'-',color='black',linewidth=2)

#%%
if show_series:

    for iyear,year in enumerate(years):

        fig2,ax2 = plt.subplots(nrows=2,ncols=1,figsize=(7,5),sharex=True)

        # date=(dt.date(year,10,1)-date_ref).days
        # i0 = np.where(time==date)[0][0]
        # i1 = i0+180

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365

        time_select = time[i0:i1].copy()
        time_plot = [ (( (date_ref+dt.timedelta(days=int(time_select[j]))) - dt.date(int(year),1,1)).days) for j in range(len(time_select))]

        for i in range(len(water_cities_name_list)):
            loc_water_city = water_cities_name_list[i]
            label_city = loc_water_city
            water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
            Twater_city = water_city_data['Twater']
            Tw_city_select = Twater_city[i0:i1,1].copy()
            x4 = Tw_city_select
            ax2[1].plot(time_plot,x4,'-',color=plt.get_cmap('tab20c')(8+i),label=label_city+' (water filt. plant)')

        for j in range(len(water_ECCC_name_list)):
            loc_water_ECCC = water_ECCC_name_list[j]
            label_ECCC = loc_water_ECCC
            water_ECCC_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+loc_water_ECCC+'.npz',allow_pickle='TRUE')
            Twater_ECCC = water_ECCC_data['Twater']
            Tw_ECCC_select = Twater_ECCC[i0:i1,1].copy()
            x6 = Tw_ECCC_select
            ax2[1].plot(time_plot,x6,'-',color=plt.get_cmap('tab20c')(6+(j)),label=label_ECCC+' (ECCC)')



        Tw_SLSMC_select = Twater_SLSMC[i0:i1,1].copy()
        Ta_select = avg_temp[i0:i1,1].copy()
        ci_select = freezeup_ci[i0:i1].copy()

        Ta_select = (Ta_select - 32) * (5/9.) # From F to degrees C.
        if np.sum(~np.isnan(ci_select)) > 0:
            ci1 = Tw_SLSMC_select[np.where(~np.isnan(ci_select))[0][0]]
        else:
            ci1 = np.nan

        Ta_smooth = running_mean(Ta_select,N,mean_opt)
        ci_select[~np.isnan(ci_select)] = ci1

        # mask_Tw1 = ~np.isnan(Tw_SLSMC_select)
        # mask_Tw2 = ~np.isnan(Tw_city_select)
        # mask_Tw3 = ~np.isnan(Tw_ECCC_select)
        # mask_Ta1 = ~np.isnan(Ta_select)
        # mask_Ta2 = ~np.isnan(Ta_smooth)
        # mask = mask_Tw1 & mask_Tw2 & mask_Ta1 & mask_Tw3

        # x1 = Ta_select[mask]
        # x2 = Ta_smooth[mask]
        # x3 = Tw_SLSMC_select[mask]
        # x4 = Tw_city_select[mask]
        # x5 = ci_select[mask]
        # x6 = Tw_ECCC_select[mask]
        # x7 = time_select[mask]

        x1 = Ta_select
        x2 = Ta_smooth
        x3 = Tw_SLSMC_select
        x5 = ci_select
        x7 = time_select

        time_plot = [ (( (date_ref+dt.timedelta(days=int(x7[j]))) - dt.date(int(year),1,1)).days) for j in range(len(x7))]
        if np.sum(~np.isnan(x5)) < 1:
            ax2[1].text(360,10,'No freezeup data')

        ax2[0].plot(time_plot,x1,'.-',color=plt.get_cmap('tab20')(1),label=label_weather)
        ax2[0].plot(time_plot,x2,'-',color=plt.get_cmap('tab20')(0))

        ax2[1].plot(time_plot,x3,'-',color=plt.get_cmap('tab20b')(8),label=label_SLSMC+' (SLSMC)')

        # ax2[1].plot(time_plot,Tw_city_select-x1,'-',color=plt.get_cmap('tab20b')(6),label=label_ECCC+' (ECCC)')

        ax2[1].plot(time_plot,x5,'*',markerfacecolor=plt.get_cmap('tab20')(5),markeredgecolor=[0,0,0],markersize=15)

        ax2[0].legend()
        ax2[1].legend()

        # ax2[1].set_xlim([305,400])
        # ax2[1].set_ylim([-2,15])

        ax2[0].yaxis.grid(color=(0.9, 0.9, 0.9))
        ax2[1].yaxis.grid(color=(0.9, 0.9, 0.9))

        ax2[1].set_xlabel('Day of year')

        ax2[0].set_ylabel('Air Temp. ($^{\circ}$C)')
        ax2[1].set_ylabel('Water Temp. ($^{\circ}$C)')
        plt.suptitle(str(year))




#%%
show_linear_model=True
if show_linear_model:
    for i in range(len(water_cities_name_list)):
        loc_water_city = water_cities_name_list[i]

        label_city = loc_water_city

        water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
        Twater_city = water_city_data['Twater']

        fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
        mask_SLSMC = ~np.isnan(Twater_SLSMC[:,1])
        mask_city = ~np.isnan(Twater_city[:,1])
        mask = mask_SLSMC & mask_city
        x = Twater_SLSMC[mask,1]
        y = Twater_city[mask,1]
        ax2.scatter(x,y,marker='.')
        ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\nSt-Lambert (SLSMC) ')
        ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\nCandiac Water Filtration Plant')
        cfit, Rsqr_fit = linear_fit(x,y)
        x_fit = np.arange(x.min(),x.max())
        y_fit = x_fit*cfit[0]+cfit[1]
        ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
        ax2.text(5.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
        plt.subplots_adjust(right=0.86)
        plt.subplots_adjust(left=0.16)
        plt.subplots_adjust(top=0.86, bottom=0.16)


        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
        mask_ECCC = ~np.isnan(Twater_ECCC[:,1])
        mask_city = ~np.isnan(Twater_city[:,1])
        mask = mask_ECCC & mask_city
        x = Twater_ECCC[mask,1]
        y = Twater_city[mask,1]
        ax.scatter(x,y,marker='.')
        ax.set_xlabel('T$_{w}$ (C$^{\circ}$)\nLa Prairie (ECCC) ')
        ax.set_ylabel('T$_{w}$ (C$^{\circ}$)\nCandiac Water Filtration Plant')
        cfit, Rsqr_fit = linear_fit(x,y)
        x_fit = np.arange(x.min(),x.max())
        y_fit = x_fit*cfit[0]+cfit[1]
        ax.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
        ax.text(5.6,25,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
        plt.subplots_adjust(right=0.86)
        plt.subplots_adjust(left=0.16)
        plt.subplots_adjust(top=0.86, bottom=0.16)

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
        mask_ECCC = ~np.isnan(Twater_ECCC[:,1])
        mask_SLSMC = ~np.isnan(Twater_SLSMC[:,1])
        mask = mask_ECCC & mask_SLSMC
        x = Twater_SLSMC[mask,1]
        y = Twater_ECCC[mask,1]
        ax.scatter(x,y,marker='.')
        ax.set_xlabel('T$_{w}$ (C$^{\circ}$)\nSt-Lambert (SLSMC) ')
        ax.set_ylabel('T$_{w}$ (C$^{\circ}$)\nLa Prairie (ECCC) ')
        cfit, Rsqr_fit = linear_fit(x,y)
        x_fit = np.arange(x.min(),x.max())
        y_fit = x_fit*cfit[0]+cfit[1]
        ax.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
        ax.text(4,11.5,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
        plt.subplots_adjust(right=0.86)
        plt.subplots_adjust(left=0.16)
        plt.subplots_adjust(top=0.86, bottom=0.16)


#%%

loc_water_city = 'DesBaillets'
water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
Twater_city = water_city_data['Twater']

loc_water_SLSMC = 'StLambert'
# loc_water_SLSMC = 'StLouisBridge'
water_SLSMC_data = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+loc_water_SLSMC+'.npz',allow_pickle='TRUE')
Twater_SLSMC = water_SLSMC_data['Twater']


fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_SLSMC = ~np.isnan(Twater_SLSMC[:,1])
mask_city = ~np.isnan(Twater_city[:,1])
mask = mask_SLSMC & mask_city
x = Twater_SLSMC[mask,1]
y = Twater_city[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc_water_SLSMC+' (SLSMC) ')
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc_water_city+' Filtration Plant')
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(4.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)




fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_t1 = Twater_city[:,0] < 40580
mask_SLSMC = ~np.isnan(Twater_SLSMC[:,1])
mask_city = ~np.isnan(Twater_city[:,1])
mask = mask_SLSMC & mask_city & mask_t1
x = Twater_SLSMC[mask,1]
y = Twater_city[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc_water_SLSMC+' (SLSMC) ')
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc_water_city+' Filtration Plant')
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(5.6,10.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)


fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_t2 = Twater_city[:,0] >= 40580
mask_t3 = Twater_city[:,0] < 42220
mask_SLSMC = ~np.isnan(Twater_SLSMC[:,1])
mask_city = ~np.isnan(Twater_city[:,1])
mask = mask_SLSMC & mask_city & mask_t2 & mask_t3
x = Twater_SLSMC[mask,1]
y = Twater_city[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc_water_SLSMC+' (SLSMC) ')
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc_water_city+' Filtration Plant')
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(5.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)

fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_t4 = Twater_city[:,0] >= 42220
mask_SLSMC = ~np.isnan(Twater_SLSMC[:,1])
mask_city = ~np.isnan(Twater_city[:,1])
mask = mask_SLSMC & mask_city & mask_t4
x = Twater_SLSMC[mask,1]
y = Twater_city[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc_water_SLSMC+' (SLSMC) ')
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc_water_city+' Filtration Plant')
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(5.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)


plt.figure()
plt.plot(Twater_city[:,1])
plt.plot(Twater_SLSMC[:,1])


#%%
# loc1 = 'BoisdesFillion'
# loc1 = 'SteAnne'
loc1 = 'Lasalle'
water_ECCC_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+loc1+'.npz',allow_pickle='TRUE')
Twater_1 = water_ECCC_data['Twater']

loc2 = 'LaPrairie'
# loc2 = 'BoisdesFillion'
water_ECCC_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+loc2+'.npz',allow_pickle='TRUE')
Twater_2 = water_ECCC_data['Twater']


fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_1 = ~np.isnan(Twater_1[:,1])
mask_2 = ~np.isnan(Twater_2[:,1])
mask = mask_1 & mask_2
x = Twater_1[mask,1]
y = Twater_2[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc1)
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc2)
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(4.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)


fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_t4 = Twater_city[:,0] >= 42220
mask_t2 = Twater_city[:,0] >= 40580
mask_t3 = Twater_city[:,0] < 42220
mask = mask_1 & mask_2& mask_t4
x = Twater_1[mask,1]
y = Twater_2[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc1)
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc2)
fit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(5.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)


plt.figure()
plt.plot(Twater_1[:,1])
plt.plot(Twater_2[:,1])




#%%

loc1 = 'Candiac'
water_city_data1 = np.load(fp+'Twater_cities/Twater_cities_'+loc1+'.npz',allow_pickle='TRUE')
Twater_1 = water_city_data1['Twater']

loc2 = 'DesBaillets'
water_city_data2 = np.load(fp+'Twater_cities/Twater_cities_'+loc2+'.npz',allow_pickle='TRUE')
Twater_2 = water_city_data2['Twater']


fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_1 = ~np.isnan(Twater_1[:,1])
mask_2 = ~np.isnan(Twater_2[:,1])
mask = mask_1 & mask_2
x = Twater_1[mask,1]
y = Twater_2[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc1)
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc2)
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(4.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)

fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_t4 = Twater_city[:,0] >= 42220
mask_t2 = Twater_city[:,0] >= 40580
mask_t3 = Twater_city[:,0] < 42220
mask = mask_1 & mask_2& mask_t4
x = Twater_1[mask,1]
y = Twater_2[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc1)
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc2)
fit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(5.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)


plt.figure()
plt.plot(Twater_1[:,1])
plt.plot(Twater_2[:,1])

#%%

loc1 = 'Atwater'
water_city_data1 = np.load(fp+'Twater_cities/Twater_cities_'+loc1+'.npz',allow_pickle='TRUE')
Twater_1 = water_city_data1['Twater']

loc2 = 'DesBaillets'
water_city_data2 = np.load(fp+'Twater_cities/Twater_cities_'+loc2+'.npz',allow_pickle='TRUE')
Twater_2 = water_city_data2['Twater']


fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_1 = ~np.isnan(Twater_1[:,1])
mask_2 = ~np.isnan(Twater_2[:,1])
mask = mask_1 & mask_2
x = Twater_1[mask,1]
y = Twater_2[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc1)
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc2)
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(4.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)

fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_1 = ~np.isnan(Twater_1[:,1])
mask_2 = ~np.isnan(Twater_2[:,1])
mask_t1 = Twater_city[:,0] < 40580
mask = mask_1 & mask_2& mask_t1
x = Twater_1[mask,1]
y = Twater_2[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc1)
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc2)
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(5.6,10.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)

fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_1 = ~np.isnan(Twater_1[:,1])
mask_2 = ~np.isnan(Twater_2[:,1])
mask_t2 = Twater_city[:,0] >= 40580
mask_t3 = Twater_city[:,0] < 42220
mask = mask_1 & mask_2& mask_t2 & mask_t3
x = Twater_1[mask,1]
y = Twater_2[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc1)
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc2)
cfit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(5.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)

fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
mask_t4 = Twater_city[:,0] >= 42220
mask_t2 = Twater_city[:,0] >= 40580
mask_t3 = Twater_city[:,0] < 42220
mask = mask_1 & mask_2& mask_t4
x = Twater_1[mask,1]
y = Twater_2[mask,1]
ax2.scatter(x,y,marker='.')
ax2.set_xlabel('T$_{w}$ (C$^{\circ}$)\n'+loc1)
ax2.set_ylabel('T$_{w}$ (C$^{\circ}$)\n'+loc2)
fit, Rsqr_fit = linear_fit(x,y)
x_fit = np.arange(x.min(),x.max())
y_fit = x_fit*cfit[0]+cfit[1]
ax2.plot(x_fit,y_fit,'--',color=plt.get_cmap('tab20')(2))
ax2.text(5.6,14.2,'y = %4.2f'%cfit[0]+'$\;$x + %4.2f'%cfit[1]+'  (r$^2$=%4.2f'%Rsqr_fit+')',color=plt.get_cmap('tab20')(2))
plt.subplots_adjust(right=0.86)
plt.subplots_adjust(left=0.16)
plt.subplots_adjust(top=0.86, bottom=0.16)


plt.figure()
plt.plot(Twater_1[:,1])
plt.plot(Twater_2[:,1])