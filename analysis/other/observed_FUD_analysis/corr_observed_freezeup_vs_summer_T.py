#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:23:35 2020

@author: Amelie
"""
import numpy as np

import datetime as dt
import calendar

import matplotlib.pyplot as plt

# ==========================================================================
years = [1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,
         1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,
         2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,
         2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

# years=[1990]

weather_name_list = ['MontrealDorval','MontrealDorval','MontrealDorval','MontrealDorval',
                     'Massena', 'Massena', 'Massena','MontrealDorval']
freezeup_name_list = ['BeauharnoisCanal','MontrealPort','SouthShoreCanal', 'LakeStLouis',
                      'Summerstown','Iroquois','LakeStLawrence','LakeStFrancisEAST']



date_ref = dt.date(1900,1,1)
fp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'

# fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(4,12),sharex=True)
mrk = ['.','.','.','.','.','.','.','.','.','.','.','.','.','.']
#
#
# for i in range(len(weather_name_list)):
#     loc_weather = weather_name_list[i]
#     loc_ice = freezeup_name_list[i]

#     file_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
#     weather_data = file_data['weather_data']

#     ice_data = np.load(fp+'freezeup_dates_SLSMC/freezeup_SLSMC_'+loc_ice+'.npz',allow_pickle='TRUE')
#     freezeup_ci = ice_data['freezeup_ci']
#     freezeup_fi = ice_data['freezeup_fi']
#     freezeup_si = ice_data['freezeup_si']

#     avg_temp = weather_data[:,[0,3]]
#     avg_temp[:,1]  = (avg_temp[:,1]- 32) * (5/9.)

#     date=(dt.date(2006,1,1)-date_ref).days
#     i0 = np.where(avg_temp[:,0]==date)[0][0]
#     Tair_clim_avg = np.nanmean(avg_temp[i0:,1])

#     thawing_degree_days= []
#     warm_degree_days= []
#     warmer_degree_days= []
#     date_ci_list = []
#     date_fi_list = []
#     date_si_list = []

#     for year in years:

#         date_start = (dt.date(year,4,1)-date_ref).days
#         date_end = (dt.date(year,10,1)-date_ref).days

#         if (len(np.where(avg_temp[:,0]==date_start)[0]) > 0) & (len(np.where(avg_temp[:,0]==date_end)[0]) > 0):
#             istart = np.where(avg_temp[:,0]==date_start)[0][0]
#             iend = np.where(avg_temp[:,0]==date_end)[0][0]+1

#             time_year = avg_temp[istart:iend,0].copy()
#             Tair_select = avg_temp[istart:iend,1]
#             ci_select = freezeup_ci[iend:iend+180]
#             fi_select = freezeup_fi[iend:iend+180]
#             si_select = freezeup_si[iend:iend+180]

#             thawing_degree_days.append(np.nansum(Tair_select[Tair_select > 0]))
#             warm_degree_days.append(np.nansum(Tair_select[Tair_select > Tair_clim_avg]-Tair_clim_avg))
#             warmer_degree_days.append(np.nansum(Tair_select[Tair_select > 18]-18))

#             if np.sum(~np.isnan(ci_select)) > 0:
#                 date_tmp = date_ref+dt.timedelta(days=int(ci_select[np.where(~np.isnan(ci_select))[0][0]][0]))
#                 doy_ci = (date_tmp - dt.date(int(year),1,1)).days
#                 date_ci_list.append(doy_ci)
#             else:
#                 date_ci_list.append(np.nan)


#             if np.sum(~np.isnan(fi_select)) > 0:
#                 date_tmp = date_ref+dt.timedelta(days=int(fi_select[np.where(~np.isnan(fi_select))[0][0]][0]))
#                 doy_fi = (date_tmp - dt.date(int(year),1,1)).days
#                 date_fi_list.append(doy_fi)
#             else:
#                 date_fi_list.append(np.nan)


#             if np.sum(~np.isnan(si_select)) > 0:
#                 date_tmp = date_ref+dt.timedelta(days=int(si_select[np.where(~np.isnan(si_select))[0][0]][0]))
#                 doy_si = (date_tmp - dt.date(int(year),1,1)).days
#                 date_si_list.append(doy_si)
#             else:
#                 date_si_list.append(np.nan)

#         else:
#             thawing_degree_days.append(np.nan)
#             warm_degree_days.append(np.nan)
#             warmer_degree_days.append(np.nan)
#             date_ci_list.append(np.nan)
#             date_fi_list.append(np.nan)
#             date_si_list.append(np.nan)


#     ax[0].plot(date_fi_list,thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
#     # ax[0,1].plot(date_fi_list,warm_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
#     # ax[0,2].plot(date_fi_list,warmer_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)

#     ax[1].plot(date_si_list,thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
#     # ax[1,1].plot(date_si_list,warm_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
#     # ax[1,2].plot(date_si_list,warmer_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)

#     ax[2].plot(date_ci_list,thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
#     # ax[2,1].plot(date_ci_list,warm_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
#     # ax[2,2].plot(date_ci_list,warmer_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)


#%%
fig_tot,ax_tot = plt.subplots(nrows=3,ncols=1,figsize=(5,12),sharex=True)
plt.suptitle('All locations')

fi_list_tot = []
si_list_tot = []
ci_list_tot = []
tdd_list_tot = []

for i in range(len(weather_name_list)):

    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(5,12),sharex=True)
    plt.suptitle(weather_name_list[i]+' / '+freezeup_name_list[i])

    loc_weather = weather_name_list[i]
    loc_ice = freezeup_name_list[i]

    file_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
    weather_data = file_data['weather_data']

    ice_data = np.load(fp+'freezeup_dates_SLSMC/freezeup_SLSMC_'+loc_ice+'.npz',allow_pickle='TRUE')
    freezeup_ci = ice_data['freezeup_ci']
    freezeup_fi = ice_data['freezeup_fi']
    freezeup_si = ice_data['freezeup_si']

    avg_temp = weather_data[:,[0,3]]
    avg_temp[:,1]  = (avg_temp[:,1]- 32) * (5/9.)

    date=(dt.date(2006,1,1)-date_ref).days
    i0 = np.where(avg_temp[:,0]==date)[0][0]
    Tair_clim_avg = np.nanmean(avg_temp[i0:,1])

    thawing_degree_days= []
    warm_degree_days= []
    warmer_degree_days= []
    date_ci_list = []
    date_fi_list = []
    date_si_list = []

    for year in years:

        date_start = (dt.date(year,4,1)-date_ref).days
        date_end = (dt.date(year,10,1)-date_ref).days

        if (len(np.where(avg_temp[:,0]==date_start)[0]) > 0) & (len(np.where(avg_temp[:,0]==date_end)[0]) > 0):
            istart = np.where(avg_temp[:,0]==date_start)[0][0]
            iend = np.where(avg_temp[:,0]==date_end)[0][0]+1

            time_year = avg_temp[istart:iend,0].copy()
            Tair_select = avg_temp[istart:iend,1]
            ci_select = freezeup_ci[iend:iend+180]
            fi_select = freezeup_fi[iend:iend+180]
            si_select = freezeup_si[iend:iend+180]

            thawing_degree_days.append(np.nansum(Tair_select[Tair_select > 0]))
            warm_degree_days.append(np.nansum(Tair_select[Tair_select > Tair_clim_avg]-Tair_clim_avg))
            warmer_degree_days.append(np.nansum(Tair_select[Tair_select > 18]-18))

            if np.sum(~np.isnan(ci_select)) > 0:
                date_tmp = date_ref+dt.timedelta(days=int(ci_select[np.where(~np.isnan(ci_select))[0][0]][0]))
                doy_ci = (date_tmp - dt.date(int(year),1,1)).days
                date_ci_list.append(doy_ci)
            else:
                date_ci_list.append(np.nan)


            if np.sum(~np.isnan(fi_select)) > 0:
                date_tmp = date_ref+dt.timedelta(days=int(fi_select[np.where(~np.isnan(fi_select))[0][0]][0]))
                doy_fi = (date_tmp - dt.date(int(year),1,1)).days
                date_fi_list.append(doy_fi)
            else:
                date_fi_list.append(np.nan)


            if np.sum(~np.isnan(si_select)) > 0:
                date_tmp = date_ref+dt.timedelta(days=int(si_select[np.where(~np.isnan(si_select))[0][0]][0]))
                doy_si = (date_tmp - dt.date(int(year),1,1)).days
                date_si_list.append(doy_si)
            else:
                date_si_list.append(np.nan)

        else:
            thawing_degree_days.append(np.nan)
            warm_degree_days.append(np.nan)
            warmer_degree_days.append(np.nan)
            date_ci_list.append(np.nan)
            date_fi_list.append(np.nan)
            date_si_list.append(np.nan)




    ax[0].plot(date_fi_list - np.nanmean(date_fi_list),thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
    ax[1].plot(date_si_list - np.nanmean(date_si_list),thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
    ax[2].plot(date_ci_list - np.nanmean(date_ci_list),thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)

    ax_tot[0].plot(date_fi_list - np.nanmean(date_fi_list),thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
    ax_tot[1].plot(date_si_list - np.nanmean(date_si_list),thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)
    ax_tot[2].plot(date_ci_list - np.nanmean(date_ci_list),thawing_degree_days,marker=mrk[i],linestyle='',label=loc_weather+'/'+loc_ice)

    for n in range(len(thawing_degree_days)):
        tdd_list_tot.append(thawing_degree_days[n])

        fi_list_tot.append(date_fi_list[n] - np.nanmean(date_fi_list))
        ci_list_tot.append(date_ci_list[n] - np.nanmean(date_ci_list))
        si_list_tot.append(date_si_list[n] - np.nanmean(date_si_list))


    fi_list = np.array(date_fi_list-np.nanmean(date_fi_list))
    si_list = np.array(date_si_list-np.nanmean(date_si_list))
    ci_list = np.array(date_ci_list-np.nanmean(date_ci_list))
    tdd_list = np.array(thawing_degree_days)

    mask_fi = ~np.isnan(fi_list)
    mask_si = ~np.isnan(si_list)
    mask_ci = ~np.isnan(ci_list)
    mask_out = tdd_list > 2600
    mask_tdd = ~np.isnan(tdd_list)

    mask = mask_fi & mask_tdd
    x = fi_list[mask]
    y = tdd_list[mask]
    if len(x) > 0:
        p=np.polyfit(x,y,1)
        Rsqr = (np.corrcoef(x,y,rowvar=False)[0,1])**2
        x_fit=np.arange(x.min(),x.max())
        A = np.vstack([x_fit, np.ones(len(x_fit))]).T
        slope_fit = np.dot(A,p)
        ax[0].plot(x_fit,slope_fit,'--',color=[0.6,0.6,0.6])
    ax[0].set_xlim(-41,41)
    ax[0].set_ylim(2550,3300)
    ax[0].set_ylim(2550,3300)
    ax[0].text(22,2700,'R$^{2}$ = '+'%3.2f'%Rsqr,color=[0.6,0.6,0.6])

    mask = mask_si & mask_tdd
    x = si_list[mask]
    y = tdd_list[mask]
    if len(x) > 0:
        p=np.polyfit(x,y,1)
        Rsqr = (np.corrcoef(x,y,rowvar=False)[0,1])**2
        x_fit=np.arange(x.min(),x.max())
        A = np.vstack([x_fit, np.ones(len(x_fit))]).T
        slope_fit = np.dot(A,p)
        ax[1].plot(x_fit,slope_fit,'--',color=[0.6,0.6,0.6])
    ax[1].set_xlim(-41,41)
    ax[1].set_ylim(2550,3300)
    ax[1].text(22,2700,'R$^{2}$ = '+'%3.2f'%Rsqr,color=[0.6,0.6,0.6])
    ax[1].set_ylabel('Thawing degree days ($^{\circ}$C)\n(April 1st to Oct. 1st)')

    mask = mask_ci & mask_tdd & mask_out
    x = ci_list[mask]
    y = tdd_list[mask]
    if len(x) > 0:
        p=np.polyfit(x,y,1)
        Rsqr = (np.corrcoef(x,y,rowvar=False)[0,1])**2
        x_fit=np.arange(x.min(),x.max())
        A = np.vstack([x_fit, np.ones(len(x_fit))]).T
        slope_fit = np.dot(A,p)
        ax[2].plot(x_fit,slope_fit,'--',color=[0.6,0.6,0.6])
    ax[2].set_xlim(-41,41)
    ax[2].set_ylim(2550,3300)
    ax[2].text(22,2700,'R$^{2}$ = '+'%3.2f'%Rsqr,color=[0.6,0.6,0.6])
    ax[2].set_xlabel('Deviation from local average freezeup date (days)')

    plt.subplots_adjust(right=0.83)
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(top=0.9, bottom=0.16)
    ax[0].text(-38,3200,'(a) First ice')
    ax[1].text(-38,3200,'(b) Stable ice')
    ax[2].text(-38,3200,'(c) From charts')







fi_list_tot = np.array(fi_list_tot)
si_list_tot = np.array(si_list_tot)
ci_list_tot = np.array(ci_list_tot)
tdd_list_tot = np.array(tdd_list_tot)

mask_fi = ~np.isnan(fi_list_tot)
mask_si = ~np.isnan(si_list_tot)
mask_ci = ~np.isnan(ci_list_tot)
mask_out = tdd_list_tot > 2600
mask_tdd = ~np.isnan(tdd_list_tot)

mask = mask_fi & mask_tdd
x = fi_list_tot[mask]
y = tdd_list_tot[mask]
p=np.polyfit(x,y,1)
Rsqr = (np.corrcoef(x,y,rowvar=False)[0,1])**2
x_fit=np.arange(x.min(),x.max())
A = np.vstack([x_fit, np.ones(len(x_fit))]).T
slope_fit = np.dot(A,p)
ax_tot[0].plot(x_fit,slope_fit,'--',color=[0.6,0.6,0.6])
ax_tot[0].set_xlim(-41,41)
ax_tot[0].set_ylim(2550,3300)
ax_tot[0].text(22,2700,'R$^{2}$ = '+'%3.2f'%Rsqr,color=[0.6,0.6,0.6])
ax_tot[0].text(-38,3200,'(a) First ice')

mask = mask_si & mask_tdd
x = si_list_tot[mask]
y = tdd_list_tot[mask]
p=np.polyfit(x,y,1)
Rsqr = (np.corrcoef(x,y,rowvar=False)[0,1])**2
x_fit=np.arange(x.min(),x.max())
A = np.vstack([x_fit, np.ones(len(x_fit))]).T
slope_fit = np.dot(A,p)
ax_tot[1].plot(x_fit,slope_fit,'--',color=[0.6,0.6,0.6])
ax_tot[1].set_xlim(-41,41)
ax_tot[1].set_ylim(2550,3300)
ax_tot[1].text(22,2700,'R$^{2}$ = '+'%3.2f'%Rsqr,color=[0.6,0.6,0.6])
ax_tot[1].set_ylabel('Thawing degree days ($^{\circ}$C)\n(April 1st to Oct. 1st)')
ax_tot[1].text(-38,3200,'(b) Stable ice')

mask = mask_ci & mask_tdd & mask_out
x = ci_list_tot[mask]
y = tdd_list_tot[mask]
p=np.polyfit(x,y,1)
Rsqr = (np.corrcoef(x,y,rowvar=False)[0,1])**2
x_fit=np.arange(x.min(),x.max())
A = np.vstack([x_fit, np.ones(len(x_fit))]).T
slope_fit = np.dot(A,p)
ax_tot[2].plot(x_fit,slope_fit,'--',color=[0.6,0.6,0.6])
ax_tot[2].set_xlim(-41,41)
ax_tot[2].set_ylim(2550,3300)
ax_tot[2].text(22,2700,'R$^{2}$ = '+'%3.2f'%Rsqr,color=[0.6,0.6,0.6])
ax_tot[2].set_xlabel('Deviation from local average freezeup date (days)')
ax_tot[2].text(-38,3200,'(c) From charts')

plt.subplots_adjust(right=0.83)
plt.subplots_adjust(left=0.2)
plt.subplots_adjust(top=0.9, bottom=0.16)



