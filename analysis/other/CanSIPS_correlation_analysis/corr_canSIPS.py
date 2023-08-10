#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:09:32 2022

@author: Amelie
"""
import numpy as np
import datetime as dt
import calendar
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cartopy.crs as ccrs
import cartopy as cartopy
from functions import detect_FUD_from_Tw, detrend_ts
#%%
def plot_contours_cartopy(var,
                          gridlats,
                          gridlons,
                          proj=ccrs.PlateCarree(),
                          vmin = -30,
                          vmax = 30,
                          colormap ='BrBG',mask_oceans=False,mask_land=False):

    plt.figure(figsize=(6,2.5))
    ax = plt.axes(projection = proj)
    ax.coastlines()
    if mask_oceans: ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
    if mask_land: ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')
    plt.contourf(gridlons,gridlats, var,levels=np.arange(vmin,vmax+1,2),cmap=plt.get_cmap(colormap),transform=proj)
    plt.colorbar()

    # Add contour:
    line_c = ax.contour(gridlons, gridlats, var, levels=np.arange(vmin,vmax+1,2),
                        colors='black',linestyles='dotted',
                        transform=proj)
    plt.setp(line_c.collections, visible=True)

def plot_pcolormesh_cartopy(var,gridlats,gridlons,proj=ccrs.PlateCarree(),mask_oceans=False,mask_land=False):
    plt.figure()
    ax = plt.axes(projection = proj)
    plt.pcolormesh(gridlons, gridlats, var)
    ax.coastlines()
    if mask_oceans: ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
    if mask_land: ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')
    plt.show()



#%%
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['Longueuil','Candiac','Atwater']
station_type = 'cities'
freezeup_opt = 2
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)

Twater_mean = np.nanmean(Twater,axis=1)

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)
years = np.array(years[:-1])
avg_freezeup_doy = avg_freezeup_doy[:-1]

#%%
base = "cansips_hindcast_raw_"
res = "latlon1.0x1.0_"
r_dir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CanSIPS/hindcast/raw/'
p_dir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CanSIPS/hindcast/'

data = np.load(p_dir+base+res+'ensemble_vars_sep_dec_f5months.npz')
# ensm_mean_fcst = data['ensm'][:]
ensm_mean_clim = data['clim'][:]
ensm_mean_anom = data['anomaly'][:]
feature_list = data['feature_list'][:]
years_cansips = np.arange(1979,2022)
clim_start_yr = data['clim_start_yr']
clim_end_yr = data['clim_end_yr']
lat_cansips = data['lat'][:]
lon_cansips = data['lon'][:]

#%%
#    im          lead
#           0  1  2  3  4
# [sept=0][ S, O, N, D, J]
# [Oct =1][ O, N, D, J, F]
# [Nov =2][ N, D, J, F, M]
# [Dec =3][ D, J, F, M, A]

#%%
# for feat in range(len(feature_list)):
#     for lead in range(5):

#         # Make both time series with same years
#         iyr_start = np.where(years_cansips == years[0])[0][0]
#         iyr_end = np.where(years_cansips == years[-1])[0][0]
#         anomaly = True
#         if anomaly:
#             ensm_mean_anom = data['anomaly'][:]
#             data_corr = ensm_mean_anom[feat,iyr_start:iyr_end+1,:,lead,:,:]
#         else:
#             ensm_mean_fcst = data['ensm'][:]
#             data_corr = ensm_mean_fcst[feat,iyr_start:iyr_end+1,:,lead,:,:]

#         p_critical = 0.01
#         rsqr = np.zeros((data_corr.shape[1],data_corr.shape[-2],data_corr.shape[-1]))*np.nan
#         signif = np.zeros((data_corr.shape[1],data_corr.shape[-2],data_corr.shape[-1]))
#         corr = np.zeros((data_corr.shape[1],data_corr.shape[-2],data_corr.shape[-1]))*np.nan

#         for im in range(data_corr.shape[1]):
#             print(im)
#             for i in range(data_corr.shape[2]):
#                 for j in range(data_corr.shape[3]):

#                     if ~np.all(np.isnan(data_corr[:,im,i,j])):
#                         x = np.squeeze(data_corr[:,im,i,j])
#                         model = sm.OLS(avg_freezeup_doy[:], sm.add_constant(x, has_constant='skip'), missing='drop').fit()
#                         rsqr[im,i,j]= model.rsquared
#                         if model.f_pvalue <= p_critical:
#                             signif[im,i,j] = 1
#                         if len(model.params) > 1:
#                             if model.params[1] < 0:
#                                 corr[im,i,j] = np.sqrt(model.rsquared)*-1
#                             if model.params[1] >= 0:
#                                 corr[im,i,j] = np.sqrt(model.rsquared)


#         np.savez(p_dir+'corr'+str(p_critical)+'_CanSIPS_'+feature_list[feat]+'_ensm_'+ 'anom'*anomaly+ 'fcst'*(~anomaly) +'_'+'LongueuilFUD'+'_lead_'+str(lead)+'_'+str(years_cansips[iyr_start])+'_'+str(years_cansips[iyr_end]),
#                   rsqr=rsqr,
#                   corr=corr,
#                   signif=signif,
#                   data_corr = data_corr)

#%%
anomaly = True
iyr_start = np.where(years_cansips == years[0])[0][0]
iyr_end = np.where(years_cansips == years[-1])[0][0]

p_critical = 0.01

feat = 1


# # FORECAST IN JAN VS FUD
# fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(18,8),sharex=True,sharey=True,subplot_kw={'projection': ccrs.PlateCarree()})
# plt.suptitle('Forecast of ' + feature_list[feat] + ' in Jan. vs FUD Longueuil')

# for lead in range(1,5):
#     data_tmp = np.load(p_dir+'corr'+str(p_critical)+'_CanSIPS_'+feature_list[feat]+'_ensm_'+ 'anom'*anomaly+ 'fcst'*(~anomaly) +'_'+'LongueuilFUD'+'_lead_'+str(lead)+'_'+str(years_cansips[iyr_start])+'_'+str(years_cansips[iyr_end])+'.npz')
#     rsqr=data_tmp['rsqr']
#     signif=data_tmp['signif']
#     corr=data_tmp['corr']
#     data_corr = data_tmp['data_corr']

#     title_str = ['','Jan','Dec.','Nov.', 'Oct.', 'Sep.']
#     iax = [0,1,1,0,0]
#     jax = [2,1,0,1,0]
#     proj = ccrs.PlateCarree()

#     im = 4-lead
#     var = corr[im,:,:]
#     c = signif[im,:,:]
#     # plot_pcolormesh_cartopy_with_contours_with_axes(ax[iax[im],jax[im]],var,c,lat_0p2,lon_0p2)
#     ax[iax[lead],jax[lead]].pcolormesh(lon_cansips-0.5, lat_cansips-0.5, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
#     ax[iax[lead],jax[lead]].coastlines()
#     ax[iax[lead],jax[lead]].add_feature(cartopy.feature.LAKES, alpha=0.3)
#     if feat == 0:
#         ax[iax[lead],jax[lead]].add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')
#     # And black line contour where significant:
#     line_c = ax[iax[lead],jax[lead]].contour(lon_cansips-0.5, lat_cansips-0.5, c,
#                         colors=['black'],levels=[1],
#                         transform=ccrs.PlateCarree())

#     plt.setp(line_c.collections, visible=True)
#     ax[iax[lead],jax[lead]].set_title(title_str[lead]+ '- (lead: '+ str(lead) +' months)')




# FORECAST IN DEC VS FUD
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(18,8),sharex=True,sharey=True,subplot_kw={'projection': ccrs.PlateCarree()})
plt.suptitle('Forecast of ' + feature_list[feat] + ' in Dec. vs FUD Longueuil')

for lead in range(4):
    data_tmp = np.load(p_dir+'corr'+str(p_critical)+'_CanSIPS_'+feature_list[feat]+'_ensm_'+ 'anom'*anomaly+ 'fcst'*(~anomaly) +'_'+'LongueuilFUD'+'_lead_'+str(lead)+'_'+str(years_cansips[iyr_start])+'_'+str(years_cansips[iyr_end])+'.npz')
    rsqr=data_tmp['rsqr']
    signif=data_tmp['signif']
    corr=data_tmp['corr']
    data_corr = data_tmp['data_corr']

    title_str = ['Dec.','Nov.', 'Oct.', 'Sep.']
    iax = [1,1,0,0]
    jax = [1,0,1,0]
    proj = ccrs.PlateCarree()

    im = 3-lead
    var = corr[im,:,:]
    c = signif[im,:,:]
    # plot_pcolormesh_cartopy_with_contours_with_axes(ax[iax[im],jax[im]],var,c,lat_0p2,lon_0p2)
    ax[iax[lead],jax[lead]].pcolormesh(lon_cansips-0.5, lat_cansips-0.5, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
    ax[iax[lead],jax[lead]].coastlines()
    ax[iax[lead],jax[lead]].add_feature(cartopy.feature.LAKES, alpha=0.3)
    if feat == 0:
        ax[iax[lead],jax[lead]].add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')
    # And black line contour where significant:
    line_c = ax[iax[lead],jax[lead]].contour(lon_cansips-0.5, lat_cansips-0.5, c,
                        colors=['black'],levels=[1],
                        transform=ccrs.PlateCarree())

    plt.setp(line_c.collections, visible=True)
    ax[iax[lead],jax[lead]].set_title(title_str[lead]+ '- (lead: '+ str(lead) +' months)')




# FORECAST IN NOV VS FUD
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(18,8),sharex=True,sharey=True,subplot_kw={'projection': ccrs.PlateCarree()})
plt.suptitle('Forecast of ' + feature_list[feat] + ' in Nov. vs FUD Longueuil')

for lead in range(3):
    data_tmp = np.load(p_dir+'corr'+str(p_critical)+'_CanSIPS_'+feature_list[feat]+'_ensm_'+ 'anom'*anomaly+ 'fcst'*(~anomaly) +'_'+'LongueuilFUD'+'_lead_'+str(lead)+'_'+str(years_cansips[iyr_start])+'_'+str(years_cansips[iyr_end])+'.npz')
    rsqr=data_tmp['rsqr']
    signif=data_tmp['signif']
    corr=data_tmp['corr']
    data_corr = data_tmp['data_corr']

    title_str = ['Nov.', 'Oct.', 'Sep.']
    iax = [1,0,0]
    jax = [0,1,0]
    proj = ccrs.PlateCarree()

    im = 2-lead
    var = corr[im,:,:]
    c = signif[im,:,:]
    # plot_pcolormesh_cartopy_with_contours_with_axes(ax[iax[im],jax[im]],var,c,lat_0p2,lon_0p2)
    ax[iax[lead],jax[lead]].pcolormesh(lon_cansips-0.5, lat_cansips-0.5, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
    ax[iax[lead],jax[lead]].coastlines()
    ax[iax[lead],jax[lead]].add_feature(cartopy.feature.LAKES, alpha=0.3)
    if feat == 0:
        ax[iax[lead],jax[lead]].add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')
    # And black line contour where significant:
    line_c = ax[iax[lead],jax[lead]].contour(lon_cansips-0.5, lat_cansips-0.5, c,
                        colors=['black'],levels=[1],
                        transform=ccrs.PlateCarree())

    plt.setp(line_c.collections, visible=True)
    ax[iax[lead],jax[lead]].set_title(title_str[lead]+ '- (lead: '+ str(lead) +' months)')



# FORECAST IN OCT VS FUD
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(18,8),sharex=True,sharey=True,subplot_kw={'projection': ccrs.PlateCarree()})
plt.suptitle('Forecast of ' + feature_list[feat] + ' in Oct. vs FUD Longueuil')

for lead in range(2):
    data_tmp = np.load(p_dir+'corr'+str(p_critical)+'_CanSIPS_'+feature_list[feat]+'_ensm_'+ 'anom'*anomaly+ 'fcst'*(~anomaly) +'_'+'LongueuilFUD'+'_lead_'+str(lead)+'_'+str(years_cansips[iyr_start])+'_'+str(years_cansips[iyr_end])+'.npz')
    rsqr=data_tmp['rsqr']
    signif=data_tmp['signif']
    corr=data_tmp['corr']
    data_corr = data_tmp['data_corr']

    title_str = ['Oct.', 'Sep.']
    iax = [0,0]
    jax = [1,0]
    proj = ccrs.PlateCarree()

    im = 1-lead
    var = corr[im,:,:]
    c = signif[im,:,:]
    # plot_pcolormesh_cartopy_with_contours_with_axes(ax[iax[im],jax[im]],var,c,lat_0p2,lon_0p2)
    ax[iax[lead],jax[lead]].pcolormesh(lon_cansips-0.5, lat_cansips-0.5, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
    ax[iax[lead],jax[lead]].coastlines()
    ax[iax[lead],jax[lead]].add_feature(cartopy.feature.LAKES, alpha=0.3)
    if feat == 0:
        ax[iax[lead],jax[lead]].add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')
    # And black line contour where significant:
    line_c = ax[iax[lead],jax[lead]].contour(lon_cansips-0.5, lat_cansips-0.5, c,
                        colors=['black'],levels=[1],
                        transform=ccrs.PlateCarree())

    plt.setp(line_c.collections, visible=True)
    ax[iax[lead],jax[lead]].set_title(title_str[lead]+ '- (lead: '+ str(lead) +' months)')



#%%

feat = 3
lead = 0
data_tmp = np.load(p_dir+'corr'+str(p_critical)+'_CanSIPS_'+feature_list[feat]+'_ensm_'+ 'anom' +'_'+'LongueuilFUD'+'_lead_'+str(lead)+'_'+str(years_cansips[iyr_start])+'_'+str(years_cansips[iyr_end])+'.npz')
rsqr=data_tmp['rsqr']
signif=data_tmp['signif']
corr=data_tmp['corr']
data_corr = data_tmp['data_corr']

# data = data_corr[:,3,:,:] ; title_month = 'Dec.'
# data = data_corr[:,2,:,:]; title_month = 'Nov.'
data = data_corr[:,1,:,:]; title_month = 'Oct.'
# data = data_corr[:,0,:,:]; title_month = 'Sep.'
region = 'YUL'
if region == 'D':
    rlon1, rlat1 = 360-77.5, 43.5
    rlon2, rlat2 = 360-73.5, 45.5
if region == 'YUL':
    rlon1, rlat1 = 360-74.5, 45.5
    rlon2, rlat2 = 360-73.5, 45.5
if region == 'all':
    rlon1, rlat1 = 0.5, -89.5
    rlon2, rlat2 = 359.5,  89.5
if region == 'Dplus':
    rlon1, rlat1 = 360-84.5, 42.5
    rlon2, rlat2 = 360-72.5, 47.5
if region == 'test':
    rlon1, rlat1 = 360-78.5, 31.5
    rlon2, rlat2 = 360-73.5, 37.5

lat = lat_cansips
lon = lon_cansips
ilat1 = np.where(lat == rlat1)[0][0]
ilat2 = np.where(lat == rlat2)[0][0]+1
ilon1 = np.where(lon == rlon1)[0][0]
ilon2 = np.where(lon == rlon2)[0][0]+1

var_select = data[:,ilat1:ilat2,ilon1:ilon2]
lat_select = lat[ilat1:ilat2+1]
lon_select = lon[ilon1:ilon2+1]

fig,ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.pcolormesh(lon_select-0.5, lat_select-0.5, var_select[3,:,:], vmin=-5,vmax=5,cmap=plt.get_cmap('BrBG'))
ax.coastlines()
ax.add_feature(cartopy.feature.LAKES, alpha=0.3)
ax.add_feature(cartopy.feature.RIVERS, alpha=0.3)
#%%
ts = np.nanmean(var_select,axis=(1,2))
anomaly_type = 'linear'
avg_freezeup_doy_det, [m,b] = detrend_ts(avg_freezeup_doy,years,anomaly_type)

m = sm.OLS(avg_freezeup_doy_det,sm.add_constant(ts),missing='drop').fit()
if m.params[1] > 0:
    print(np.sqrt(m.rsquared),m.f_pvalue)
else:
    print(np.sqrt(m.rsquared)*-1,m.f_pvalue)

fig, ax = plt.subplots(figsize=(7,3))
ax.plot(years,avg_freezeup_doy_det,'o-',color='k')
ax.grid()
ax.set_xlabel('Year')
ax.set_ylabel('FUD Anomaly (days)',color='black')
ax.text(1992,21,'R$^2$ = {:.2f}'.format(m.rsquared) + ' (p={:6.4f})'.format(m.f_pvalue))
ax2=ax.twinx()
ax2.plot(years,ts,'*-')
ax2.set_ylabel('Forecast Anomaly', color= plt.get_cmap('tab10')(0))
plt.title(feature_list[feat]+' Forecast Anomaly for '+title_month+' (lead = '+str(lead)+' months)', color= plt.get_cmap('tab10')(0))



#%%
# # data_tmp = np.load(fdir_p+'corr_SST_FUD_threemonthly_SST_0.1deg_' + 'sampled'*sample + 'averaged'*(~sample)+'_'+str(years[yr_start])+'_'+str(years[yr_end-1])+'.npz')
# # rsqr_sst_FUD=data_tmp['rsqr_sst_FUD']
# # signif_sst_FUD=data_tmp['signif_sst_FUD']
# # corr_sst_FUD=data_tmp['corr_sst_FUD']

# fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(18,8),sharex=True,sharey=True,subplot_kw={'projection': ccrs.PlateCarree()})
# title_str = ['Sep.', 'Oct.', 'Nov.', 'Dec.']
# iax = [0,1,0,1]
# jax = [0,0,1,1]
# proj = ccrs.PlateCarree()
# for im in range(corr.shape[0]):
#     var = corr[im,:,:]
#     c = signif[im,:,:]
#     # plot_pcolormesh_cartopy_with_contours_with_axes(ax[iax[im],jax[im]],var,c,lat_0p2,lon_0p2)
#     ax[iax[im],jax[im]].pcolormesh(lon_cansips, lat_cansips, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
#     ax[iax[im],jax[im]].coastlines()
#     # And black line contour where significant:
#     line_c = ax[iax[im],jax[im]].contour(lon_cansips, lat_cansips, c,
#                         colors=['black'],levels=[1],
#                         transform=ccrs.PlateCarree())

#     plt.setp(line_c.collections, visible=True)

#     ax[iax[im],jax[im]].set_title(title_str[im])



