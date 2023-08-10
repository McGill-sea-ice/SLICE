#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:48:46 2021

@author: Amelie
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy
from scipy import ndimage
from functions import running_nanmean,find_freezeup_Tw,find_breakup_Tw
from functions import fill_gaps
#==========================================================================

water_SLSMC = []
# water_ECCC_name_list = ['Lasallepatchlinear']
# water_city_list = []
water_ECCC_name_list = []
water_city_list = ['DesBaillets_cleaned_filled']
freezeup_name_list = ['SouthShoreCanal']
weather_name_list = ['MontrealDorvalMontrealPETMontrealMcTavishmerged']

fp = '../../data/processed/'

n1=len(water_SLSMC)
n2=len(water_ECCC_name_list)
n3=len(water_city_list)
n4=len(freezeup_name_list)*3
n5=len(weather_name_list)*7

ntot = n1+n2+n3+n4+n5+1

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

years = np.arange(30)+1991
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#%%
"""
Variables for ML:
    - time
    - Tw
    - ice (1) or no ice (0)
    - Avg. Ta_max
    - Avg. Ta_min
    - Avg. Ta_mean
    - Tot. TDD
    - Tot. FDD
    - Tot. CDD
    - Tot. precip.
    - Avg. SLP
    - Avg. wind speed
    - Tot. snowfall
    - Avg. cloud cover
    - Avg. spec. hum.
    - Avg. rel. hum.
    - St-Lawrence River discharge @ Lasalle
    - St-Lawrence River level @ Pointe-Claire
    - Ottawa River level @ Ste-Anne-de-Bellevue
    - NAO index

"""

#%%
# LOAD TWATER TIME SERIES
water_name_list = ['Atwater_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
station_labels = ['Atwater','Longueuil','Candiac']
station_type = 'cities'

Twater_all = np.zeros((len(time),len(water_name_list)+1))*np.nan

for iloc,loc in enumerate(water_name_list):
    loc_water_loc = water_name_list[iloc]
    water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
    Twater_tmp = water_loc_data['Twater'][:,1]

    # APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST
    Twater_all[:,iloc] = Twater_tmp
    if loc == 'Candiac_cleaned_filled':
        Twater_all[:,iloc] = Twater_tmp-0.8
    if (loc == 'Atwater_cleaned_filled'):
        Twater_all[0:12490,iloc] = Twater_tmp[0:12490]-0.7


# ADD THE MEAN OF ALL WATER TEMPERATURE AT THE END
Twater_all[:,-1] = np.nanmean(Twater_all[:,0:len(water_name_list)],axis=1)


#%%
# FIND FREEZE-UP + BREAK-UP DATES FROM THE TWATER TIME SERIES
freezeup_opt = 3
month_start_day = 1

# OPTION 1
if freezeup_opt == 1:
    def_opt = 1
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = False
    T_thresh = 0.75
    dTdt_thresh = 0.25
    d2Tdt2_thresh = 0.25
    nd = 1
    no_negTw = True

# OPTION 2
if freezeup_opt == 2:
    def_opt = 3
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = True
    sig_dog = 3.5
    T_thresh = 3.
    dTdt_thresh = 0.15
    d2Tdt2_thresh = 0.15
    # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
    # d2Tdt2_thresh = 0.20
    nd = 30
    no_negTw = True

# OPTION 3 = TESTS FOR BREAK-UP DEFINITION
if freezeup_opt == 3:
    def_opt = 3
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = True
    sig_dog = 3.5
    T_thresh = 0.75
    dTdt_thresh = 0.15
    d2Tdt2_thresh = 0.15
    nd = 7
    no_negTw = True

    # def_opt = 1
    # smooth_T =False; N_smooth = 3; mean_type='centered'
    # round_T = False; round_type= 'half_unit'
    # Gauss_filter = False
    # T_thresh = 0.75
    # dTdt_thresh = 0.25
    # d2Tdt2_thresh = 0.25
    # nd = 1
    # no_negTw = True

plot_Tw_ts = True
freezeup_date_all = np.zeros((len(years),3,len(water_name_list)+1))*np.nan
breakup_date_all = np.zeros((len(years),3,len(water_name_list)+1))*np.nan
freezeup_doy_all = np.zeros((len(years),len(water_name_list)+1))*np.nan
breakup_doy_all = np.zeros((len(years),len(water_name_list)+1))*np.nan

if plot_Tw_ts:
    fig_tw,ax_tw = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))

# for iloc,loc in enumerate(water_name_list):
for iloc in range(len(water_name_list)+1):
    Twater_tmp = Twater_all[:,iloc].copy()

    #FIND DTDt, D2TDt2, etc.
    if round_T:
        if round_type == 'unit':
            Twater_tmp = np.round(Twater_tmp.copy())
        if round_type == 'half_unit':
            Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
    if smooth_T:
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)
    if no_negTw:
        Twater_tmp[Twater_tmp < 0] = 0.0

    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_d2Tdt2 = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

    if Gauss_filter:
        Twater_DoG1 = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
        Twater_DoG2 = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

    # THEN FIND FREEZE-UP & BREAK-UP DATES ACCORDING TO CHOSEN OPTION:
    if def_opt == 3:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_DoG1,Twater_DoG2,time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        bd, btw, T_breakup, mask_break = find_breakup_Tw(def_opt,Twater_tmp,Twater_DoG1,Twater_DoG2,time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_date_all[:,:,iloc] = fd
        breakup_date_all[:,:,iloc] = bd
    else:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt,Twater_d2Tdt2,time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        bd, btw, T_breakup, mask_break = find_breakup_Tw(def_opt,Twater_tmp,Twater_dTdt,Twater_d2Tdt2,time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_date_all[:,:,iloc] = fd
        breakup_date_all[:,:,iloc] = bd

    if plot_Tw_ts:
        ax_tw.plot(time,Twater_tmp,color=plt.get_cmap('tab20')(iloc*2+1))
        ax_tw.plot(time,T_freezeup, '*',color=plt.get_cmap('tab20')(iloc*2))
        ax_tw.plot(time,T_breakup, 'o',color=plt.get_cmap('tab20')(iloc*2))


    # FINALLY, TRANSFORM DATES FORMAT TO DOY FORMAT:
    for iyr,year in enumerate(years):
        if ~np.isnan(fd[iyr,0]):
            fd_yy = int(fd[iyr,0])
            fd_mm = int(fd[iyr,1])
            fd_dd = int(fd[iyr,2])

            fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
            if fd_doy < 60: fd_doy += 365

            freezeup_doy_all[iyr,iloc]=fd_doy

        if ~np.isnan(bd[iyr,0]):
            bd_yy = int(bd[iyr,0])
            bd_mm = int(bd[iyr,1])
            bd_dd = int(bd[iyr,2])

            breakup_doy_all[iyr,iloc] = (dt.date(bd_yy,bd_mm,bd_dd)-dt.date(bd_yy,1,1)).days + 1



#%%
# CREATE AN ICE/NO ICE VARIABLE

ice_noice_all = np.zeros((len(time),len(water_name_list)+1))

for iloc in range(len(water_name_list)+1):
    for iyr,year in enumerate(years):
        if ~np.isnan(freezeup_date_all[iyr,0,iloc]):
            fd_yy = int(freezeup_date_all[iyr,0,iloc])
            fd_mm = int(freezeup_date_all[iyr,1,iloc])
            fd_dd = int(freezeup_date_all[iyr,2,iloc])

            fd_yr = (dt.date(fd_yy,fd_mm,fd_dd) - date_ref).days
            ifd = np.where(time == fd_yr)[0][0]

        if ~np.isnan(breakup_date_all[iyr,0,iloc]):
            bd_yy = int(breakup_date_all[iyr,0,iloc])
            bd_mm = int(breakup_date_all[iyr,1,iloc])
            bd_dd = int(breakup_date_all[iyr,2,iloc])

            bd_yr = (dt.date(bd_yy,bd_mm,bd_dd) - date_ref).days
            ibd = np.where(time == bd_yr)[0][0]

        # Both the freeze-up and break-up dates were detected for the season
        if (~np.isnan(freezeup_date_all[iyr,0,iloc]) & ~np.isnan(breakup_date_all[iyr,0,iloc])):
            ice_noice_all[ifd:ibd,iloc] = 1

        # Only a break-up date was detected (this only happens when the time series of Tw start during winter)
        if np.isnan(freezeup_date_all[iyr,0,iloc]) & ~np.isnan(breakup_date_all[iyr,0,iloc]):
            ice_noice_all[0:ibd,iloc] = 1

        # Only a freeze-up date was detected (this happens when there is missing Tw data during winter, for example)
        # Then, use the break-up date of the mean Tw time series as a reference break-up date
        if ~np.isnan(freezeup_date_all[iyr,0,iloc]) & np.isnan(breakup_date_all[iyr,0,iloc]):

            if ~np.isnan(breakup_date_all[iyr,0,-1]):
                bd_yy = int(breakup_date_all[iyr,0,-1])
                bd_mm = int(breakup_date_all[iyr,1,-1])
                bd_dd = int(breakup_date_all[iyr,2,-1])
                bd_yr = (dt.date(bd_yy,bd_mm,bd_dd) - date_ref).days
                ibd = np.where(time == bd_yr)[0][0]

                ice_noice_all[ifd:ibd,iloc] = 1
            else:
                ice_noice_all[ifd:,iloc] = 1

    # Remove where there is no Twater data:
    ice_noice_all[:,iloc][np.isnan(Twater_all[:,iloc])]=np.nan

# Plot to compare with Tw timeseries:
if plot_Tw_ts:
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,3.5))
    for iloc in range(4):
        ax.plot(time,Twater_all[:,iloc],color=plt.get_cmap('tab20')(iloc*2+1))
        Tplot = Twater_all[:,iloc].copy()
        Tplot[ice_noice_all[:,iloc] == 0] = np.nan
        ax.plot(time,Tplot,color=plt.get_cmap('tab20')(iloc*2))


#%%
# GET WEATHER DATA FROM ERA5

weather_loc_list = ['D','A','B','E']

weather_varnames = ['Avg. Ta_max [deg C]',
                    'Avg. Ta_min [deg C]',
                    'Avg. Ta_mean [deg C]',
                    'Tot. TDD [deg C]',
                    'Tot. FDD [deg C]',
                    'Tot. precip. [m]',
                    'Avg. SLP [kPa]',
                    'Avg. uwind [m/s]',
                    'Avg. vwind [m/s]',
                    'Avg. wind speed [m/s]',
                    'Tot. snowfall [m of water equiv.]',
                    'Avg. cloud cover [%]',
                    'Avg. spec. hum. []',
                    'Avg. rel. hum. [%]']


weather_vars_all = np.zeros((len(time),len(weather_varnames),len(weather_loc_list)))*np.nan

for iloc,weather_loc in enumerate(weather_loc_list):

    weather_data = np.load(fp+'weather_ERA5/weather_ERA5_region'+weather_loc+'.npz',allow_pickle='TRUE')
    weather = weather_data['weather_data']

    max_Ta = weather[:,1] # Daily max. Ta [K]
    min_Ta = weather[:,2] # Daily min. Ta [K]
    avg_Ta = weather[:,3] # Daily avg. Ta [K]
    precip = weather[:,4] # Daily total precip. [m]
    slp = weather[:,5] # Sea level pressure [Pa]
    uwind = weather[:,6] # U-velocity of 10m wind [m/s]
    vwind = weather[:,7] # V-velocity of 10m wind [m/s]
    max_Td = weather[:,9] # Daily max. dew point [K]
    min_Td = weather[:,10] # Daily min. dew point [K]
    avg_Td = weather[:,11] # Daily avg. dew point [K]
    snow = weather[:,12] # Snowfall [m of water equivalent]
    clouds = weather[:,13] # Total cloud cover [%]

    # Convert to kPa:
    slp = slp/1000.
    # Convert Kelvins to Celsius:
    max_Ta  = (max_Ta-273.15)
    min_Ta  = (min_Ta-273.15)
    avg_Ta  = (avg_Ta-273.15)
    max_Td  = (max_Td-273.15)
    min_Td  = (min_Td-273.15)
    avg_Td  = (avg_Td-273.15)

    # Derive new variables:
    windspeed = np.sqrt(uwind**2 + vwind**2)
    e_sat =0.61094*np.exp((17.625*avg_Ta)/(avg_Ta +243.04)) # Saturation vapor pressure (August–Roche–Magnus formula)
    avg_SH = 0.622*e_sat/(slp-0.378*e_sat) # Daily avg. specific humidity
    avg_RH = (np.exp((17.625*avg_Td)/(243.04+avg_Td))/np.exp((17.625*avg_Ta)/(243.04+avg_Ta))) # Daily avg. relative humidity

    mask_FDD = (avg_Ta <= 0)
    FDD = avg_Ta.copy()
    FDD[~mask_FDD] = np.nan

    mask_TDD = (avg_Ta > 0)
    TDD = avg_Ta.copy()
    TDD[~mask_TDD] = np.nan

    weather_vars_all[:,0,iloc] = max_Ta
    weather_vars_all[:,1,iloc] = min_Ta
    weather_vars_all[:,2,iloc] = avg_Ta
    weather_vars_all[:,3,iloc] = TDD
    weather_vars_all[:,4,iloc] = -1*FDD
    weather_vars_all[:,5,iloc] = precip
    weather_vars_all[:,6,iloc] = slp
    weather_vars_all[:,7,iloc] = uwind
    weather_vars_all[:,8,iloc] = vwind
    weather_vars_all[:,9,iloc] = windspeed
    weather_vars_all[:,10,iloc] = snow
    weather_vars_all[:,11,iloc] = clouds
    weather_vars_all[:,12,iloc] = avg_SH
    weather_vars_all[:,13,iloc] = avg_RH


#%%
# GET DISCHARGE AND LEVEL DATA
level_SL_loc = 'PointeClaire'
level_OR_loc = 'SteAnnedeBellevue'
discharge_loc = 'Lasalle'

level_SL_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+level_SL_loc+'.npz',allow_pickle='TRUE')
level_OR_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+level_OR_loc+'.npz',allow_pickle='TRUE')
discharge_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+discharge_loc+'.npz',allow_pickle='TRUE')

hydro_varnames = ['Avg. St-L. level [m]',
                  'Avg. Ott. Riv. level [m]',
                  'Avg. St-L. discharge [m^3/s]'
                  ]

hydro_vars = np.zeros((len(time),len(hydro_varnames)))*np.nan
hydro_vars[:,0],_ = fill_gaps(level_SL_data['level'][:,1])
hydro_vars[:,1],_  = fill_gaps(level_OR_data['level'][:,1])
hydro_vars[:,2],_  = fill_gaps(discharge_data['discharge'][:,1])

#%%
# GET NAO INDEX:
NAO_data = np.load(fp+'NAO_index_NOAA/NAO_index_NOAA_monthly.npz',allow_pickle='TRUE')
NAO = NAO_data['NAO_data']
NAO_varnames = ['NAO']


#%%
# GET PDO INDEX:
fn = 'PDO_index_NOAA_monthly_ersstv3.npz'
PDO_version = 'ersstv3'
PDO_data = np.load(fp+'PDO_index_NOAA/'+fn,allow_pickle='TRUE')
PDO = PDO_data['PDO_data']
PDO_varnames = ['PDO']

#%%
# SELECT OR MERGE TWATER LOCATION
Tw_loc = 1
Tw_loc_name = 'Longueuil'
# Tw_loc = -1
# Tw_loc_name = 'MeanLongueuilAtwaterCandiac'

Twater = np.zeros((len(time),1))*np.nan
ice_noice = np.zeros((len(time),1))*np.nan
freezeup_date = np.zeros((len(years),3,1))*np.nan
breakup_date = np.zeros((len(years),3,1))*np.nan

Twater[:,0] = Twater_all[:,Tw_loc]
ice_noice[:,0] = ice_noice_all[:,Tw_loc]
freezeup_date[:,:,0] = freezeup_date_all[:,:,Tw_loc]
breakup_date[:,:,0] = breakup_date_all[:,:,Tw_loc]

# SELECT ERA5 WEATHER LOCATION
weather_loc = 0 # 'D': MLO + OR
weather_loc_name = 'D'

weather_vars = np.zeros((len(time),len(weather_varnames),1))*np.nan
weather_vars[:,:,0] = weather_vars_all[:,:,weather_loc]


#%%
# NOW COMBINE EVERYTHING INTO A SINGLE ARRAY

# Initialize array
ntot = 1+2+1+len(weather_varnames)+len(hydro_varnames)+1+1

ds = np.zeros((len(time),ntot))
labels = np.chararray(ntot, itemsize=100)
save_name = ''

# Add time as first column:
ds[:,0] = time
labels[0] = 'Days since '+str(date_ref)

# Add water temperature time series:
ds[:,1] = Twater[:,0]
labels[1] = 'Twater city [deg.C] - '+Tw_loc_name
ds[:,2] = ice_noice[:,0]
labels[2] = 'Binary ice/no ice (1/0)'
save_name += 'Twater'+Tw_loc_name + '_'

# Add Twater - Tair_mean:
ds[:,3] = Twater[:,0] - weather_vars[:,2,0]
labels[3] = 'Twater-Tair [deg.C]'

# Add weather data:
for i in range(len(weather_varnames)):
    labels[4+i] = weather_varnames[i]
    ds[:,4+i] = weather_vars[:,i,0]

save_name += 'ERA5region'+weather_loc_name + '_'

# Add discharge and water level data:
for i in range(len(hydro_varnames)):
    labels[4+len(weather_varnames)+i] = hydro_varnames[i]
    ds[:,4+len(weather_varnames)+i] = hydro_vars[:,i]

save_name += 'Q'+discharge_loc+level_SL_loc+level_OR_loc + '_'

#Add NAO index:
ds[:,4+len(weather_varnames)+len(hydro_varnames)] = NAO[:,0]
labels[4+len(weather_varnames)+len(hydro_varnames)] = 'Monthly NAO index'

#Add PDO index:
ds[:,4+len(weather_varnames)+len(hydro_varnames)+1] = PDO[:,0]
labels[4+len(weather_varnames)+len(hydro_varnames)+1] = 'Monthly PDO index'

save_name += 'PDO'+PDO_version


#%%
# SAVE DATASET
np.savez('../../data/ML_timeseries/ML_dataset_'+save_name,
          data=ds,labels=labels,date_ref=date_ref)

#%%
fig,ax = plt.subplots(nrows=ntot-1,ncols=1,figsize=(12,1.5*ntot),sharex=True)

for n in range(ntot-1):
    ax[n].plot(time,ds[:,n+1])




