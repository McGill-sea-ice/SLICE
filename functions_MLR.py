#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:49:42 2021

@author: Amelie
"""


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import calendar
import scipy as sp
import pandas as pd
import cdsapi
import statsmodels.api as sm
import os
import itertools
from netCDF4 import Dataset
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir='./temp_files/') #python

from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib

from functions import linear_fit, read_csv, clean_csv, predicted_r2
from functions import r_confidence_interval

#%%%==============================================================

###
def get_monthly_vars_from_daily(vars_in,names_in,years,time,replace_with_nan=True,date_ref=dt.date(1900,1,1)):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),12))*np.nan

    for ivar,varname in enumerate(names_in):

        for iyr, year in enumerate(years):

            for imonth in range(12):
                month = imonth+1

                if month == 12:
                    imonth1st = (dt.date(int(year),month,1)-date_ref).days
                    imonth1st = np.where(time== imonth1st)[0][0]
                    imonthend = (dt.date(int(year+1),1,1)-date_ref).days-1
                    imonthend = np.where(time==imonthend)[0][0]+1
                else:
                    imonth1st = (dt.date(int(year),month,1)-date_ref).days
                    imonth1st = np.where(time== imonth1st)[0][0]
                    imonthend = (dt.date(int(year),month+1,1)-date_ref).days-1
                    imonthend = np.where(time==imonthend)[0][0]+1

                var_month = vars_in[imonth1st:imonthend,ivar]

                if (varname[0:3] == 'Avg'):
                    if np.sum(np.isnan(var_month)) < 0.5*len(var_month):
                        # Perform the monthly average or accumulation only if at least 75% the month is available
                        vars_out[ivar,iyr,imonth] = np.nanmean(var_month)
                    else:
                        vars_out[ivar,iyr,imonth] = np.nan

                if (varname[0:3] == 'Tot'):
                    if np.nansum(var_month) == 0:
                        if replace_with_nan:
                            vars_out[ivar,iyr,imonth] = np.nan # REPLACE CUMMULATED VARIABLES THAT ARE ZERO WITH NAN TO REMOVE THIS POINT FROM THE CORRELATION
                        else:
                            vars_out[ivar,iyr,imonth] = 0
                    else:
                        vars_out[ivar,iyr,imonth] = np.nansum(var_month)


    return vars_out


###
def get_3month_vars_from_daily(vars_in,names_in,years,time,replace_with_nan=True,date_ref=dt.date(1900,1,1)):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),12))*np.nan

    for ivar,varname in enumerate(names_in):

        for iyr, year in enumerate(years):

            for month_3r in range(2,12):
                # -, -, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND
                if month_3r == 0:
                    year1 = year-1
                    year3 = year
                    month1 = 11
                    month3 = 1
                elif month_3r == 1:
                    year1 = year-1
                    year3 = year
                    month1 = 12
                    month3 = 2
                else:
                    year1 = year
                    year3 = year
                    month1 = month_3r-1
                    month3 = month1 + 2

                if month3 == 12:
                    imonth1st = (dt.date(int(year1),month1,1)-date_ref).days
                    imonth1st = np.where(time== imonth1st)[0][0]
                    imonthend = (dt.date(int(year3+1),1,1)-date_ref).days-1
                    imonthend = np.where(time==imonthend)[0][0]+1
                else:
                    imonth1st = (dt.date(int(year1),month1,1)-date_ref).days
                    imonth1st = np.where(time== imonth1st)[0][0]
                    imonthend = (dt.date(int(year3),month3+1,1)-date_ref).days-1
                    imonthend = np.where(time==imonthend)[0][0]+1

                var_month = vars_in[imonth1st:imonthend,ivar]

                if (varname[0:3] == 'Avg'):
                    if np.sum(np.isnan(var_month)) < 0.5*len(var_month):
                        # Perform the monthly average or accumulation only if at least 75% the month is available
                        vars_out[ivar,iyr,month_3r] = np.nanmean(var_month)
                    else:
                        vars_out[ivar,iyr,month_3r] = np.nan

                if (varname[0:3] == 'Tot'):
                    if np.nansum(var_month) == 0:
                        if replace_with_nan:
                            vars_out[ivar,iyr,month_3r] = np.nan # REPLACE CUMMULATED VARIABLES THAT ARE ZERO WITH NAN TO REMOVE THIS POINT FROM THE CORRELATION
                        else:
                            vars_out[ivar,iyr,month_3r] = 0
                    else:
                        vars_out[ivar,iyr,month_3r] = np.nansum(var_month)

    return vars_out




###
def get_rollingwindow_vars_from_daily(vars_in,names_in,start_doy,window_size,nslide,years,time,replace_with_nan=True,date_ref=dt.date(1900,1,1)):

    nvars = len(names_in)
    nwindows = np.max([int(np.floor(((start_doy-1)-window_size)/nslide))+1, int(np.floor((start_doy-window_size)/nslide))+1])
    vars_out = np.zeros((nvars,len(years),nwindows))*np.nan

    for ivar,varname in enumerate(names_in):
        var= vars_in[:,ivar]
        varname = names_in[ivar]
        for iyr, year in enumerate(years):

            i0 = (dt.date(int(year),1,1)-date_ref).days
            i0 = np.where(time == i0)[0][0]

            if calendar.isleap(year):
                i1 = ((dt.date(int(year),1,1)+dt.timedelta(days=start_doy))-date_ref).days
            else:
                i1 = ((dt.date(int(year),1,1)+dt.timedelta(days=start_doy-1))-date_ref).days

            try:
                i1 = np.where(time == i1)[0][0]
            except:
                i1 = len(time)-1

            var_yr = var[i0:i1]
            # doy_arr = np.arange(1, (i1-i0)+1)
            # print(i0,i1,date_ref+dt.timedelta(days=int(time[i1])),date_ref+dt.timedelta(days=int(time[i0])))
            # print(i1-i0,var[i0:i1].shape,time[i0:i1].shape,date_ref+dt.timedelta(days=int(time[i0:i1][0])),date_ref+dt.timedelta(days=int(time[i0:i1][-1])))

            for iw in range(nwindows):
                # The window '0' is the closest to the start date, and the higher the window number, the further it gets back in time.
                istart = -window_size-(iw*nslide)
                iend = (iw>0)*(-(iw*nslide)) + len(var_yr)*(iw==0)

                var_w = var_yr[istart:iend]

                if (varname[0:3] == 'Avg'):
                    if np.sum(np.isnan(var_w)) < 0.5*len(var_w):
                        # Perform the monthly average or accumulation only if at least 75% the month is available
                        vars_out[ivar,iyr,iw] = np.nanmean(var_w)
                    else:
                        vars_out[ivar,iyr,iw] = np.nan

                if (varname[0:3] == 'Tot'):
                    if np.nansum(var_w) == 0:
                        if replace_with_nan:
                            vars_out[ivar,iyr,iw] = np.nan # REPLACE CUMMULATED VARIABLES THAT ARE ZERO WITH NAN TO REMOVE THIS POINT FROM THE CORRELATION
                        else:
                            vars_out[ivar,iyr,iw] = 0
                    else:
                        vars_out[ivar,iyr,iw] = np.nansum(var_w)


    return vars_out


#%%%==============================================================

###
def datecheck_var_npz(varname,varpath,date_check,past_days=30,n=0.75,date_ref = dt.date(1900,1,1)):

    var = np.load(varpath+'.npz')[varname]
    var = np.squeeze(var)

    if len(var.shape) > 1 :
        var = var[:,1]
    var = np.squeeze(var)

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)

    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)


    date_time = (date_check-date_ref).days

    if (date_time in time) & ((date_time-past_days) in time):

        it0 = np.where(time == date_time-past_days)[0][0]
        it1 = np.where(time == date_time)[0][0] +1

        n_available = np.sum(~np.isnan(var[it0:it1]))/var[it0:it1].shape[0]

        if n_available < n :
            available = False
        else:
            available = True

        return available, n_available

    else:
        print('!!!! PROBLEM !!!!')
        print("--> Requested date_check is not in available dates... ")


###
def datecheck_var(var,date_check,past_days=30,n=0.75,date_ref = dt.date(1900,1,1)):

    if len(var.shape) > 1 :
        var = var[:,1]
    var = np.squeeze(var)

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)

    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)


    date_time = (date_check-date_ref).days

    if (date_time in time) & ((date_time-past_days) in time):

        it0 = np.where(time == date_time-past_days)[0][0]
        it1 = np.where(time == date_time)[0][0] +1

        n_available = np.sum(~np.isnan(var[it0:it1]))/var[it0:it1].shape[0]

        if n_available < n :
            available = False
        else:
            available = True

        return available, n_available

    else:
        print('!!!! PROBLEM !!!!')
        print("--> Requested date_check is not in available dates... ")


###
def update_water_discharge(date_update,loc_name,loc_nb,raw_fp,processed_fp,save=False):
    # raw_fp = '../../data/raw/QL_ECCC/'
    # processed_fp = '../../data/processed/water_levels_discharge_ECCC/'
    # loc_name = 'PointeClaire'
    # loc_nb = '02OA039'
    # # date_update = dt.date.today().strftime("%b")+'-'+str(dt.date.today().day)+'-'+str(dt.date.today().year)
    # date_update = 'Nov-11-2021'
    # save = False

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    ndays = (date_end-date_start).days + 1

    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

    # Load previous data
    past_data = np.load(processed_fp+'water_levels_discharge_'+loc_name+'.npz')
    past_discharge = past_data['discharge']


    # Load new csv data
    fname = loc_nb+'_QR_'+ date_update
    if os.path.isfile(raw_fp+fname+'.csv'):


        csv = read_csv(raw_fp+fname+'.csv',skip=11)
        if np.sum((csv[:,4]) == 47) == csv.shape[0]:
            csv = clean_csv(csv,[1,2,3,5])
        else:
            print('!!!! PROBLEM !!!!')
            print('There is other data than discharge in there...')


        # Add time variable as first column in level_csv
        discharge_csv = np.zeros((csv.shape[0],csv.shape[1]+1))*np.nan
        discharge_csv[:,1:] = csv

        for it_csv in range(discharge_csv.shape[0]):
            discharge_csv[it_csv,0] = (dt.date(int(discharge_csv[it_csv,1]),int(discharge_csv[it_csv,2]),int(discharge_csv[it_csv,3]))-date_ref).days


        # Average all data for the same day into a daily mean variable and
        # put in new_level matrix at the same corresponding day/time.
        new_discharge = np.zeros((ndays,2))*np.nan

        for it in range(14976,ndays): # start only in 2021
            t = time[it]
            # date = date_ref+dt.timedelta(days=int(t))

            if t in discharge_csv[:,0]:
                mean_discharge = np.nanmean(discharge_csv[np.where(discharge_csv[:,0] == t)][:,-1])
                new_discharge[it,0] = t
                new_discharge[it,1] = mean_discharge


        # Put all new_level data into updated_level matrix
        updated_discharge = past_discharge.copy()
        updated_discharge[~np.isnan(new_discharge[:,1])] = new_discharge[~np.isnan(new_discharge[:,1])]


        # Save updated data
        if save:
            np.savez(processed_fp+'water_levels_discharge_'+loc_name,
                      discharge=updated_discharge,date_ref=date_ref)

        return updated_discharge

    else:
        print('!!!! PROBLEM !!!!')
        print('--> File to update water discharge up to selected date does not exist...')
        print('--> DISCHARGE NOT UPDATED - CHOOSE ANOTHER UPDATE DATE')




###
def update_water_level(date_update,loc_name,loc_nb,raw_fp,processed_fp,save=False):
    # raw_fp = '../../data/raw/QL_ECCC/'
    # processed_fp = '../../data/processed/water_levels_discharge_ECCC/'
    # loc_name = 'PointeClaire'
    # loc_nb = '02OA039'
    # # date_update = dt.date.today().strftime("%b")+'-'+str(dt.date.today().day)+'-'+str(dt.date.today().year)
    # date_update = 'Nov-11-2021'
    # save = False

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    ndays = (date_end-date_start).days + 1

    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

    # Load previous data
    past_data = np.load(processed_fp+'water_levels_discharge_'+loc_name+'.npz')
    past_level = past_data['level']


    # Load new csv data
    fname = loc_nb+'_HG_'+ date_update
    if os.path.isfile(raw_fp+fname+'.csv'):


        csv = read_csv(raw_fp+fname+'.csv',skip=10)
        if np.sum((csv[:,4]) == 46) == csv.shape[0]:
            csv = clean_csv(csv,[1,2,3,5])
        else:
            print('!!!! PROBLEM !!!!')
            print('There is other data than water level in there...')


        # Add time variable as first column in level_csv
        level_csv = np.zeros((csv.shape[0],csv.shape[1]+1))*np.nan
        level_csv[:,1:] = csv

        for it_csv in range(level_csv.shape[0]):
            level_csv[it_csv,0] = (dt.date(int(level_csv[it_csv,1]),int(level_csv[it_csv,2]),int(level_csv[it_csv,3]))-date_ref).days


        # Average all data for the same day into a daily mean variable and
        # put in new_level matrix at the same corresponding day/time.
        new_level = np.zeros((ndays,2))*np.nan

        for it in range(14976,ndays): # start only in 2021
            t = time[it]
            # date = date_ref+dt.timedelta(days=int(t))

            if t in level_csv[:,0]:
                mean_level = np.nanmean(level_csv[np.where(level_csv[:,0] == t)][:,-1])
                new_level[it,0] = t
                new_level[it,1] = mean_level


        # Put all new_level data into updated_level matrix
        updated_level = past_level.copy()
        updated_level[~np.isnan(new_level[:,1])] = new_level[~np.isnan(new_level[:,1])]


        # Save updated data
        if save:
            np.savez(processed_fp+'water_levels_discharge_'+loc_name,
                      level=updated_level,date_ref=date_ref)

        return updated_level

    else:
        print('!!!! PROBLEM !!!!')
        print('--> File to update water level up to selected date does not exist...')
        print('--> WATER LEVEL NOT UPDATED - CHOOSE ANOTHER UPDATE DATE')

#%%

###
def download_era5(features, region, output_dir = os.getcwd(), start_year=1991, end_year=2020, start_month=1, end_month=12, start_day=1, end_day=31):
    """
    Download ERA5 files

    :param path: String. Full directory to download files to
    :param start_year: Integer. Start year in YYYY.
    :param end_year: Integer. End year in YYYY.
    :return: None
    """

    base = "ERA5_"
    url = "https://cds.climate.copernicus.eu/api/v2"
    key = "68986:cd4929b1-ca5d-4f2b-b884-4d89b243703c" # CDS creditionals key

    if region == 'A':
        toplat, leftlon, downlat,rightlon = 46.00, -75.25, 44.75, -73.25
    elif region == 'B':
        toplat, leftlon, downlat,rightlon = 44.50, -77.25, 43.25, -75.50
    elif region == 'C':
        toplat, leftlon, downlat,rightlon = 46.00, -77.25, 44.75, -73.25
    elif region == 'D':
        toplat, leftlon, downlat,rightlon = 46.00, -77.25, 43.25, -73.25
    else:
        toplat, leftlon, downlat,rightlon = 53.00, -93.00, 40.00, -58.00

    day_list = []
    for iday in range(start_day,end_day+1):
        day_list += [str(iday).rjust(2, '0')]

    for year in range(start_year, end_year+1):
        print(year)
        os.chdir(output_dir)

        for month in range(start_month, end_month+1):
            os.chdir(output_dir)

            print(month)
            # '01' instead of '1'
            month = str(month).rjust(2, '0')

            # eg. 1979-01
            subdirectory = "{}-{}".format(year, month)
            if not os.path.isdir(subdirectory):
                os.mkdir(subdirectory)

            os.chdir(subdirectory)

            # _197901.nc
            if (start_day == 1) & (end_day == 31):
                extension = "_{}{}.nc".format(year, str(month).rjust(2, '0'))
            else:
                extension = "_{}{}{}_{}{}{}.nc".format(year, str(month).rjust(2, '0'),str(start_day).rjust(2, '0'),year, str(month).rjust(2, '0'),str(end_day).rjust(2, '0'))

            for feature in features:
                print(feature)

                # eg. ERA5_10m_u_component_of_wind_197901.nc
                filename = base + feature + extension

                if not os.path.isfile(filename):
                    print("Downloading file {}".format(filename))

                    downloaded = False

                    while not downloaded:
                        try:
                            client = cdsapi.Client(url=url, key=key, retry_max=5)
                            client.retrieve(
                                'reanalysis-era5-single-levels',
                                {
                                    'product_type': 'reanalysis',
                                    'format': 'netcdf',
                                    'variable': feature,
                                    'area': [
                                        toplat, leftlon, downlat,
                                        rightlon,
                                    ],
                                    'time': [
                                        '00:00','01:00', '02:00','03:00',
                                        '04:00','05:00', '06:00','07:00',
                                        '08:00','09:00', '10:00','11:00',
                                        '12:00','13:00', '14:00','15:00',
                                        '16:00','17:00', '18:00','19:00',
                                        '20:00','21:00', '22:00','23:00',
                                    ],
                                    # 'day': [
                                    #     '01', '02', '03',
                                    #     '04', '05', '06',
                                    #     '07', '08', '09',
                                    #     '10', '11', '12',
                                    #     '13', '14', '15',
                                    #     '16', '17', '18',
                                    #     '19', '20', '21',
                                    #     '22', '23', '24',
                                    #     '25', '26', '27',
                                    #     '28', '29', '30', '31'
                                    # ],
                                    'day': day_list,

                                    # API ignores cases where there are less than 31 days
                                    'month': month,
                                    'year': str(year)
                                },
                                filename)

                        except Exception as e:
                            print(e)

                            # Delete the partially downloaded file.
                            if os.path.isfile(filename):
                                os.remove(filename)

                        else:
                            # no exception implies download was complete
                            downloaded = True


###
def update_ERA5_var(var,var_type,processed_varname,region,update_startdate,update_enddate,raw_fp,processed_fp,time_freq=24,save=False):

    if (var == 'windspeed') | (var == 'RH') | (var == 'SH') | (var == 'FDD') | (var == 'TDD'):

        # Load past data
        past_data = np.load(processed_fp+'ERA5_dailymean_'+var+'.npz')['data']

        # Load updated base variables
        u10 = np.load(processed_fp+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
        v10 = np.load(processed_fp+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
        avg_Ta = np.load(processed_fp+'ERA5_dailymean_2m_temperature.npz')['data']
        avg_Td = np.load(processed_fp+'ERA5_dailymean_2m_dewpoint_temperature.npz')['data']
        slp = np.load(processed_fp+'ERA5_dailymean_mean_sea_level_pressure.npz')['data']

        # Define new variables
        windspeed = np.sqrt(u10**2 + v10**2)
        e_sat =0.61094*np.exp((17.625*avg_Ta)/(avg_Ta +243.04)) # Saturation vapor pressure (August–Roche–Magnus formula)
        avg_SH = 0.622*e_sat/(slp-0.378*e_sat) # Daily avg. specific humidity
        avg_RH = (np.exp((17.625*avg_Td)/(243.04+avg_Td))/np.exp((17.625*avg_Ta)/(243.04+avg_Ta))) # Daily avg. relative humidity

        mask_FDD = (avg_Ta <= 0)
        FDD = avg_Ta.copy()
        FDD[~mask_FDD] = np.nan

        mask_TDD = (avg_Ta > 0)
        TDD = avg_Ta.copy()
        TDD[~mask_TDD] = np.nan

        if (var == 'windspeed'): new_data=windspeed
        if (var == 'RH'): new_data=avg_RH
        if (var == 'SH'): new_data=avg_SH
        if (var == 'FDD'): new_data=FDD
        if (var == 'TDD'): new_data=TDD

        # Update variables
        updated_data = past_data.copy()
        updated_data[~np.isnan(new_data)] = new_data[~np.isnan(new_data)]

        if save:
            # Save new variables
            np.savez(processed_fp+'ERA5_dailymean_'+var,data=updated_data)

        return updated_data


    else:

        date_ref = dt.date(1900,1,1)
        date_start = dt.date(1980,1,1)
        date_end = dt.date(2021,12,31)

        time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

        # Load previous data
        past_data = np.load(processed_fp+'ERA5_'+var_type+'_'+processed_varname+'.npz')
        past_data = past_data['data']

        # Download new ERA5 file
        download_era5([processed_varname],region,
                      output_dir=raw_fp,
                      start_year=update_startdate.year,
                      end_year=update_enddate.year,
                      start_month=update_startdate.month,
                      end_month=update_enddate.month,
                      start_day=update_startdate.day,
                      end_day=update_enddate.day)

        # Load newly downloaded data
        if (update_startdate.day == 1) & (update_enddate.day == 31):
            fname = 'ERA5_{}_{}{}.nc'.format(processed_varname, str(update_startdate.year), str(update_startdate.month).rjust(2, '0'))
        else:
            fname = 'ERA5_{}_{}{}{}_{}{}{}.nc'.format(processed_varname,str(update_startdate.year),str(update_startdate.month).rjust(2, '0'),str(update_startdate.day).rjust(2, '0'),str(update_enddate.year),str(update_enddate.month).rjust(2, '0'),str(update_enddate.day).rjust(2, '0'))

        fpath = raw_fp+str(update_startdate.year)+'-'+str(update_startdate.month).rjust(2, '0')+'/'+fname
        if os.path.isfile(fpath):
            ncid = Dataset(fpath, 'r')
            ncid.set_auto_mask(False)
            time_tmp = ncid.variables['time'][:]
            time_tmp = time_tmp[0:time_tmp.size:time_freq]
            ncid.close()

            if region == 'A':
                # REGION A: ST_LAWRENCE
                rlon1, rlat1 = -75.25, 44.75
                rlon2, rlat2 = -73.25, 46.00
            elif region == 'B':
                # REGION B: LAKE ONTARIO
                rlon1, rlat1 = -77.25, 43.25
                rlon2, rlat2 = -75.50, 44.50
            elif region == 'C':
                # REGION C: OTTAWA RIVER + MONTREAL
                rlon1, rlat1 = -77.25, 44.75
                rlon2, rlat2 = -73.25, 46.00
            elif region == 'D':
                # REGION D: ALL
                rlon1, rlat1 = -77.25, 43.25
                rlon2, rlat2 = -73.25, 46.00
            elif region == 'E':
                # REGION E: OTTAWA RIVER ONLY
                rlon1, rlat1 = -77.25, 44.75
                rlon2, rlat2 = -75.25, 46.00

            # Average var in the selected region and for the correct temporal representation
            if var_type == 'dailymean':
                vdaily_new = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymean(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=fpath)))), returnArray = var))
                if len(vdaily_new.shape) > 1:
                    # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                    # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                    # equal to 1 for ERA5 and equal to 5 for ERA5T.
                    # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                    vdaily_new = np.nanmean(vdaily_new,axis=1)

            elif var_type == 'dailymin':
                vdaily_new = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymin(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=fpath)))), returnArray = var))
                if len(vdaily_new.shape) > 1:
                    # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                    # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                    # equal to 1 for ERA5 and equal to 5 for ERA5T.
                    # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                    vdaily_new = np.nanmean(vdaily_new,axis=1)

            elif var_type == 'dailymax':
                vdaily_new = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymax(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=fpath)))), returnArray = var))
                if len(vdaily_new.shape) > 1:
                    # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                    # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                    # equal to 1 for ERA5 and equal to 5 for ERA5T.
                    # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                    vdaily_new = np.nanmean(vdaily_new,axis=1)


            elif var_type == 'dailysum':
                if (update_startdate.year == update_enddate.year) & (update_startdate.month == update_enddate.month) & (update_startdate.day == update_enddate.day):
                    # There is only one day in the update file...
                    ncid = Dataset(fpath, 'r')
                    ncid.set_auto_mask(False)
                    var_nc1 = ncid.variables[var][:]
                    var_nc1[var_nc1 < 0] = 0
                    if len(var_nc1.shape) > 3:
                        var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))

                    vdaily_new = np.nansum(var_nc1[1:,:,:],axis=0)
                    vdaily_new = np.nanmean(np.nanmean(vdaily_new,axis=2), axis=1)

                else:
                    ncid = Dataset(fpath, 'r')
                    ncid.set_auto_mask(False)
                    var_nc1 = ncid.variables[var][:]
                    var_nc1[var_nc1 < 0] = 0
                    if len(var_nc1.shape) > 3:
                        var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))

                    vdaily_new = np.zeros((int(var_nc1.shape[0]/time_freq),var_nc1.shape[1],var_nc1.shape[2]))
                    for iday in range(int(var_nc1.shape[0]/time_freq)-1):
                            vdaily_new[iday,:,:] = np.nansum(var_nc1[1+iday*time_freq:1+(iday+1)*time_freq,:,:],axis=0)
                    vdaily_new[-1,:,:] = np.nansum(var_nc1[1+(iday+1)*time_freq:1+(iday+1+1)*time_freq,:,:],axis=0)
                    vdaily_new = np.nanmean(np.nanmean(vdaily_new,axis=2), axis=1)


            # Correct units:
            if var == 'msl':
                vdaily_new = vdaily_new/1000. # Convert to kPa
            if (var == 't2m') | (var == 'd2m') | (var == 'lict') | (var == 'ltlt'):
                vdaily_new  = (vdaily_new-273.15)# Convert Kelvins to Celsius

            # Then, arrange variables in the same format as weather from NCEI (i.e. [it,var])
            new_data = np.zeros((len(time)))*np.nan

            for it in range(time_tmp.size):
                date_it = date_ref+dt.timedelta(hours=int(time_tmp[it]))
                new_time = (date_it - date_ref).days
                if (new_time <= time[-1]) & (new_time >= time[0]):
                    itvar = np.where(time == int(new_time))[0][0]
                    new_data[itvar] = vdaily_new[it]

            cdo.cleanTempDir()

            # Put all new_data data into updated_data matrix
            updated_data = past_data.copy()
            updated_data[~np.isnan(new_data)] = new_data[~np.isnan(new_data)]

            # Save updated data
            if save:
                savename = 'ERA5_{}_{}'.format(var_type, processed_varname)
                np.savez(processed_fp+savename,
                         data = updated_data)

            return updated_data

        else:
            print('!!!! PROBLEM !!!!')
            print('--> File to update data for selected dates does not exist...')
            print('--> DATA NOT UPDATED - CHOOSE ANOTHER UPDATE DATE')


###
def load_weather_vars_ERA5(fpath_ERA5_processed,varlist,region,time):

    weather_vars = np.zeros((len(time),len(varlist)))*np.nan
    varnames = []
    for ivar,var_s in enumerate(varlist):
        if (var_s == 'dailymean_windspeed') | (var_s == 'dailymean_TDD') | (var_s == 'dailymean_FDD') | (var_s == 'dailymean_RH'):
            data = np.load(fpath_ERA5_processed+'uERA5_'+var_s+'.npz')['data']

            if var_s == 'dailymean_TDD': var_n = 'Tot. TDD'
            if var_s == 'dailymean_FDD': var_n = 'Tot. FDD'
            if var_s == 'dailymean_windspeed': var_n = 'Avg. windspeed'
            if var_s == 'dailymean_RH': var_n = 'Avg. rel. hum.'

        else:
            if (var_s == 'daily_theta_wind'):
                datau = np.load(fpath_ERA5_processed+'uERA5_dailymean_10m_u_component_of_wind.npz')['data']
                datau = np.squeeze(datau)
                datav = np.load(fpath_ERA5_processed+'uERA5_dailymean_10m_v_component_of_wind.npz')['data']
                datav = np.squeeze(datav)

                data = np.mod( 180+ (180/np.pi)*np.arctan2(datav,datau), 360)
                var_n = 'Avg. wind direction'
            else:
                data = np.load(fpath_ERA5_processed+'uERA5_'+var_s+'.npz')['data']

            if var_s == 'dailymean_FDD': data *= -1

            if var_s == 'dailymax_2m_temperature' : var_n = 'Avg. Ta_max'
            if var_s == 'dailymin_2m_temperature' : var_n = 'Avg. Ta_min'
            if var_s == 'dailymean_2m_temperature' : var_n = 'Avg. Ta_mean'
            if var_s == 'dailymean_2m_dewpoint_temperature' : var_n = 'Avg. Td_mean'
            if var_s == 'dailymean_total_precipitation': var_n = 'Avg. precip.'
            if var_s == 'dailysum_total_precipitation': var_n = 'Tot. precip.'
            if var_s == 'dailymean_mean_sea_level_pressure': var_n = 'Avg. SLP'
            if var_s == 'dailymean_10m_u_component_of_wind': var_n = 'Avg. u-wind'
            if var_s == 'dailymean_10m_v_component_of_wind': var_n = 'Avg. v-wind'
            if var_s == 'dailymean_total_cloud_cover': var_n = 'Avg. cloud cover'
            if var_s == 'dailymean_snowfall': var_n = 'Avg. snowfall'
            if var_s == 'dailysum_snowfall': var_n = 'Tot. snowfall'
            if var_s == 'dailymean_snowmelt': var_n = 'Avg. snowmelt'
            if var_s == 'dailysum_snowmelt': var_n = 'Tot. snowmelt'
            if var_s == 'dailymean_runoff': var_n = 'Avg. runoff'
            if var_s == 'dailysum_runoff': var_n = 'Tot. runoff'
            if var_s == 'dailymean_surface_solar_radiation_downwards': var_n = 'Avg. SW down (sfc)'
            if var_s == 'dailymean_surface_latent_heat_flux': var_n = 'Avg. LH (sfc)'
            if var_s == 'dailymean_surface_sensible_heat_flux': var_n = 'Avg. SH (sfc)'

            data= data[365:]
            data = np.squeeze(data)


        weather_vars[:,ivar] = data
        varnames += [var_n]

    return weather_vars, varnames

#%%

###
def update_monthly_NAO_index(fpath_processed,url='https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table',save=False):

    # Load new data from URL
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    soup_text = soup.get_text()
    lines = soup_text.splitlines()

    NAO_tmp = np.zeros((len(lines),13))*np.nan

    for il,l in enumerate(lines):
        entries = np.array([float(entry) for entry in lines[il].split("  ")]).T
        NAO_tmp[il,0:entries.shape[0]] = entries

    # Rearrange data in same format as previous data
    years_tmp = np.arange(1950,2022)
    data_tmp = np.zeros((len(years_tmp)*12,3))

    for il,l in enumerate(lines):
        data_tmp[il*12:(il+1)*12,0]= NAO_tmp[il,0]
        for imonth in range(0,12):
            data_tmp[il*12+imonth,1]= imonth+1
            data_tmp[il*12+imonth,2]= NAO_tmp[il,imonth+1]

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
    years = np.arange(1991,2022)

    new_data = np.zeros((len(time),1))*np.nan
    for it in range(len(time)):
        date_it = date_ref + dt.timedelta(days=int(time[it]))
        if date_it.year in years:
            NAO_year = data_tmp[np.where(data_tmp[:,0] == date_it.year)]
            NAO_month = NAO_year[np.where(NAO_year[:,1] == date_it.month)][0]
            new_data[it,0] = NAO_month[2]

    # Load past data
    # fpath_processed = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/NAO_index_NOAA/'
    past_data = np.load(fpath_processed+'NAO_index_NOAA_monthly.npz')
    past_data = past_data['data']

    # Put new data in updated data
    updated_data = past_data.copy()
    updated_data[~np.isnan(new_data)] = new_data[~np.isnan(new_data)]

    # Save updated data
    if save:
        savename = 'NAO_index_NOAA_monthly'
        np.savez(fpath_processed+savename,
                 data = updated_data)

    return updated_data


###
def update_daily_NAO_index(fpath_raw,fpath_processed,url='ftp://ftp.cpc.ncep.noaa.gov/cwlinks//para/norm.daily.nao.cdas.z500.19500101_current.csv',save=False):

    import shutil
    import urllib.request as request
    from contextlib import closing

    with closing(request.urlopen(url)) as r:
        with open(fpath_raw+'nao_index_cdas.csv', 'wb') as f:
            shutil.copyfileobj(r, f)

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

    NAO_daily = read_csv(fpath_raw+'nao_index_cdas.csv',skip=1)
    updated_data = np.zeros((len(time),1))*np.nan

    for it in range(NAO_daily.shape[0]):

        date_time = (dt.date(int(NAO_daily[it,0]),int(NAO_daily[it,1]),int(NAO_daily[it,2]))-date_ref).days

        if date_time in time:
            time_it = np.where(time == date_time)[0][0]
            updated_data[time_it,0] = NAO_daily[it,3]

    if save:
        savename = 'NAO_daily'
        np.savez(fpath_processed+savename,data=updated_data)

    return updated_data

#%%==============================================================
# MLR Model & Analysis

###
def freezeup_multiple_linear_regression_model(df,x_model,nyears,training_size,anomaly = False,rolling_training = False, show = False, verbose = True):

    if rolling_training:
        yh = np.zeros((nyears-training_size,training_size))*np.nan
        yh_hat = np.zeros((nyears-training_size,training_size))*np.nan
        xh = np.zeros((nyears-training_size,training_size,len(x_model)))*np.nan
        xh_plot = np.zeros((nyears-training_size,training_size))*np.nan
        models_h = np.zeros((nyears-training_size),dtype='object')*np.nan

        yf = np.zeros((nyears-training_size,1))*np.nan
        yf_hat = np.zeros((nyears-training_size,1))*np.nan
        xf = np.zeros((nyears-training_size,1))*np.nan

        for n in range(nyears-training_size):
            i = 0+n
            f = i+training_size

            df_h = df[i:f] # Hindcast Data: past 15 years
            df_f = df[f:f+1]# Forecast Data: next year


            # Hindcast
            xh_m = df_h[x_model]
            xh_plot[n,:] = df_h['Year']
            yh_m = df_h['Freeze-up'+' Anomaly'*anomaly]
            xh[n,:,:] = df_h[x_model]
            yh[n,:] = np.array(df_h['Freeze-up'+' Anomaly'*anomaly])

            xh_m = sm.add_constant(xh_m, has_constant='add') # adding a constant
            model = sm.OLS(yh_m, xh_m, missing='drop').fit()
            models_h[n] = model
            yh_hat[n,:] = model.predict(xh_m)

            # if verbose:
            #     print_model = model.summary()
            #     print(print_model)

            # Forecast
            xf[n,:] = np.array(df_f[['Year']])
            xf_m = df_f[x_model]
            yf[n,:] = df_f['Freeze-up'+' Anomaly'*anomaly]


            xf_m = sm.add_constant(xf_m, has_constant='add') # adding a constant
            yf_hat[n,:] = model.predict(xf_m)


    else:

        df_h = df[0:training_size]
        df_f = df[training_size:]

        xh = df_h[x_model]
        xf = df_f[x_model]

        yh = df_h['Freeze-up'+' Anomaly'*anomaly]
        yf = df_f['Freeze-up'+' Anomaly'*anomaly]

        # Hindcast
        xh = sm.add_constant(xh) # adding a constant
        model = sm.OLS(yh, xh, missing='drop').fit()
        yh_hat = model.predict(xh)

        # if verbose:
        #     print_model = model.summary()
        #     print(print_model)

        # Forecast
        xf = sm.add_constant(xf) # adding a constant
        yf_hat = model.predict(xf)



    # Evaluate Hindcast and Forecast:
    if show:
        fig, ax = plt.subplots()
        ax.plot(np.array(df['Year']),np.array(df['Freeze-up'+' Anomaly'*anomaly]),'o-')

    if not rolling_training:
        std_h = np.nanstd(yh-yh_hat)
        std_f = np.nanstd(yf-yf_hat)

        mae_h = np.nanmean(np.abs(yh-yh_hat))
        mae_f = np.nanmean(np.abs(yf-yf_hat))

        rmse_h = np.sqrt(np.nanmean((yh-yh_hat)**2.))
        rmse_f = np.sqrt(np.nanmean((yf-yf_hat)**2.))

        _, Rsqr_h = linear_fit(yh,yh_hat)
        _, Rsqr_f = linear_fit(yf,yf_hat)

        Rsqr_adj_h = 1-(((1-Rsqr_h)*(yh.shape[0]-1))/(yh.shape[0]-model.df_model-1))
        Rsqr_adj_f = 1-(((1-Rsqr_f)*(yf.shape[0]-1))/(yf.shape[0]-model.df_model-1))

        if verbose:
            print('------------------------------------')
            # print('Hindcast: 1992-2006')
            # print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
            # print(mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h)
            # print('')
            print('Forecast: 2007-2016')
            print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
            print(mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f)
            print('------------------------------------')

        if show:
            ax.plot(df_f['Year'],yf_hat, 'o--', color= plt.get_cmap('tab20')(4))
            ax.plot(np.array(df_h['Year']),yh_hat, '-', color= plt.get_cmap('tab20')(2))
            ax.set_xlabel('Year')
            ax.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))


        return model,np.array(df['Year']),np.array(df['Freeze-up'+' Anomaly'*anomaly]),df_f['Year'],yf,yf_hat,np.array(df_h['Year']),yh_hat,mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h,mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f

    else:

        std_h = np.nanstd(yh-yh_hat,axis=1)
        std_f = np.nanstd(yf-yf_hat)

        mae_h = np.nanmean(np.abs(yh-yh_hat),axis=1)
        mae_f = np.nanmean(np.abs(yf-yf_hat))

        rmse_h = np.sqrt(np.nanmean((yh-yh_hat)**2.,axis=1))
        rmse_f = np.sqrt(np.nanmean((yf-yf_hat)**2.))

        Rsqr_h = np.zeros((yh.shape[0]))*np.nan
        for m in range(yh.shape[0]):
            _, Rsqr_h[m] = linear_fit(np.squeeze(yh[m,:]),np.squeeze(yh_hat[m,:]))
        _, Rsqr_f = linear_fit(np.squeeze(yf),np.squeeze(yf_hat))

        Rsqr_adj_h = 1-(((1-Rsqr_h)*(yh.shape[1]-1))/(yh.shape[1]-model.df_model-1))
        Rsqr_adj_f = 1-(((1-Rsqr_f)*(yf.shape[0]-1))/(yf.shape[0]-model.df_model-1))

        if verbose:
            print('------------------------------------')
            # print('First Hindcast: 1992-2006')
            # print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
            # print(mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h)
            # print('')
            print('Forecast: 2007-2016')
            print('MAE,    RMSE,    Rsqr,    Rsqr_adj, sigm_err')
            print(mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f)
            print('------------------------------------')

        if show:
            ax.plot(xf[:,0],yf_hat, 'o--', color= plt.get_cmap('tab20')(4))
            ax.plot(np.array(df['Year'][0:training_size]),yh_hat[0,:], '-', color= plt.get_cmap('tab20')(2))
            ax.set_xlabel('Year')
            ax.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))

        # return model,np.array(df['Year']),np.array(df['Freeze-up'+' Anomaly'*anomaly]),xf[:,0],yf,yf_hat,np.array(df['Year'][0:training_size]),yh_hat[0,:],mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h,mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f
        return models_h,np.array(df['Year']),np.array(df['Freeze-up'+' Anomaly'*anomaly]),xf[:,0],yf,yf_hat,xh_plot,xh,yh_hat,mae_h, rmse_h, Rsqr_h, Rsqr_adj_h, std_h,mae_f, rmse_f, Rsqr_f, Rsqr_adj_f, std_f


###
def find_models_MLR(npred,combinations,df,columns,avg_freezeup_doy,training_size,nyears,rolling_training,rlim,ralim,fsave,anomaly,station_labels,freezeup_opt,deseasonalize,detrend):
    models_h = []
    x_models = []
    mae_f_models = []
    rmse_f_models = []
    Rsqr_f_models = []
    Rsqr_adj_f_models = []
    std_f_models = []
    mae_h_models = []
    rmse_h_models = []
    Rsqr_h_models = []
    Rsqr_adj_h_models = []
    Rsqr_pred_h_models = []
    std_h_models = []
    xf_models = []
    yf_models = []
    xh_plot_models = []
    xh_models = []
    yh_models = []
    yall_models = []
    xall_models = []
    yh_true_models = []

    for i in range(len(combinations)):
        x_model = [ columns[3:][c] for c in combinations[i] ]

        [model_h,xall,yall,xf,yf_true,yf,xh_plot,xh,yh,
        mae_h,rmse_h,Rsqr_h,Rsqr_adj_h,std_h,
        mae_f,rmse_f,Rsqr_f,Rsqr_adj_f, std_f]  = freezeup_multiple_linear_regression_model(df[['Year','Freeze-up','Freeze-up Anomaly']+x_model],x_model,nyears,training_size,anomaly = anomaly,rolling_training=rolling_training,verbose=False)

        if (Rsqr_f >= rlim) & (Rsqr_adj_f >= ralim):
            models_h += [model_h]
            x_models += [x_model]
            mae_f_models += [mae_f]
            rmse_f_models += [rmse_f]
            Rsqr_f_models += [Rsqr_f]
            Rsqr_adj_f_models += [Rsqr_adj_f]
            std_f_models += [std_f]
            mae_h_models += [mae_h]
            rmse_h_models += [rmse_h]
            Rsqr_h_models += [Rsqr_h]
            Rsqr_adj_h_models += [Rsqr_adj_h]
            std_h_models += [std_h]
            xf_models += [xf]
            yf_models += [yf]
            xh_models += [xh]
            xh_plot_models += [xh_plot]
            yh_models += [yh]
            yall_models += [yall]
            xall_models += [xall]

            yh_true = []
            for h in range(nyears-training_size):
                yh_true += [avg_freezeup_doy[h:h+training_size]]

            yh_true = np.array(yh_true)
            yh_true_models += [yh_true]

            # rsqr_pred = []
            # for h in range(nyears-training_size):
            #     rsqr_pred += predicted_r2(np.squeeze(yh_true[h]),np.squeeze(yh[h]),np.squeeze(xh[h]))
            # Rsqr_pred_h_models += [rsqr_pred]

            r_tmp = np.zeros(yh_models[0].shape[0])*np.nan
            for h in range(nyears-training_size):
                try:
                    r_tmp[h] = predicted_r2(np.squeeze(yh_true[h][:]),np.squeeze(yh[h,:]),np.squeeze(xh[h,:]))
                except Exception as e:
                    print(e)
                    r_tmp[h] = np.nan
                else:
                    r_tmp[h] = r_tmp[h]
            Rsqr_pred_h_models += [r_tmp]

        if (i/(round(len(combinations)/10)*10))%(0.1) == 0:
            print(str(i)+'/'+str(len(combinations)))
            np.savez(fsave+'models'+str(npred)+'_it'+str(i)+'_ts'+str(training_size),
                        models_h = models_h,
                        x_models = x_models,
                        mae_f_models = mae_f_models,
                        rmse_f_models = rmse_f_models,
                        Rsqr_f_models = Rsqr_f_models,
                        Rsqr_adj_f_models = Rsqr_adj_f_models,
                        std_f_models = std_f_models,
                        mae_h_models = mae_h_models,
                        rmse_h_models = rmse_h_models,
                        Rsqr_h_models =  Rsqr_h_models,
                        Rsqr_adj_h_models = Rsqr_adj_h_models,
                        Rsqr_pred_h_models = Rsqr_pred_h_models,
                        std_h_models = std_h_models,
                        xf_models = xf_models,
                        yf_models = yf_models,
                        xh_models = xh_models,
                        xh_plot_models = xh_plot_models,
                        yh_models = yh_models,
                        yh_true_models = yh_true_models,
                        yall = yall_models,
                        xall = xall_models,
                        station_labels = station_labels,
                        freezeup_opt = freezeup_opt,
                        deseasonalize = deseasonalize ,
                        detrend = detrend,
                        training_size = training_size,
                        rolling_training = rolling_training,
                        anomaly = anomaly,
                        rlim = rlim,
                        ralim = ralim,
                      )


    np.savez(fsave+'models'+str(npred)+'_ts'+str(training_size),
                models_h = models_h,
                x_models = x_models,
                mae_f_models = mae_f_models,
                rmse_f_models = rmse_f_models,
                Rsqr_f_models = Rsqr_f_models,
                Rsqr_adj_f_models = Rsqr_adj_f_models,
                std_f_models = std_f_models,
                mae_h_models = mae_h_models,
                rmse_h_models = rmse_h_models,
                Rsqr_h_models =  Rsqr_h_models,
                Rsqr_adj_h_models = Rsqr_adj_h_models,
                Rsqr_pred_h_models = Rsqr_pred_h_models,
                std_h_models = std_h_models,
                xf_models = xf_models,
                yf_models = yf_models,
                xh_models = xh_models,
                xh_plot_models = xh_plot_models,
                yh_models = yh_models,
                yh_true_models = yh_true_models,
                yall = yall_models,
                xall = xall_models,
                station_labels = station_labels,
                freezeup_opt = freezeup_opt,
                deseasonalize = deseasonalize ,
                detrend = detrend,
                training_size = training_size,
                rolling_training = rolling_training,
                anomaly = anomaly,
                rlim = rlim,
                ralim = ralim,
              )


    models_h_Nov1 = []
    x_models_Nov1 = []
    mae_f_models_Nov1 = []
    rmse_f_models_Nov1 = []
    Rsqr_f_models_Nov1 = []
    Rsqr_adj_f_models_Nov1 = []
    std_f_models_Nov1 = []
    mae_h_models_Nov1 = []
    rmse_h_models_Nov1 = []
    Rsqr_h_models_Nov1 = []
    Rsqr_adj_h_models_Nov1 = []
    Rsqr_pred_h_models_Nov1 = []
    std_h_models_Nov1 = []
    xf_models_Nov1 = []
    yf_models_Nov1 = []
    xh_plot_models_Nov1 = []
    xh_models_Nov1 = []
    yh_models_Nov1 = []
    yall_models_Nov1 = []
    xall_models_Nov1 = []
    imodels_Nov1 = []
    yh_true_models_Nov1 = []

    for i in range(len(x_models)):
        xim = [x_models[i][m] for m in range(npred)]
        if (~np.any(['Apr. aFDD' in xim[j] for j in range(npred)]) ) & (~np.any(['Nov' in xim[j] for j in range(npred)])) & (~np.any(['Fall' in xim[j] for j in range(npred)])):
            x_tmp = np.squeeze(yall[training_size+1:])
            y_tmp = np.squeeze(yf_models[i][:])
            coeff,_ = linear_fit(x_tmp,y_tmp)
            if coeff[0] > 0 :
                imodels_Nov1 += [i]
                models_h_Nov1 += [models_h[i]]
                x_models_Nov1 += [x_models[i]]
                mae_f_models_Nov1 += [mae_f_models[i]]
                rmse_f_models_Nov1 += [rmse_f_models[i]]
                Rsqr_f_models_Nov1 += [Rsqr_f_models[i]]
                Rsqr_adj_f_models_Nov1 += [Rsqr_adj_f_models[i]]
                std_f_models_Nov1 += [std_f_models[i]]
                mae_h_models_Nov1 += [mae_h_models[i]]
                rmse_h_models_Nov1 += [rmse_h_models[i]]
                Rsqr_h_models_Nov1 += [Rsqr_h_models[i]]
                Rsqr_adj_h_models_Nov1 += [Rsqr_adj_h_models[i]]
                Rsqr_pred_h_models_Nov1 += [Rsqr_pred_h_models[i]]
                std_h_models_Nov1 += [std_h_models[i]]
                xf_models_Nov1 += [xf_models[i]]
                yf_models_Nov1 += [yf_models[i]]
                xh_plot_models_Nov1 += [xh_plot_models[i]]
                xh_models_Nov1 += [xh_models[i]]
                yh_models_Nov1 += [yh_models[i]]
                yall_models_Nov1 += [yall_models[i]]
                xall_models_Nov1 += [xall_models[i]]
                yh_true_models_Nov1 += [yh_true_models[i]]



    np.savez(fsave+'models'+str(npred)+'_ts'+str(training_size)+'_Nov1',
                models_h = models_h_Nov1,
                x_models = x_models_Nov1,
                mae_f_models = mae_f_models_Nov1,
                rmse_f_models = rmse_f_models_Nov1,
                Rsqr_f_models = Rsqr_f_models_Nov1,
                Rsqr_adj_f_models = Rsqr_adj_f_models_Nov1,
                std_f_models = std_f_models_Nov1,
                mae_h_models = mae_h_models_Nov1,
                rmse_h_models = rmse_h_models_Nov1,
                Rsqr_h_models =  Rsqr_h_models_Nov1,
                Rsqr_adj_h_models = Rsqr_adj_h_models_Nov1,
                Rsqr_pred_h_models = Rsqr_pred_h_models_Nov1,
                std_h_models = std_h_models_Nov1,
                xf_models = xf_models_Nov1,
                yf_models = yf_models_Nov1,
                xh_models = xh_models_Nov1,
                xh_plot_models = xh_plot_models_Nov1,
                yh_models = yh_models_Nov1,
                yh_true_models = yh_true_models_Nov1,
                yall = yall_models_Nov1,
                xall = xall_models_Nov1,
                station_labels = station_labels,
                freezeup_opt = freezeup_opt,
                deseasonalize = deseasonalize ,
                detrend = detrend,
                training_size = training_size,
                rolling_training = rolling_training,
                anomaly = anomaly,
                rlim = rlim,
                ralim = ralim,
              )

    models_h_Dec1 = []
    x_models_Dec1 = []
    mae_f_models_Dec1 = []
    rmse_f_models_Dec1 = []
    Rsqr_f_models_Dec1 = []
    Rsqr_adj_f_models_Dec1 = []
    std_f_models_Dec1 = []
    mae_h_models_Dec1 = []
    rmse_h_models_Dec1 = []
    Rsqr_h_models_Dec1 = []
    Rsqr_adj_h_models_Dec1 = []
    Rsqr_pred_h_models_Dec1 = []
    std_h_models_Dec1 = []
    xf_models_Dec1 = []
    yf_models_Dec1 = []
    xh_plot_models_Dec1 = []
    xh_models_Dec1 = []
    yh_models_Dec1 = []
    yall_models_Dec1 = []
    xall_models_Dec1 = []
    imodels_Dec1 = []
    yh_true_models_Dec1 = []

    for i in range(len(x_models)):
        xim = [x_models[i][m] for m in range(npred)]
        if (~np.any(['Apr. aFDD' in xim[j] for j in range(npred)]) ):
            x_tmp = np.squeeze(yall[training_size+1:])
            y_tmp = np.squeeze(yf_models[i][:])
            coeff,_ = linear_fit(x_tmp,y_tmp)
            if coeff[0] > 0 :
                imodels_Dec1 += [i]
                models_h_Dec1 += [models_h[i]]
                x_models_Dec1 += [x_models[i]]
                mae_f_models_Dec1 += [mae_f_models[i]]
                rmse_f_models_Dec1 += [rmse_f_models[i]]
                Rsqr_f_models_Dec1 += [Rsqr_f_models[i]]
                Rsqr_adj_f_models_Dec1 += [Rsqr_adj_f_models[i]]
                std_f_models_Dec1 += [std_f_models[i]]
                mae_h_models_Dec1 += [mae_h_models[i]]
                rmse_h_models_Dec1 += [rmse_h_models[i]]
                Rsqr_h_models_Dec1 += [Rsqr_h_models[i]]
                Rsqr_adj_h_models_Dec1 += [Rsqr_adj_h_models[i]]
                Rsqr_pred_h_models_Dec1 += [Rsqr_pred_h_models[i]]
                std_h_models_Dec1 += [std_h_models[i]]
                xf_models_Dec1 += [xf_models[i]]
                yf_models_Dec1 += [yf_models[i]]
                xh_plot_models_Dec1 += [xh_plot_models[i]]
                xh_models_Dec1 += [xh_models[i]]
                yh_models_Dec1 += [yh_models[i]]
                yall_models_Dec1 += [yall_models[i]]
                xall_models_Dec1 += [xall_models[i]]
                yh_true_models_Dec1 += [yh_true_models[i]]


    np.savez(fsave+'models'+str(npred)+'_ts'+str(training_size)+'_Dec1',
                models_h = models_h_Dec1,
                x_models = x_models_Dec1,
                mae_f_models = mae_f_models_Dec1,
                rmse_f_models = rmse_f_models_Dec1,
                Rsqr_f_models = Rsqr_f_models_Dec1,
                Rsqr_adj_f_models = Rsqr_adj_f_models_Dec1,
                std_f_models = std_f_models_Dec1,
                mae_h_models = mae_h_models_Dec1,
                rmse_h_models = rmse_h_models_Dec1,
                Rsqr_h_models =  Rsqr_h_models_Dec1,
                Rsqr_adj_h_models = Rsqr_adj_h_models_Dec1,
                Rsqr_pred_h_models = Rsqr_pred_h_models_Dec1,
                std_h_models = std_h_models_Dec1,
                xf_models = xf_models_Dec1,
                yf_models = yf_models_Dec1,
                xh_models = xh_models_Dec1,
                xh_plot_models = xh_plot_models_Dec1,
                yh_models = yh_models_Dec1,
                yh_true_models = yh_true_models_Dec1,
                yall = yall_models_Dec1,
                xall = xall_models_Dec1,
                station_labels = station_labels,
                freezeup_opt = freezeup_opt,
                deseasonalize = deseasonalize ,
                detrend = detrend,
                training_size = training_size,
                rolling_training = rolling_training,
                anomaly = anomaly,
                rlim = rlim,
                ralim = ralim,
              )



###
def MLR_model_analysis(fpath,npred,start_date,training_size,xall,yall,show=True,anomaly=False):

    model_data = np.load(fpath+'models'+str(npred)+'_ts'+str(training_size)+'.npz',allow_pickle=True)
    models_startdate = np.load(fpath+'models'+str(npred)+'_'+start_date+'_ts'+str(training_size)+'.npz',allow_pickle=True)

    imodels = models_startdate['imodels_'+str(npred)+'_'+start_date]

    x_models = model_data['x_models_'+str(npred)]
    yh_true_models = model_data['yh_true_models_'+str(npred)]
    yh_models = model_data['yh_models_'+str(npred)]
    xh_models = model_data['xh_models_'+str(npred)]


    Rsqr_pred_h_models = []
    for im in imodels:
        r_tmp = np.zeros(yh_models[0].shape[0])*np.nan
        for h in range(yh_models[0].shape[0]):
            try:
                r_tmp[h] = predicted_r2(yh_true_models[h],np.squeeze(yh_models[im][h,:]),np.squeeze(xh_models[im][h,:]))
            except Exception as e:
                print(e)
                r_tmp[h] = np.nan
            else:
                r_tmp[h] = r_tmp[h]
        Rsqr_pred_h_models += [r_tmp]
    Rsqr_pred_h_models = np.array(Rsqr_pred_h_models)

    x_models = model_data['x_models_'+str(npred)][imodels]
    mae_f_models = model_data['mae_f_models_'+str(npred)][imodels]
    rmse_f_models = model_data['rmse_f_models_'+str(npred)][imodels]
    Rsqr_f_models = model_data['Rsqr_f_models_'+str(npred)][imodels]
    Rsqr_adj_f_models = model_data['Rsqr_adj_f_models_'+str(npred)][imodels]
    std_f_models = model_data['std_f_models_'+str(npred)][imodels]
    mae_h_models = model_data['mae_h_models_'+str(npred)][imodels]
    rmse_h_models = model_data['rmse_h_models_'+str(npred)][imodels]
    Rsqr_h_models =  model_data['Rsqr_h_models_'+str(npred)][imodels]
    Rsqr_adj_h_models = model_data['Rsqr_adj_h_models_'+str(npred)][imodels]
    std_h_models = model_data['std_h_models_'+str(npred)][imodels]
    xf_models = model_data['xf_models_'+str(npred)][imodels]
    yf_models = model_data['yf_models_'+str(npred)][imodels]
    xh_models = model_data['xh_models_'+str(npred)][imodels]
    yh_models = model_data['yh_models_'+str(npred)][imodels]


    pred_list = []
    for p in range(npred):
        pred_list += ['Pred. '+str(p+1)]


    arr1 = x_models
    arr2 = np.array([mae_f_models,rmse_f_models,Rsqr_f_models,Rsqr_adj_f_models,std_f_models]).T
    arr_c = np.array(np.zeros((arr1.shape[0],arr1.shape[1]+arr2.shape[1])), dtype=object)
    arr_c[:,0:arr1.shape[1]] = arr1
    arr_c[:,arr1.shape[1]:arr1.shape[1]+arr2.shape[1]] = arr2
    stats_f_all = pd.DataFrame(arr_c,
                      columns=pred_list+['MAE', 'RMSE','Rsqr','Rsqr_adj','sig_err'])

    arr1 = x_models
    arr2 = np.array([np.nanmean(mae_h_models,axis=1),np.nanmean(rmse_h_models,axis=1),np.nanmean(Rsqr_h_models,axis=1),np.nanmean(Rsqr_adj_h_models,axis=1),np.nanmean(Rsqr_pred_h_models,axis=1),np.nanmean(std_h_models,axis=1)]).T
    arr_c = np.array(np.zeros((arr1.shape[0],arr1.shape[1]+arr2.shape[1])), dtype=object)
    arr_c[:,0:arr1.shape[1]] = arr1
    arr_c[:,arr1.shape[1]:arr1.shape[1]+arr2.shape[1]] = arr2
    stats_h_all = pd.DataFrame(arr_c,
                      columns=pred_list+['MAE', 'RMSE','Rsqr','Rsqr_adj','Rsqr_pred','sig_err'])

    if show:
        fig, ax = plt.subplots()
        plt.title('Models '+start_date+' - '+str(npred)+' predictors')
        plt.ylim(-0.2,1)

        fig_FUD, ax_FUD = plt.subplots()
        plt.title('Models '+start_date+' - '+str(npred)+' predictors')
        ax_FUD.set_xlabel('Year')
        ax_FUD.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))

    p = 0
    arr_c_f = np.array(np.zeros((len(imodels),npred+5)), dtype=object)
    arr_c_h = np.array(np.zeros((len(imodels),npred+6)), dtype=object)

    for i,im in enumerate(imodels):
        # if (np.all(Rsqr_adj_h_models[i]>0):
        # if (np.all(Rsqr_adj_h_models[i] > 0)) & (np.all(Rsqr_h_models[i] >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.)):
        # if (np.all(Rsqr_adj_h_models[i] > 0)) & (np.nanmean(Rsqr_h_models[i]) >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.):
        # if (np.nanmean(Rsqr_h_models[i]) >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.):
        if (np.nanmean(Rsqr_adj_h_models[i]) > 0) & (np.nanmean(Rsqr_h_models[i]) >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.):
        # if np.all(Rsqr_pred_h_models[i]>0):
        # if np.all(Rsqr_adj_f_models[i]>0):
            # print(x_models[i])
            arr1_fi = x_models[i]
            arr2_fi = np.array([mae_f_models[i],rmse_f_models[i],Rsqr_f_models[i],Rsqr_adj_f_models[i],std_f_models[i]]).T
            arr_c_f[p,0:npred] = arr1_fi
            arr_c_f[p,npred:npred+5] = arr2_fi
            # stats_f_all = pd.DataFrame(arr_c,
            #                   columns=pred_list+['MAE', 'RMSE','Rsqr','Rsqr_adj','sig_err'])

            arr1_hi = x_models[i]
            arr2_hi = np.array([np.nanmean(mae_h_models[i]),np.nanmean(rmse_h_models[i]),np.nanmean(Rsqr_h_models[i]),np.nanmean(Rsqr_adj_h_models[i]),np.nanmean(Rsqr_pred_h_models[i]),np.nanmean(std_h_models[i])]).T
            arr_c_h[p,0:npred] = arr1_hi
            arr_c_h[p,npred:npred+6] = arr2_hi

            if show:
                ax.plot(xall[training_size+1:],Rsqr_h_models[i],'-',color=plt.get_cmap('tab20')(p*2))
                ax.plot(xall[training_size+1:],Rsqr_adj_h_models[i],'--',color=plt.get_cmap('tab20')(p*2))
                ax.plot(xall[training_size+1:],Rsqr_pred_h_models[i],':',color=plt.get_cmap('tab20')(p*2))
                ax.text(2008,0.95-p*0.06,'Rsqr forecast = {:03.2f}'.format(Rsqr_f_models[i]),color=plt.get_cmap('tab20')(p*2))
                ax.plot(xall[training_size+1:],np.ones(len(Rsqr_h_models[i]))*Rsqr_f_models[i],'-',color=plt.get_cmap('tab20')(p*2+1))

                ax_FUD.plot(xall,yall,'o-',color='k')
                ax_FUD.plot(xf_models[i],yf_models[i], 'o:',color=plt.get_cmap('tab20')(p*2))

            p += 1

    arr_c_f = arr_c_f[0:p,:]
    arr_c_h = arr_c_h[0:p,:]

    stats_f_s = pd.DataFrame(arr_c_f,
                             columns=pred_list+['MAE','RMSE','Rsqr','Rsqr_adj','sig_err'])
    stats_h_s = pd.DataFrame(arr_c_h,
                             columns=pred_list+['MAE','RMSE','Rsqr','Rsqr_adj','Rsqr_pred','sig_err'])

    return stats_h_all, stats_f_all, stats_h_s, stats_f_s

###
def remove_collinear_features(df_model, target_var, threshold, target_in_df = False, verbose=False):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold and which have the least correlation with the
        target (dependent) variable. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        df_model: features dataframe
        target_var: name (string) of target variable if target_in_df = True, otherwise the target variable (1-D array)
        threshold: features with correlations greater than this value are removed
        target_in_df: Bool. - Wether the target_var is already in df or needs to be added
        verbose: set to "True" for the log printing

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    if target_in_df:
        tname = target_var
    else:
        df_model['target'] = np.expand_dims(target_var,1)
        tname = 'target'

    # Calculate the correlation matrix
    corr_matrix = df_model.drop(tname, 1).corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    dropped_feature = ""

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                if verbose:
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                col_value_corr = np.abs(df_model[col.values[0]].corr(df_model[tname]))
                row_value_corr = np.abs(df_model[row.values[0]].corr(df_model[tname]))
                if verbose:
                    print("{}: {}".format(col.values[0], np.round(col_value_corr, 3)))
                    print("{}: {}".format(row.values[0], np.round(row_value_corr, 3)))
                if col_value_corr < row_value_corr:
                    drop_cols.append(col.values[0])
                    dropped_feature = "dropped: " + col.values[0]
                else:
                    drop_cols.append(row.values[0])
                    dropped_feature = "dropped: " + row.values[0]
                if verbose:
                    print(dropped_feature)
                    print("-----------------------------------------------------------------------------")

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    df_model = df_model.drop(columns=drops)
    if not target_in_df:
        df_model = df_model.drop(tname, 1)

    if verbose:
        print("dropped columns: ")
        print(list(drops))
        print("-----------------------------------------------------------------------------")
        print("used columns: ")
        print(df_model.columns.tolist())

    return df_model, drops

###
def find_models(combinations,
                df_tr,df_va,df_te,
                clim_valid,clim_test,
                target_tr,target_va,target_te,
                train_predictions_list,valid_predictions_list,test_predictions_list,
                predictors_list,models_list,
                valid_MAE_list,valid_RMSE_list,valid_R2_list,valid_R2adj_list,valid_pval_list,valid_ss_list,
                test_MAE_list,test_RMSE_list,test_R2_list,test_R2adj_list,test_pval_list,test_ss_list,
                p_critical, verbose = False):

    for i in range(len(combinations)):
        x_model = [ df_tr.columns[c] for c in combinations[i]]

        pred_train_select = df_tr[x_model]
        pred_valid_select = df_va[x_model]
        pred_test_select = df_te[x_model]

        mlr_model_train = sm.OLS(target_tr, sm.add_constant(pred_train_select,has_constant='skip'), missing='drop').fit()

        train_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_train_select,has_constant='skip'))
        valid_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_valid_select,has_constant='skip'))
        test_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_test_select,has_constant='skip'))


        if (mlr_model_train.f_pvalue < p_critical) & (np.any(mlr_model_train.pvalues[1:] < p_critical)):

            mask_tr =  (~np.isnan(target_tr)) & (~np.isnan(train_predictions_FUD))
            if np.corrcoef(target_tr[mask_tr],train_predictions_FUD[mask_tr].values)[0,1] > 0:
                if verbose:
                    print(x_model, mlr_model_train.f_pvalue)

                train_predictions_list.append(train_predictions_FUD)
                valid_predictions_list.append(valid_predictions_FUD)
                test_predictions_list.append(test_predictions_FUD)

                predictors_list.append(x_model)
                models_list.append(mlr_model_train)

                valid_MAE_list.append(np.nanmean(np.abs(target_va-valid_predictions_FUD)))
                valid_RMSE_list.append(np.sqrt(np.nanmean((target_va-valid_predictions_FUD)**2.)))
                mlr_model_valid = sm.OLS(target_va, sm.add_constant(valid_predictions_FUD,has_constant='skip'), missing='drop').fit()
                valid_R2_list.append(mlr_model_valid.rsquared)
                # valid_R2adj_list.append( 1 - ((1-mlr_model_valid.rsquared)*((n_va-1)/(n_va-len(x_model))) ))
                valid_R2adj_list.append(mlr_model_valid.rsquared_adj)
                valid_pval_list.append(mlr_model_valid.f_pvalue)
                valid_ss_list.append(1-( (np.nanmean((target_va-valid_predictions_FUD)**2.)) / (np.nanmean((target_va-clim_valid)**2.)) ))

                test_MAE_list.append(np.nanmean(np.abs(target_te-test_predictions_FUD)))
                test_RMSE_list.append(np.sqrt(np.nanmean((target_te-test_predictions_FUD)**2.)))
                mlr_model_test = sm.OLS(target_te, sm.add_constant(test_predictions_FUD,has_constant='skip'), missing='drop').fit()
                test_R2_list.append(mlr_model_test.rsquared)
                # test_R2adj_list.append( 1 - ((1-mlr_model_test.rsquared)*((n_te-1)/(n_te-len(x_model))) ))
                test_R2adj_list.append(mlr_model_test.rsquared_adj)
                test_pval_list.append(mlr_model_test.f_pvalue)
                test_ss_list.append(1-( (np.nanmean((target_te-test_predictions_FUD)**2.)) / (np.nanmean((target_te-clim_test)**2.)) ))



    return train_predictions_list,valid_predictions_list,test_predictions_list,predictors_list,models_list,valid_MAE_list,valid_RMSE_list,valid_R2_list,valid_R2adj_list,valid_pval_list,valid_ss_list,test_MAE_list,test_RMSE_list,test_R2_list,test_R2adj_list,test_pval_list,test_ss_list

###
def eval_accuracy_multiple_models(predictions_in,target_in,cat_in,tercile1,tercile2):

    cat_out = np.zeros(len(predictions_in),dtype='object')*np.nan
    accuracy_out = np.zeros(len(predictions_in))*np.nan

    for m in range(len(predictions_in)):

        cat_tmp = np.zeros((predictions_in[m]).shape)*np.nan

        sum_acc = 0
        for iyr in range(cat_tmp.shape[0]):
            if ~np.isnan(predictions_in[m][iyr]):
                if predictions_in[m][iyr] <= tercile1:
                    cat_tmp[iyr] = -1
                elif predictions_in[m][iyr] > tercile2:
                    cat_tmp[iyr] = 1
                else:
                    cat_tmp[iyr] = 0

                if (cat_tmp[iyr] == cat_in[iyr]):
                    sum_acc += 1

        cat_out[m] = cat_tmp
        accuracy_out[m] = sum_acc/(len(~np.isnan(target_in)))

    return accuracy_out.tolist(), cat_out


###
def find_all_column_combinations(columns,n):
    a_list = np.arange(len(columns))

    all_combinations = []
    for r in range(n,n+1):
        combinations_object = itertools.combinations(a_list, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list

    return all_combinations


###
def make_metric_df(arr_col,predictors_in,MAE_in,RMSE_in,Acc_in,R2_in,R2adj_in,pval_in,predictions_in):
    arr_tmp = np.zeros((len(predictors_in),len(arr_col)),dtype= 'object')*np.nan

    for i in range(len(predictors_in)):
        arr_tmp[i,0] = predictors_in[i][:]
        arr_tmp[i,7] = predictions_in[i]
    arr_tmp[:,1] = MAE_in
    arr_tmp[:,2] = RMSE_in
    arr_tmp[:,3] = Acc_in
    arr_tmp[:,4] = R2_in
    arr_tmp[:,5] = R2adj_in
    arr_tmp[:,6] = pval_in
    df_out = pd.DataFrame(arr_tmp,columns=arr_col)

    return df_out

###
def get_daily_var_from_monthly_cansips_forecasts(sort_type,v,varname,anomaly,region,time,lag=1,date_ref = dt.date(1900,1,1)):

    def sort_by_lead(fdata_select,lead,varname,time,years_cansips):
        """
        # ftype = '5months'
        #    im          lead
        #           0  1  2  3  4
        # [sept=0][ S, O, N, D, J]
        # [Oct =1][ O, N, D, J, F]
        # [Nov =2][ N, D, J, F, M]
        # [Dec =3][ D, J, F, M, A]

        data[im,lead,:,:]

        In entry, 'lead' is specified. There are always 4 months output
        for each lead, but these months change.
        So for,
        lead = 0: S, O, N, D,
        lead = 1: O, N, D, J
        lead = 2: N, D, J, F
        lead = 3: D, J, F, M
        lead = 4: J, F, M, A
        """
        fout = np.zeros((len(time),4))*np.nan
        namelist_out = []
        monthlist = np.array([ 9,10,11,12])
        if lead == 0: namelist = ['Sep.', 'Oct.', 'Nov.', 'Dec.']#; monthlist = [ 9,10,11,12]+lead
        if lead == 1: namelist = ['Oct.', 'Nov.', 'Dec.', 'Jan.']#; monthlist = [10,11,12,13]
        if lead == 2: namelist = ['Nov.', 'Dec.', 'Jan.', 'Feb.']#; monthlist = [11,12,13,14]
        if lead == 3: namelist = ['Dec.', 'Jan.', 'Feb.', 'Mar.']#; monthlist = [12,13,14,15]
        if lead == 4: namelist = ['Jan.', 'Feb.', 'Mar.', 'Apr.']#; monthlist = [13,14,15,16]

        for im in range(4):
            fdata_select_out = np.nanmean(fdata_select[:,im,lead,:,:],axis=(1,2))
            namelist_out += ["{:} month lead ".format(lead)+namelist[im] + varname +  ' forecast']
            if monthlist[im] <= 12:
                tmonth = monthlist[im]
                for it in range(len(time)):
                    date_it = date_ref + dt.timedelta(days=int(time[it]))
                    if date_it.year in years_cansips:
                        if date_it.month == tmonth:
                            fout[it:it+lag,im] = fdata_select_out[np.where(years_cansips == date_it.year)]
            else:
                tmonth = monthlist[im]-12

                for it in range(len(time)):
                    date_it = date_ref + dt.timedelta(days=int(time[it]))
                    if (date_it.year-1) in years_cansips:
                        if date_it.month == tmonth:
                            fout[it:it+lag,im] = fdata_select_out[np.where(years_cansips == (date_it.year-1))]

        return fout, namelist_out

    def sort_by_startmonth(fdata_select,sm,varname,time,years_cansips):
        lead = 0
        data_monthly_lead0, name_monthly_lead0 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        lead = 1
        data_monthly_lead1, name_monthly_lead1 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        lead = 2
        data_monthly_lead2, name_monthly_lead2 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        lead = 3
        data_monthly_lead3, name_monthly_lead3 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        lead = 4
        data_monthly_lead4, name_monthly_lead4 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        fvars_sep = np.zeros((len(time),5))*np.nan
        fvars_oct = np.zeros((len(time),9))*np.nan
        fvars_nov = np.zeros((len(time),12))*np.nan
        fvars_dec = np.zeros((len(time),14))*np.nan

        varnames_sep = []
        varnames_oct = []
        varnames_nov = []
        varnames_dec = []

        fvars_sep[:,0] = data_monthly_lead0[:,0]
        varnames_sep += [name_monthly_lead0[0]]
        fvars_sep[:,1] = data_monthly_lead1[:,0]
        varnames_sep += [name_monthly_lead1[0]]
        fvars_sep[:,2] = data_monthly_lead2[:,0]
        varnames_sep += [name_monthly_lead2[0]]
        fvars_sep[:,3] = data_monthly_lead3[:,0]
        varnames_sep += [name_monthly_lead3[0]]
        fvars_sep[:,4] = data_monthly_lead4[:,0]
        varnames_sep += [name_monthly_lead4[0]]

        fvars_oct[:,0:len(varnames_sep)] = fvars_sep[:,0:len(varnames_sep)]
        varnames_oct += [varnames_sep[i] for i in range(len(varnames_sep))]
        fvars_oct[:,len(varnames_sep)] = data_monthly_lead0[:,1]
        varnames_oct += [name_monthly_lead0[1]]
        fvars_oct[:,len(varnames_sep)+1] = data_monthly_lead1[:,1]
        varnames_oct += [name_monthly_lead1[1]]
        fvars_oct[:,len(varnames_sep)+2] = data_monthly_lead2[:,1]
        varnames_oct += [name_monthly_lead2[1]]
        fvars_oct[:,len(varnames_sep)+3] = data_monthly_lead3[:,1]
        varnames_oct += [name_monthly_lead3[1]]

        fvars_nov[:,0:len(varnames_oct)] = fvars_oct[:,0:len(varnames_oct)]
        varnames_nov += [varnames_oct[i] for i in range(len(varnames_oct))]
        fvars_nov[:,len(varnames_oct)] = data_monthly_lead0[:,2]
        varnames_nov += [name_monthly_lead0[2]]
        fvars_nov[:,len(varnames_oct)+1] = data_monthly_lead1[:,2]
        varnames_nov += [name_monthly_lead1[2]]
        fvars_nov[:,len(varnames_oct)+2] = data_monthly_lead2[:,2]
        varnames_nov += [name_monthly_lead2[2]]

        fvars_dec[:,0:len(varnames_nov)] = fvars_nov[:,0:len(varnames_nov)]
        varnames_dec += [varnames_nov[i] for i in range(len(varnames_nov))]
        fvars_dec[:,len(varnames_nov)] = data_monthly_lead0[:,3]
        varnames_dec += [name_monthly_lead0[3]]
        fvars_dec[:,len(varnames_nov)+1] = data_monthly_lead1[:,3]
        varnames_dec += [name_monthly_lead1[3]]

        if sm == 0:
            return fvars_sep, varnames_sep
        if sm == 1:
            return fvars_oct, varnames_oct
        if sm == 2:
            return fvars_nov, varnames_nov
        if sm == 3:
            return fvars_dec, varnames_dec

    base = "cansips_hindcast_raw_"
    res = "latlon1.0x1.0_"
    p_dir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CanSIPS/hindcast/'
    ftype = '5months'

    data = np.load(p_dir+base+res+'ensemble_vars_sep_dec_f'+ftype+'.npz')
    feature_list = data['feature_list'][:]
    if anomaly:
        fdata = data['anomaly'][:]
    else:
        fdata = data['ensm'][:]
    years_cansips = np.arange(1979,2022)
    lat_cansips = data['lat'][:]
    lon_cansips = data['lon'][:]

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

    f = np.where(feature_list == varname)[0][0]
    fdata_select = fdata[f,:,:,:,ilat1:ilat2,ilon1:ilon2].copy()
    lat_select = lat[ilat1:ilat2+1]
    lon_select = lon[ilon1:ilon2+1]

    if sort_type == 'lead':
        lead = v
        dout,nameout = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

    if sort_type == 'startmonth':
        sm = v
        dout,nameout = sort_by_startmonth(fdata_select,sm,varname,time,years_cansips)

    return dout, nameout


###
def get_daily_var_from_seasonal_cansips_forecasts(sort_type,v,varname,anomaly,region,time,lag=1,date_ref = dt.date(1900,1,1)):

    def sort_by_lead(fdata_select,lead,varname,time,years_cansips):
        """
        ftype = 'seasonal'

            im          iseason
                    0    1    2    3
          [sept=0][SON, OND, NDJ, DJF]
          [Oct =1][nan, OND, NDJ, DJF]
          [Nov =2][nan, nan, NDJ, DJF]
          [Dec =3][nan, nan, nan, DJF]

        iseason = 0 SON --> lead = 0
        iseason = 1 OND --> lead = 0, 1
        iseason = 2 NDJ --> lead = 0, 1, 2
        iseason = 3 DJF --> lead = 0, 1, 2, 3

        data[im, iseason,:,:]

        when lead = 0: im = iseason
        when lead = 1: im = iseason-1
        when lead = 2: im = iseason-2
        when lead = 3: im = iseason-3

        In entry, 'lead' is specified.
        So for,
        lead = 0: there are 4 seasons output, SON, OND, NDJ, DJF  #; monthlist = [ 9,10,11,12]
        lead = 1: there are 3 seasons output, OND, NDJ, DJF,      #; monthlist = [   10,11,12]
        lead = 2: there are 2 seasons output, NDJ, DJF            #; monthlist = [      11,12]
        lead = 3: there are 1 seasons output, DJF                 #; monthlist = [         12]
        """
        fout = np.zeros((len(time),4))*np.nan
        season_list = ['SON ', 'OND ', 'NDJ ', 'DJF ']
        namelist_out = []
        for iseason in range(lead,4):
            if lead == 0: im = iseason
            if lead == 1: im = iseason-1
            if lead == 2: im = iseason-2
            if lead == 3: im = iseason-3

            fdata_select_out = np.nanmean(fdata_select[:,im,iseason,:,:],axis=(1,2))
            namelist_out += ["{:} month lead ".format(lead)+season_list[iseason] + varname +  ' forecast']

            # if season_list[iseason][0] == 'S': tmonth = 9
            # if season_list[iseason][0] == 'O': tmonth = 10
            # if season_list[iseason][0] == 'N': tmonth = 11
            # if season_list[iseason][0] == 'D': tmonth = 12
            if im == 0: tmonth = 9
            if im == 1: tmonth = 10
            if im == 2: tmonth = 11
            if im == 3: tmonth = 12

            for it in range(len(time)):
                date_it = date_ref + dt.timedelta(days=int(time[it]))
                if (date_it.year) in years_cansips:
                    if date_it.month == tmonth:
                        fout[it:it+lag,im] = fdata_select_out[np.where(years_cansips == (date_it.year))]

        return fout, namelist_out



    def sort_by_startmonth(fdata_select,sm,varname,time,years_cansips):
        lead = 0
        data_monthly_lead0, name_monthly_lead0 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        lead = 1
        data_monthly_lead1, name_monthly_lead1 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        lead = 2
        data_monthly_lead2, name_monthly_lead2 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        lead = 3
        data_monthly_lead3, name_monthly_lead3 = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

        fvars_sep = np.zeros((len(time),4))*np.nan
        fvars_oct = np.zeros((len(time),7))*np.nan
        fvars_nov = np.zeros((len(time),9))*np.nan
        fvars_dec = np.zeros((len(time),10))*np.nan

        varnames_sep = []
        varnames_oct = []
        varnames_nov = []
        varnames_dec = []

        fvars_sep[:,0] = data_monthly_lead0[:,0]
        varnames_sep += [name_monthly_lead0[0]]
        fvars_sep[:,1] = data_monthly_lead1[:,0]
        varnames_sep += [name_monthly_lead1[0]]
        fvars_sep[:,2] = data_monthly_lead2[:,0]
        varnames_sep += [name_monthly_lead2[0]]
        fvars_sep[:,3] = data_monthly_lead3[:,0]
        varnames_sep += [name_monthly_lead3[0]]

        fvars_oct[:,0:len(varnames_sep)] = fvars_sep[:,0:len(varnames_sep)]
        varnames_oct += [varnames_sep[i] for i in range(len(varnames_sep))]
        fvars_oct[:,len(varnames_sep)] = data_monthly_lead0[:,1]
        varnames_oct += [name_monthly_lead0[1]]
        fvars_oct[:,len(varnames_sep)+1] = data_monthly_lead1[:,1]
        varnames_oct += [name_monthly_lead1[1]]
        fvars_oct[:,len(varnames_sep)+2] = data_monthly_lead2[:,1]
        varnames_oct += [name_monthly_lead2[1]]

        fvars_nov[:,0:len(varnames_oct)] = fvars_oct[:,0:len(varnames_oct)]
        varnames_nov += [varnames_oct[i] for i in range(len(varnames_oct))]
        fvars_nov[:,len(varnames_oct)] = data_monthly_lead0[:,2]
        varnames_nov += [name_monthly_lead0[2]]
        fvars_nov[:,len(varnames_oct)+1] = data_monthly_lead1[:,2]
        varnames_nov += [name_monthly_lead1[2]]

        fvars_dec[:,0:len(varnames_nov)] = fvars_nov[:,0:len(varnames_nov)]
        varnames_dec += [varnames_nov[i] for i in range(len(varnames_nov))]
        fvars_dec[:,len(varnames_nov)] = data_monthly_lead0[:,3]
        varnames_dec += [name_monthly_lead0[3]]

        if sm == 0:
            return fvars_sep, varnames_sep
        if sm == 1:
            return fvars_oct, varnames_oct
        if sm == 2:
            return fvars_nov, varnames_nov
        if sm == 3:
            return fvars_dec, varnames_dec

    base = "cansips_hindcast_raw_"
    res = "latlon1.0x1.0_"
    p_dir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CanSIPS/hindcast/'
    ftype = '5months'

    data = np.load(p_dir+base+res+'ensemble_vars_sep_dec_f'+ftype+'.npz')
    feature_list = data['feature_list'][:]
    if anomaly:
        fdata = data['anomaly'][:]
    else:
        fdata = data['ensm'][:]
    years_cansips = np.arange(1979,2022)
    lat_cansips = data['lat'][:]
    lon_cansips = data['lon'][:]

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

    f = np.where(feature_list == varname)[0][0]
    fdata_select = fdata[f,:,:,:,ilat1:ilat2,ilon1:ilon2].copy()
    lat_select = lat[ilat1:ilat2+1]
    lon_select = lon[ilon1:ilon2+1]

    if sort_type == 'lead':
        lead = v
        dout,nameout = sort_by_lead(fdata_select,lead,varname,time,years_cansips)

    if sort_type == 'startmonth':
        sm = v
        dout,nameout = sort_by_startmonth(fdata_select,sm,varname,time,years_cansips)

    return dout, nameout


