#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:10:50 2022

@author: Amelie
"""
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

#%%
def get_daily_var_from_seasonal_cansips_forecasts(sort_type,v,varname,anomaly,region,time,date_ref = dt.date(1900,1,1)):

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

            if season_list[iseason][0] == 'S': tmonth = 9
            if season_list[iseason][0] == 'O': tmonth = 10
            if season_list[iseason][0] == 'N': tmonth = 11
            if season_list[iseason][0] == 'D': tmonth = 12

            for it in range(len(time)):
                date_it = date_ref + dt.timedelta(days=int(time[it]))
                if (date_it.year) in years_cansips:
                    if date_it.month == tmonth:
                        fout[it,im] = fdata_select_out[np.where(years_cansips == (date_it.year))]

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





#%%

feature_list = ['WTMP_SFC_0',
                'PRATE_SFC_0',
                'TMP_TGL_2m',
                'PRMSL_MSL_0']
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
region = 'D'
anomaly = True
varname = 'TMP_TGL_2m'
#%%
sort_type = 'lead'
lead = 0
data_daily_lead0, varnames_lead0 = get_daily_var_from_seasonal_cansips_forecasts(sort_type,lead,varname,anomaly,region,time)
lead = 1
data_daily_lead1, varnames_lead1 = get_daily_var_from_seasonal_cansips_forecasts(sort_type,lead,varname,anomaly,region,time)
lead = 2
data_daily_lead2, varnames_lead2 = get_daily_var_from_seasonal_cansips_forecasts(sort_type,lead,varname,anomaly,region,time)
lead = 3
data_daily_lead3, varnames_lead3 = get_daily_var_from_seasonal_cansips_forecasts(sort_type,lead,varname,anomaly,region,time)
#%%
sort_type = 'startmonth'
sm = 0 # 0:Sept.  1:Oct.  2:Nov.  3:Dec
data_daily_startS, varnames_startS = get_daily_var_from_seasonal_cansips_forecasts(sort_type,sm,varname,anomaly,region,time)
sm = 1 # 0:Sept.  1:Oct.  2:Nov.  3:Dec
data_daily_startO, varnames_startO = get_daily_var_from_seasonal_cansips_forecasts(sort_type,sm,varname,anomaly,region,time)
sm = 2 # 0:Sept.  1:Oct.  2:Nov.  3:Dec
data_daily_startN, varnames_startN = get_daily_var_from_seasonal_cansips_forecasts(sort_type,sm,varname,anomaly,region,time)
sm = 3 # 0:Sept.  1:Oct.  2:Nov.  3:Dec
data_daily_startD, varnames_startD = get_daily_var_from_seasonal_cansips_forecasts(sort_type,sm,varname,anomaly,region,time)

#%%
# TEST TO SEE IF I RECOVER THE SME MONTHLY MEAN WITH THE METHOD USED IN THE MLR ROUTINE:
from functions_MLR import get_monthly_vars_from_daily

years = np.array([1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021])

d = data_daily_startO
vn = ['Avg. ' + varnames_startO[j] for j in range(d.shape[1])]
monthly_test = get_monthly_vars_from_daily(d,vn,years,time,replace_with_nan=False)

