#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:51:19 2020

@author: Amelie
"""

import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import calendar
import scipy as sp
import cdsapi
import os
import scipy
from scipy import ndimage


#%%%==============================================================
# Units

###
def K_to_C(var_in):
    # Convert Kelvins to Celsius
    return var_in-273.15

def C_to_K(var_in):
    # Convert Celsius to Kelvins
    return var_in+273.15

#%%%==============================================================
# ERA5 Data

###
def download_era5(features, region, output_dir = os.getcwd(), start_year=1991, end_year=2021, start_month=1, end_month=13):
    """
    Download ERA5 files

    :param path: String. Full directory to download files to
    :param start_year: Integer. Start year in YYYY.
    :param end_year: Integer. End year in YYYY.
    :return: None
    """

    base = "ERA5_"
    url = "https://cds.climate.copernicus.eu/api/v2"
    key = "68986:cd4929b1-ca5d-4f2b-b884-4d89b243703c"

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

    for year in range(start_year, end_year):
        print(year)
        os.chdir(output_dir)

        for month in range(start_month, end_month):
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
            extension = "_{}{}.nc".format(year, str(month).rjust(2, '0'))

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
                                        '00:00', '06:00',
                                        '12:00', '18:00'
                                    ],
                                    'day': [
                                        '01', '02', '03',
                                        '04', '05', '06',
                                        '07', '08', '09',
                                        '10', '11', '12',
                                        '13', '14', '15',
                                        '16', '17', '18',
                                        '19', '20', '21',
                                        '22', '23', '24',
                                        '25', '26', '27',
                                        '28', '29', '30', '31'
                                    ],
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



#%%==============================================================
# Interaction with CSV files
###
def read_csv(file_path,skip=0,delimiter=','):

    f = open(file_path, 'r')
    data = np.genfromtxt(f,skip_header=skip,delimiter=',')
    f.close()

    return data

###
def clean_csv(arr,columns,nan_id=None,return_type=float):

    arr = arr[:,columns]
    if nan_id is not None:
        arr[np.isin(arr,nan_id)] = np.nan

    clean_arr = arr

    return clean_arr.astype(return_type)

#%%==============================================================
# Interaction with NetCDF files
###
def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print( "\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim)
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print( "\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


#%%==============================================================
# (Least-square) Fitting
###
def linear_fit(x_in,y_in):
    mask_x = ~np.isnan(x_in)
    mask_y = ~np.isnan(y_in)
    mask = mask_x & mask_y

    x_in = x_in[mask]
    y_in = y_in[mask]

    A = np.vstack([x_in, np.ones(len(x_in))]).T
    b = y_in
    lstsqr_fit = np.linalg.lstsq(A, b)
    coeff = lstsqr_fit[0]
    slope_fit = np.dot(A,coeff)

    SS_res = np.sum((slope_fit-y_in)**2.)
    SS_tot = np.sum((y_in - y_in.mean())**2.)
    R_sqr = 1-(SS_res/SS_tot)

    return coeff, R_sqr

###
def linear_fit_no_intercept(x_in,y_in):
    mask_x = ~np.isnan(x_in)
    mask_y = ~np.isnan(y_in)
    mask = mask_x & mask_y

    x_in = x_in[mask]
    y_in = y_in[mask]

    A = A = np.vstack([x_in]).T
    b = y_in
    lstsqr_fit = np.linalg.lstsq(A, b)
    linfit = lstsqr_fit[0]
    slope_fit = np.dot(A,linfit)

    SS_res = np.sum((slope_fit-y_in)**2.)
    SS_tot = np.sum((y_in - y_in.mean())**2.)
    rsqr = 1-(SS_res/SS_tot)

    return linfit, rsqr

###
def press_statistic(y_true, y_pred, xs):
    """
    Calculation of the `Press Statistics <https://www.otexts.org/1580>`_
    """
    res = y_pred - y_true
    hat = xs.dot(np.linalg.pinv(xs))
    den = (1 - np.diagonal(hat))
    sqr = np.square(res/den)
    return sqr.sum()

###
def predicted_r2(y_true, y_pred, xs):
    """
    Calculation of the `Predicted R-squared <https://rpubs.com/RatherBit/102428>`_
    """
    press = press_statistic(y_true=y_true,
                            y_pred=y_pred,
                            xs=xs
    )

    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - press / sst

###
def bootstrap(xvar_in, yvar_in, nboot=1000):
    nyears = len(xvar_in)
    r_out = np.zeros((nboot))*np.nan

    for n in range(nboot):
        if nboot >1:
            boot_indx = np.random.choice(nyears,size=nyears,replace=True)
        else:
            boot_indx = np.random.choice(nyears,size=nyears,replace=False)


        xvar_boot = xvar_in[boot_indx].copy()
        yvar_boot = yvar_in[boot_indx].copy()

        lincoeff, Rsqr = linear_fit(xvar_boot,yvar_boot)

        r_out[n] = np.sqrt(Rsqr)
        if (lincoeff[0]< 0):
            r_out[n] *= -1

    return r_out

#%%==============================================================
# Stats

##
def r_to_z(r):
    # Fisher z-transformation: https://en.wikipedia.org/wiki/Fisher_transformation
    # Transforms Pearson's correlation coefficient (r) to an approximately normal statistics
    return np.log((1 + r) / (1 - r)) / 2.0

###
def z_to_r(z):
    # Inverse Fisher z-transformation: https://en.wikipedia.org/wiki/Fisher_transformation
    e = np.exp(2 * z)
    return((e - 1) / (e + 1))

###
def r_confidence_interval(r, alpha, n, tailed = 'two'):
    z = r_to_z(r)
    se = 1.0 / np.sqrt(n - 3)
    if tailed == 'two': z_crit = sp.stats.norm.ppf(1 - alpha/2)  # 2-tailed z critical value
    if tailed == 'one': z_crit = sp.stats.norm.ppf(1 - alpha)  # 1-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    # Return a sequence
    return (z_to_r(lo), z_to_r(hi))



#%%==============================================================
# Plotting
###
def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

#%%==============================================================
# Mapping/Coordinates
###
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (input lat/lon specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


#%%==============================================================
# Time series operations
###
def running_mean(x, N, mean_type='centered'):
    cumsum = np.cumsum(np.insert(x, 0, 0))

    xmean_tmp = (cumsum[N:] - cumsum[:-N]) / float(N)
    if mean_type == 'centered':
        xmean = np.insert(xmean_tmp,0,np.zeros(int((N-1)/2.))*np.nan)
        xmean = np.insert(xmean,xmean.size, np.zeros(int((N-1)/2.))*np.nan)
    if mean_type == 'before':
        xmean = np.insert(xmean_tmp,0,np.zeros(int((N-1)))*np.nan)

    return xmean


###
def running_nanmean(x, N=3, mean_type='centered'):
    xmean = np.ones(x.shape[0])*np.nan
    temp = np.vstack([x[i:-(N-i)] for i in range(N)]) # stacks vertically the strided arrays
    temp = np.nanmean(temp, axis=0)

    if mean_type == 'before':
        xmean[N-1:N-1+temp.shape[0]] = temp

    if mean_type == 'centered':
        xmean[int((N-1)/2):int((N-1)/2)+temp.shape[0]] = temp

    return xmean


###
def detrend_ts(var_in,years,anomaly_type):

    if anomaly_type == 'linear':
        [m,b],_ = linear_fit(years, var_in)
        trend = m*years + b

        var_out = var_in-trend

        return var_out, [m,b]

    if anomaly_type == 'mean':
        mean = np.nanmean(var_in)

        var_out = var_in-mean

        return var_out, mean

###
def rolling_climo(Nwindow,ts_in,output_type,time_in,years,time_other=None,date_ref = dt.date(1900,1,1)):

    import calendar
    ts_daily = np.zeros((Nwindow,366,len(years)))*np.nan

    for it in range(ts_in.shape[0]):

        iw0 = np.max([0,it-int((Nwindow-1)/2)])
        iw1 = np.min([it+int((Nwindow-1)/2)+1,len(time_in)-1])

        ts_window = ts_in[iw0:iw1]
        # print(it==iw0,it,len(time_in),iw1,iw1-1,time_in[iw0],time_in[iw1],time_in[iw1-1])

        if ((time_in[iw1-1]-time_in[iw0]+1) <= Nwindow):
            # Only keep windows that cover continuous data, i.e.
            # exclude windows in which time jumps from one year to the
            # next due to non-sequential training intervals (in the
            # case of k-fold cross-validation, for example)

            date_mid = date_ref+dt.timedelta(days=int(time_in[it]))
            year_mid = date_mid.year
            month_mid = date_mid.month
            day_mid = date_mid.day

            if len(np.where(years == year_mid)[0]) > 0:
                iyear = np.where(years == year_mid)[0][0]
                doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days+1

                ts_daily[0:len(ts_window),doy-1,iyear] = ts_window

                if not calendar.isleap(year_mid) and (doy == 365) and (year_mid != years[-1]):
                    imid = int((Nwindow-1)/2)
                    ts_window_366 = np.zeros((Nwindow))*np.nan
                    ts_window_366[imid] = np.array(np.nanmean([ts_in[it],ts_in[np.nanmin([len(ts_in)-1,it+1])]]))
                    ts_window_366[0:imid] = ts_in[int(it+1-((Nwindow-1)/2)):it+1]
                    ts_window_366[imid+1:Nwindow] = ts_in[it+1:int(it+1+((Nwindow-1)/2))]
                    ts_daily[:,365,iyear] = ts_window_366


    # Then, find the climatological mean and std for each window/date
    if output_type == 'year':
        mean_clim = np.zeros((366))*np.nan
        std_clim = np.zeros((366))*np.nan
        mean_clim[:] = np.nanmean(ts_daily,axis=(0,2))
        std_clim[:] = np.nanstd(ts_daily,axis=(0,2))

    if output_type == 'all_time':
        mean_clim = np.zeros(len(time_in))*np.nan
        std_clim = np.zeros(len(time_in))*np.nan

        yr_st = (date_ref+dt.timedelta(days=int(time_in[0]))).year
        yr_end = (date_ref+dt.timedelta(days=int(time_in[-1]))).year
        all_years = np.arange(yr_st,yr_end+1)
        for iyr,year in enumerate(all_years):
            istart = np.where(time_in == (dt.date(int(year),1,1)-date_ref).days)[0][0]
            iend = np.where(time_in == (dt.date(int(year),12,31)-date_ref).days)[0][0]+1
            if not calendar.isleap(year):
                mean_clim[istart:iend] = np.nanmean(ts_daily,axis=(0,2))[:-1]
                std_clim[istart:iend] = np.nanstd(ts_daily,axis=(0,2))[:-1]
            else:
                mean_clim[istart:iend] = np.nanmean(ts_daily,axis=(0,2))
                std_clim[istart:iend] = np.nanstd(ts_daily,axis=(0,2))

    if output_type == 'time_in':
        mean_clim = np.zeros(len(time_in))*np.nan
        std_clim = np.zeros(len(time_in))*np.nan

        for it in range(len(time_in)):
            day_it = int((date_ref+dt.timedelta(days=int(time_in[it]))).day)
            month_it = int((date_ref+dt.timedelta(days=int(time_in[it]))).month)
            year_it = int((date_ref+dt.timedelta(days=int(time_in[it]))).year)
            doy_it = (dt.date(year_it,month_it,day_it) - dt.date(year_it,1,1)).days + 1

            mean_clim[it] = np.nanmean(ts_daily,axis=(0,2))[doy_it-1]
            std_clim[it] = np.nanstd(ts_daily,axis=(0,2))[doy_it-1]


    if output_type == 'other':
        if time_other is not None:
            mean_clim = np.zeros(len(time_other))*np.nan
            std_clim = np.zeros(len(time_other))*np.nan

            for it in range(len(time_other)):
                day_it = int((date_ref+dt.timedelta(days=int(time_other[it]))).day)
                month_it = int((date_ref+dt.timedelta(days=int(time_other[it]))).month)
                year_it = int((date_ref+dt.timedelta(days=int(time_other[it]))).year)
                doy_it = (dt.date(year_it,month_it,day_it) - dt.date(year_it,1,1)).days + 1

                mean_clim[it] = np.nanmean(ts_daily,axis=(0,2))[doy_it-1]
                std_clim[it] = np.nanstd(ts_daily,axis=(0,2))[doy_it-1]

        else:
            raise Exception('NO TIME ARRAY: The "other" output option requires a target time array.')


    return mean_clim, std_clim, ts_daily

###
def season_mask(time,season,msd = 1,date_ref = dt.date(1900,1,1)):

    mask = np.zeros(time.shape).astype(bool)

    for it in range(mask.shape[0]):
        date_it = date_ref+dt.timedelta(days=int(time[it]))

        # Spring: March 1st to June 1st
        if season == 'spring':
            if (((date_it - dt.date(int(date_it.year),3,msd)).days > 0) &
               ((date_it - dt.date(int(date_it.year),6,msd)).days <= 0) ):
                   mask[it] = True

        # Summer: June 1st to September 1st
        if season == 'summer':
            if (((date_it - dt.date(int(date_it.year),6,msd)).days > 0) &
               ((date_it - dt.date(int(date_it.year),9,msd)).days <= 0) ):
                   mask[it] = True

       # Fall: September 1st to December 1st
        if season == 'fall':
            if (((date_it - dt.date(int(date_it.year),9,msd)).days > 0) &
               ((date_it - dt.date(int(date_it.year),12,msd)).days <= 0) ):
                   mask[it] = True

       # Winter: December 1st to March 1st
        if season == 'winter':
            if (((date_it - dt.date(int(date_it.year),12,msd)).days > 0)):
                 mask[it] = True #December 1st to December 31st

            if (((date_it - dt.date(int(date_it.year),3,msd)).days <= 0)):
                 mask[it] = True # January 1st to March 1st

    return mask


###
# OLD VERSION!!!!!! DO NOT USE - JUST KEEP FOR ARCHIVE CHECKS
# def find_freezeup_Tw(def_opt,Twater_in,dTdt_in,d2Tdt2_in,time,years,thresh_T = 2.0,thresh_dTdt = 0.2,thresh_d2Tdt2 = 0.2,ndays = 7,date_ref = dt.date(1900,1,1)):

#     if def_opt == 1:
#         # T is below thresh_T:
#         mask_tmp = Twater_in <= thresh_T

#     if (def_opt == 2):
#         # T is below thresh_T and dTdt is below thresh_dTdt and d2Tdt2 is below thresh_d2Tdt2:
#         zero_dTdt_mask = np.abs(dTdt_in) <= thresh_dTdt
#         # zero_d2Tdt2_mask = np.abs(d2Tdt2_in) <= thresh_d2Tdt2
#         zero_T_mask = Twater_in <= thresh_T
#         # mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
#         mask_tmp = zero_T_mask & zero_dTdt_mask

#     if (def_opt == 3):
#         # T is below thresh_T and dTdt is below thresh_dTdt and d2Tdt2 is below thresh_d2Tdt2:
#         zero_dTdt_mask = np.abs(dTdt_in) <= thresh_dTdt
#         # zero_d2Tdt2_mask = np.abs(d2Tdt2_in) <= thresh_d2Tdt2
#         zero_T_mask = Twater_in <= thresh_T
#         # mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
#         mask_tmp = zero_T_mask & zero_dTdt_mask
#         # mask_tmp = zero_T_mask & zero_d2Tdt2_mask
#         # mask_tmp = zero_dTdt_mask

#     mask_freezeup = mask_tmp.copy()
#     mask_freezeup[:] = False

#     freezeup_Tw=np.zeros((len(years)))*np.nan
#     freezeup_date=np.zeros((len(years),3))*np.nan
#     iyr = 0
#     flag = 0 # This flag will become 1 if the data time series starts during the freezing season, to indicate that we cannot use the first date of temp. below freezing point as the freezeup date.

#     for im in range(1,mask_freezeup.size):

#         if (im == 1) | (~mask_tmp[im-1]):

#             sum_m = 0
#             if ~mask_tmp[im]:
#                 sum_m = 0
#             else:
#                 # start new group
#                 sum_m +=1
#                 istart = im

#                 # Below will only occur if ndays is set to 1, e.g. first day of freezing temp.
#                 if (sum_m >= ndays):
#                     # Temperature has been lower than thresh_T
#                     # for more than (or equal to) ndays.
#                     # Define freezeup date as first date of group

#                     date_start = date_ref+dt.timedelta(days=int(time[istart]))
#                     doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

#                     if doy_start > 300:
#                         if (np.where(np.array(years) == date_start.year)[0].size > 0):
#                             iyr = np.where(np.array(years) == date_start.year)[0][0]
#                     elif doy_start < 60:
#                         if (np.where(np.array(years) == date_start.year-1)[0].size > 0):
#                             iyr = np.where(np.array(years) == date_start.year-1)[0][0]
#                     else:
#                         continue

#                     if np.isnan(freezeup_date[iyr,0]):

#                         if iyr == 0:
#                             if (np.sum(np.isnan(Twater_in[istart-ndays-1-30:istart-ndays-1])) < 7) & (flag == 0):
#                                 freezeup_date[iyr,0] = date_start.year
#                                 freezeup_date[iyr,1] = date_start.month
#                                 freezeup_date[iyr,2] = date_start.day
#                                 freezeup_Tw[iyr] = Twater_in[istart]
#                                 mask_freezeup[istart] = True
#                             else:
#                                 flag = 1 # This flag indicates that the data time series has started during the freezing season already, so that we cannot use the first date of temp. below freezing point as the freezeup date.
#                                 continue
#                         else:
#                             if np.isnan(freezeup_date[iyr-1,0]):
#                                 freezeup_date[iyr,0] = date_start.year
#                                 freezeup_date[iyr,1] = date_start.month
#                                 freezeup_date[iyr,2] = date_start.day
#                                 freezeup_Tw[iyr] = Twater_in[istart]
#                                 mask_freezeup[istart] = True
#                             else:
#                                 if freezeup_date[iyr-1,0] == date_start.year:
#                                     # 2012 01 05
#                                     # 2012 01 07 NO
#                                     # 2012 12 22 YES
#                                     if (freezeup_date[iyr-1,1] < 5) & (date_start.month > 10):
#                                         freezeup_date[iyr,0] = date_start.year
#                                         freezeup_date[iyr,1] = date_start.month
#                                         freezeup_date[iyr,2] = date_start.day
#                                         freezeup_Tw[iyr] = Twater_in[istart]
#                                         mask_freezeup[istart] = True

#                                 elif date_start.year == freezeup_date[iyr-1,0]+1:
#                                     # 2012 12 22
#                                     # 2013 01 14 NO
#                                     # 2013 12 24 YES

#                                     #2014 01 03 (2013 season)
#                                     #2015 01 13 (2014 season)
#                                     if (date_start.month > 10):
#                                         freezeup_date[iyr,0] = date_start.year
#                                         freezeup_date[iyr,1] = date_start.month
#                                         freezeup_date[iyr,2] = date_start.day
#                                         freezeup_Tw[iyr] = Twater_in[istart]
#                                         mask_freezeup[istart] = True
#                                     elif (date_start.month < 5) & (freezeup_date[iyr-1,1] < 5) :
#                                         freezeup_date[iyr,0] = date_start.year
#                                         freezeup_date[iyr,1] = date_start.month
#                                         freezeup_date[iyr,2] = date_start.day
#                                         freezeup_Tw[iyr] = Twater_in[istart]
#                                         mask_freezeup[istart] = True

#                                 elif date_start.year == freezeup_date[iyr-1,0]+2:
#                                     if (date_start.month < 5):
#                                         freezeup_date[iyr,0] = date_start.year
#                                         freezeup_date[iyr,1] = date_start.month
#                                         freezeup_date[iyr,2] = date_start.day
#                                         freezeup_Tw[iyr] = Twater_in[istart]
#                                         mask_freezeup[istart] = True

#                                 else:
#                                     print(iyr)
#                                     print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)
#                                     # if (date_start.year > freezeup_date[iyr-1,0]+2):
#                                     #     freezeup_date[iyr,0] = date_start.year
#                                     #     freezeup_date[iyr,1] = date_start.month
#                                     #     freezeup_date[iyr,2] = date_start.day
#                                     #     mask_freezeup[istart] = True
#                                     # else:
#                                     #     print(iyr)
#                                     #     print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)


#         else:
#             if mask_tmp[im]:
#                 sum_m += 1

#                 if (sum_m >= ndays):
#                     # Temperature has been lower than thresh_T
#                     # for more than (or equal to) ndays.
#                     # Define freezeup date as first date of group

#                     date_start = date_ref+dt.timedelta(days=int(time[istart]))
#                     doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

#                     if doy_start > 300:
#                         if (np.where(np.array(years) == date_start.year)[0].size > 0):
#                             iyr = np.where(np.array(years) == date_start.year)[0][0]
#                     elif doy_start < 60:
#                         if (np.where(np.array(years) == date_start.year-1)[0].size > 0):
#                             iyr = np.where(np.array(years) == date_start.year-1)[0][0]
#                     else:
#                         continue

#                     if np.isnan(freezeup_date[iyr,0]):

#                         if iyr == 0:
#                             if (np.sum(np.isnan(Twater_in[istart-ndays-1-30:istart-ndays-1])) < 7) & (flag == 0):
#                                 freezeup_date[iyr,0] = date_start.year
#                                 freezeup_date[iyr,1] = date_start.month
#                                 freezeup_date[iyr,2] = date_start.day
#                                 freezeup_Tw[iyr] = Twater_in[istart]
#                                 mask_freezeup[istart] = True
#                             else:
#                                 flag = 1 # This flag indicates that the data time series has started during the freezing season already, so that we cannot use the first date of temp. below freezing point as the freezeup date.
#                                 continue
#                         else:
#                             if np.isnan(freezeup_date[iyr-1,0]):
#                                 freezeup_date[iyr,0] = date_start.year
#                                 freezeup_date[iyr,1] = date_start.month
#                                 freezeup_date[iyr,2] = date_start.day
#                                 freezeup_Tw[iyr] = Twater_in[istart]
#                                 mask_freezeup[istart] = True
#                             else:
#                                 if freezeup_date[iyr-1,0] == date_start.year:
#                                     # 2012 01 05
#                                     # 2012 01 07 NO
#                                     # 2012 12 22 YES
#                                     if (freezeup_date[iyr-1,1] < 5) & (date_start.month > 10):
#                                         freezeup_date[iyr,0] = date_start.year
#                                         freezeup_date[iyr,1] = date_start.month
#                                         freezeup_date[iyr,2] = date_start.day
#                                         freezeup_Tw[iyr] = Twater_in[istart]
#                                         mask_freezeup[istart] = True

#                                 elif date_start.year == freezeup_date[iyr-1,0]+1:
#                                     # 2012 12 22
#                                     # 2013 01 14 NO
#                                     # 2013 12 24 YES

#                                     #2014 01 03 (2013 season)
#                                     #2015 01 13 (2014 season)
#                                     if (date_start.month > 10):
#                                         freezeup_date[iyr,0] = date_start.year
#                                         freezeup_date[iyr,1] = date_start.month
#                                         freezeup_date[iyr,2] = date_start.day
#                                         freezeup_Tw[iyr] = Twater_in[istart]
#                                         mask_freezeup[istart] = True
#                                     elif (date_start.month < 5) & (freezeup_date[iyr-1,1] < 5) :
#                                         freezeup_date[iyr,0] = date_start.year
#                                         freezeup_date[iyr,1] = date_start.month
#                                         freezeup_date[iyr,2] = date_start.day
#                                         freezeup_Tw[iyr] = Twater_in[istart]
#                                         mask_freezeup[istart] = True

#                                 elif date_start.year == freezeup_date[iyr-1,0]+2:
#                                     if (date_start.month < 5):
#                                         freezeup_date[iyr,0] = date_start.year
#                                         freezeup_date[iyr,1] = date_start.month
#                                         freezeup_date[iyr,2] = date_start.day
#                                         freezeup_Tw[iyr] = Twater_in[istart]
#                                         mask_freezeup[istart] = True

#                                 else:
#                                     print(iyr)
#                                     print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)
#                                     # if (date_start.year > freezeup_date[iyr-1,0]+2):
#                                     #     freezeup_date[iyr,0] = date_start.year
#                                     #     freezeup_date[iyr,1] = date_start.month
#                                     #     freezeup_date[iyr,2] = date_start.day
#                                     #     mask_freezeup[istart] = True
#                                     # else:
#                                     #     print(iyr)
#                                     #     print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)


#     Twater_out = Twater_in.copy()
#     Twater_out[~mask_freezeup] = np.nan

#     return freezeup_date, freezeup_Tw, Twater_out, mask_freezeup

###
def find_freezeup_Tw_all_yrs(def_opt,Twater_in,dTdt_in,d2Tdt2_in,time,years,thresh_T = 2.0,thresh_dTdt = 0.2,thresh_d2Tdt2 = 0.2,ndays = 7,date_ref = dt.date(1900,1,1)):

    def record_event(istart,time,years,Twater_in,freezeup_date,freezeup_Tw,mask_freezeup,date_ref):
        # Temperature has been lower than thresh_T
        # for more than (or equal to) ndays.
        # Define freezeup date as first date of group

        date_start = date_ref+dt.timedelta(days=int(time[istart]))
        doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

        if ((date_start.year > 1992) | ((date_start.year == 1992) & (date_start.month > 10)) ):

            if (doy_start > 319) | (doy_start < 46):
                if (np.where(np.array(years) == date_start.year-1*(doy_start < 46))[0].size > 0):
                    iyr = np.where(np.array(years) == date_start.year-1*(doy_start < 46))[0][0]

                if np.isnan(freezeup_date[iyr,0]):
                    if np.isnan(freezeup_date[iyr-1,0]):
                        freezeup_date[iyr,0] = date_start.year
                        freezeup_date[iyr,1] = date_start.month
                        freezeup_date[iyr,2] = date_start.day
                        freezeup_Tw[iyr] = Twater_in[istart]
                        mask_freezeup[istart] = True
                    else:
                        if freezeup_date[iyr-1,0] == date_start.year:
                            # 2012 01 05
                            # 2012 01 07 NO
                            # 2012 12 22 YES
                            if (freezeup_date[iyr-1,1] < 5) & (date_start.month > 10):
                                freezeup_date[iyr,0] = date_start.year
                                freezeup_date[iyr,1] = date_start.month
                                freezeup_date[iyr,2] = date_start.day
                                freezeup_Tw[iyr] = Twater_in[istart]
                                mask_freezeup[istart] = True

                        elif date_start.year == freezeup_date[iyr-1,0]+1:
                            # 2012 12 22
                            # 2013 01 14 NO
                            # 2013 12 24 YES

                            #2014 01 03 (2013 season)
                            #2015 01 13 (2014 season)
                            if (date_start.month > 10):
                                freezeup_date[iyr,0] = date_start.year
                                freezeup_date[iyr,1] = date_start.month
                                freezeup_date[iyr,2] = date_start.day
                                freezeup_Tw[iyr] = Twater_in[istart]
                                mask_freezeup[istart] = True
                            elif (date_start.month < 5) & (freezeup_date[iyr-1,1] < 5) :
                                freezeup_date[iyr,0] = date_start.year
                                freezeup_date[iyr,1] = date_start.month
                                freezeup_date[iyr,2] = date_start.day
                                freezeup_Tw[iyr] = Twater_in[istart]
                                mask_freezeup[istart] = True

                        elif date_start.year == freezeup_date[iyr-1,0]+2:
                            if (date_start.month < 5):
                                freezeup_date[iyr,0] = date_start.year
                                freezeup_date[iyr,1] = date_start.month
                                freezeup_date[iyr,2] = date_start.day
                                freezeup_Tw[iyr] = Twater_in[istart]
                                mask_freezeup[istart] = True

                        else:
                            print(iyr)
                            print('PROBLEM!!!!!!!! : ',iyr,int(freezeup_date[iyr-1,0]),int(freezeup_date[iyr-1,1]),int(freezeup_date[iyr-1,2]),date_start.year,date_start.month,date_start.day)

        return freezeup_date, freezeup_Tw, mask_freezeup


    if def_opt == 1: # T is below thresh_T:
        mask_tmp = Twater_in <= thresh_T

    if (def_opt == 2):# T is below thresh_T and dTdt is below thresh_dTdt:
        zero_dTdt_mask = np.abs(dTdt_in) <= thresh_dTdt
        zero_T_mask = Twater_in <= thresh_T
        # zero_d2Tdt2_mask = np.abs(d2Tdt2_in) <= thresh_d2Tdt2
        # mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
        mask_tmp = zero_T_mask & zero_dTdt_mask

    if (def_opt == 3): # T is below thresh_T and dTdt is below thresh_dTdt and d2Tdt2 is below thresh_d2Tdt2:
        zero_dTdt_mask = np.abs(dTdt_in) <= thresh_dTdt
        zero_T_mask = Twater_in <= thresh_T
        # zero_d2Tdt2_mask = np.abs(d2Tdt2_in) <= thresh_d2Tdt2
        # mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
        mask_tmp = zero_T_mask & zero_dTdt_mask

    mask_freezeup = mask_tmp.copy()
    mask_freezeup[:] = False

    freezeup_Tw=np.zeros((len(years)))*np.nan
    freezeup_date=np.zeros((len(years),3))*np.nan

    for im in range(mask_freezeup.size):

        if (im == 0):
            sum_m = 0
            istart = -1 # This ensures that a freeze-up is not detected if the time series started already below the freezing temp.
        else:

            if (~mask_tmp[im-1]):
                sum_m = 0
                if ~mask_tmp[im]:
                    sum_m = 0
                else:
                    # start new group
                    sum_m +=1
                    istart = im
                    # Below will only occur if ndays is set to 1, e.g. first day of freezing temp.
                    if (sum_m >= ndays):
                        freezeup_date,freezeup_Tw,mask_freezeup = record_event(istart,time,years,Twater_in,freezeup_date,freezeup_Tw,mask_freezeup,date_ref)
            else:
                if (mask_tmp[im]) & (istart > 0):
                    sum_m += 1
                    if (sum_m >= ndays):
                        freezeup_date,freezeup_Tw,mask_freezeup = record_event(istart,time,years,Twater_in,freezeup_date,freezeup_Tw,mask_freezeup,date_ref)

    Twater_out = Twater_in.copy()
    Twater_out[~mask_freezeup] = np.nan

    return freezeup_date, freezeup_Tw, Twater_out, mask_freezeup
#                      fd, ftw,        T_freezeup, mask_freeze
###
def find_freezeup_Tw(def_opt,Twater_in,dTdt_in,d2Tdt2_in,time,year,thresh_T = 2.0,thresh_dTdt = 0.2,thresh_d2Tdt2 = 0.2,ndays = 7,date_ref = dt.date(1900,1,1)):

    def record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref):
        # Temperature has been lower than thresh_T
        # for more than (or equal to) ndays.
        # Define freezeup date as first date of group

        date_start = date_ref+dt.timedelta(days=int(time[istart]))
        doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

        if ((date_start.year > 1992) | ((date_start.year == 1992) & (date_start.month > 10)) ):
            if ( (date_start.year == year) & (doy_start > 319) ) | ((date_start.year == year+1) & (doy_start < 46)):
                    freezeup_date[0] = date_start.year
                    freezeup_date[1] = date_start.month
                    freezeup_date[2] = date_start.day
                    freezeup_Tw = Twater_in[istart]
                    mask_freezeup[istart] = True
            else:
                freezeup_date[0] = np.nan
                freezeup_date[1] = np.nan
                freezeup_date[2] = np.nan
                freezeup_Tw = np.nan
                mask_freezeup[istart] = False
        else:
        # I think this condition exists because the Tw time series
        # starts in January 1992, so it is already frozen, but we
        # do not want to detect this as freezeup for 1992, so we
        # have to wait until at least October 1992 before recording
        # any FUD events.
            freezeup_date[0] = np.nan
            freezeup_date[1] = np.nan
            freezeup_date[2] = np.nan
            freezeup_Tw = np.nan
            mask_freezeup[istart] = False

        return freezeup_date, freezeup_Tw, mask_freezeup


    if def_opt == 1: # T is below thresh_T:
        mask_tmp = Twater_in <= thresh_T

    if (def_opt == 2):# T is below thresh_T and dTdt is below thresh_dTdt:
        zero_dTdt_mask = np.abs(dTdt_in) <= thresh_dTdt
        zero_T_mask = Twater_in <= thresh_T
        # zero_d2Tdt2_mask = np.abs(d2Tdt2_in) <= thresh_d2Tdt2
        # mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
        mask_tmp = zero_T_mask & zero_dTdt_mask

    if (def_opt == 3): # T is below thresh_T and dTdt is below thresh_dTdt and d2Tdt2 is below thresh_d2Tdt2:
        zero_dTdt_mask = np.abs(dTdt_in) <= thresh_dTdt
        zero_T_mask = Twater_in <= thresh_T
        # zero_d2Tdt2_mask = np.abs(d2Tdt2_in) <= thresh_d2Tdt2
        # mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask
        mask_tmp = zero_T_mask & zero_dTdt_mask

    mask_freezeup = mask_tmp.copy()
    mask_freezeup[:] = False

    freezeup_Tw = np.nan
    freezeup_date=np.zeros((3))*np.nan

    for im in range(mask_freezeup.size):

        if (im == 0):
            sum_m = 0
            istart = -1 # This ensures that a freeze-up is not detected if the time series started already below the freezing temp.
        else:
            if (np.sum(mask_freezeup) == 0): # Only continue while no prior freeze-up was detected for the sequence
                if (~mask_tmp[im-1]):
                    sum_m = 0
                    if ~mask_tmp[im]:
                        sum_m = 0
                    else:
                        # start new group
                        sum_m +=1
                        istart = im
                        # Below will only occur if ndays is set to 1, e.g. first day of freezing temp.
                        if (sum_m >= ndays):
                            freezeup_date,freezeup_Tw,mask_freezeup = record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref)
                else:
                    if (mask_tmp[im]) & (istart > 0):
                        sum_m += 1
                        if (sum_m >= ndays):
                            freezeup_date,freezeup_Tw,mask_freezeup = record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref)

    Twater_out = Twater_in.copy()
    Twater_out[~mask_freezeup] = np.nan

    return freezeup_date, freezeup_Tw, Twater_out, mask_freezeup


###
def find_breakup_Tw(def_opt,Twater_in,dTdt_in,d2Tdt2_in,time,years,thresh_T = 2.0,thresh_dTdt = 0.2,thresh_d2Tdt2 = 0.2,ndays = 7,date_ref = dt.date(1900,1,1)):

    # mask_tmp is True if T is above thresh_T:
    mask_tmp = Twater_in >= thresh_T


    if def_opt == 1:
        # T is above thresh_T:
        mask_tmp = Twater_in >= thresh_T

    if (def_opt == 2):
        # T is above thresh_T and dTdt is above thresh_dTdt and d2Tdt2 is above thresh_d2Tdt2:
        zero_T_mask = Twater_in >= thresh_T
        zero_dTdt_mask = np.abs(dTdt_in) >= thresh_dTdt
        zero_d2Tdt2_mask = np.abs(d2Tdt2_in) >= thresh_d2Tdt2
        mask_tmp = zero_T_mask & zero_dTdt_mask & zero_d2Tdt2_mask

    if (def_opt == 3):
        # T is above thresh_T and dTdt is above thresh_dTdt:
        zero_T_mask = Twater_in >= thresh_T
        zero_dTdt_mask = np.abs(dTdt_in) >= thresh_dTdt
        mask_tmp = zero_T_mask & zero_dTdt_mask


    mask_breakup = mask_tmp.copy()
    mask_breakup[:] = False

    breakup_Tw=np.zeros((len(years)))*np.nan
    breakup_date=np.zeros((len(years),3))*np.nan
    iyr = 0
    flag = 0 # This flag will become 1 if the data time series starts after the freezing season, to indicate that we cannot use the first date of temp. below freezing point as the freezeup date.


    for iyr in range(len(years)):

        if len(np.where(time == (dt.date(years[iyr],1,1)-date_ref).days)) > 0:
            iJan1 = np.where(time == (dt.date(years[iyr],1,1)-date_ref).days)[0][0]

            for it,im in enumerate(np.arange(iJan1,iJan1+150)):
                # print('ITTTTTT: ', it, years[iyr])
                if (it == 0) | (~mask_tmp[im-1]):

                    sum_m = 0
                    if ~mask_tmp[im]:
                        sum_m = 0
                    else:
                        # start new group
                        sum_m +=1
                        istart = im

                        # Below will only occur if ndays is set to 1, e.g. first day of freezing temp.
                        if (sum_m >= ndays):
                            # Temperature has been higher than thresh_T
                            # for more than (or equal to) ndays.
                            # Define breakupp date as first date of group

                            date_start = date_ref+dt.timedelta(days=int(time[istart]))
                            doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

                            if np.isnan(breakup_date[iyr-1,0]):

                                if date_start.year == years[iyr]: # This should always be the case

                                    if years[iyr] == 2018:
                                        # if (doy_start > 31) & (doy_start < 110):
                                        if np.nanmean(Twater_in[istart:istart+30]) > 2.3:
                                            if np.sum(np.isnan(Twater_in[istart-7:istart])) < 7:
                                                breakup_date[iyr-1,0] = date_start.year
                                                breakup_date[iyr-1,1] = date_start.month
                                                breakup_date[iyr-1,2] = date_start.day
                                                breakup_Tw[iyr-1] = Twater_in[istart]
                                                mask_breakup[istart] = True
                                        else:
                                            # print('FLAG 3 - 2018:  ', doy_start,np.nanmean(Twater_in[istart:istart+30]))
                                            continue
                                    else:
                                        # if (doy_start > 31) & (doy_start < 110):
                                        if np.nanmean(Twater_in[istart:istart+30]) > 2.5:
                                            if np.sum(np.isnan(Twater_in[istart-7:istart])) < 7:
                                                breakup_date[iyr-1,0] = date_start.year
                                                breakup_date[iyr-1,1] = date_start.month
                                                breakup_date[iyr-1,2] = date_start.day
                                                breakup_Tw[iyr-1] = Twater_in[istart]
                                                mask_breakup[istart] = True
                                        else:
                                            # print('FLAG 3:  ', doy_start,np.nanmean(Twater_in[istart:istart+30]))
                                            continue
                                else:
                                    print('PROBLEM!!!!')



                        # if it == 0:
                        #     flag = 1 # This flag indicates that the time series has started above freezing temperatures already, so that we cannot use the first date of temp. above freezing point as the breakup date.
                        #     print('FLAG 1: Series started with Tw > 0')
                        #     continue
                        # else:
                        #     # start new group
                        #     sum_m +=1
                        #     istart = im

                        #     # Below will only occur if ndays is set to 1, e.g. first day of freezing temp.
                        #     if (sum_m >= ndays):
                        #         # ADD THE SAME AS BELOW TO TERMINTATE GROUP AND REGISTER FIRST DAT OF ABOVE FREEZING AS BREAKUP DATE.
                        #         print('NEED TO WRITE THIS')


                else:
                    if mask_tmp[im]:
                        sum_m += 1

                        if (sum_m >= ndays):
                            # Temperature has been higher than thresh_T
                            # for more than (or equal to) ndays.
                            # Define breakupp date as first date of group

                            date_start = date_ref+dt.timedelta(days=int(time[istart]))
                            doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

                            if np.isnan(breakup_date[iyr-1,0]):

                                if date_start.year == years[iyr]: # This should always be the case

                                    if years[iyr] == 2018:
                                        # if (doy_start > 31) & (doy_start < 110):
                                        if np.nanmean(Twater_in[istart:istart+30]) > 2.3:
                                            if np.sum(np.isnan(Twater_in[istart-7:istart])) < 7:
                                                breakup_date[iyr-1,0] = date_start.year
                                                breakup_date[iyr-1,1] = date_start.month
                                                breakup_date[iyr-1,2] = date_start.day
                                                breakup_Tw[iyr-1] = Twater_in[istart]
                                                mask_breakup[istart] = True
                                        else:
                                            # print('FLAG 2 - 2018:  ', doy_start,np.nanmean(Twater_in[istart:istart+30]))
                                            continue
                                    else:
                                        # if (doy_start > 31) & (doy_start < 110):
                                        if np.nanmean(Twater_in[istart:istart+30]) > 2.5:
                                            if np.sum(np.isnan(Twater_in[istart-7:istart])) < 7:
                                                breakup_date[iyr-1,0] = date_start.year
                                                breakup_date[iyr-1,1] = date_start.month
                                                breakup_date[iyr-1,2] = date_start.day
                                                breakup_Tw[iyr-1] = Twater_in[istart]
                                                mask_breakup[istart] = True
                                        else:
                                            # print('FLAG 2:  ', doy_start,np.nanmean(Twater_in[istart:istart+30]))
                                            continue
                                else:
                                    print('PROBLEM!!!!')


    Twater_out = Twater_in.copy()
    Twater_out[~mask_breakup] = np.nan

    return breakup_date, breakup_Tw, Twater_out, mask_breakup


###
def fill_gaps(var_in, ndays = 7, fill_type = 'linear'):

    # mask_tmp is true if there is no data (i.e. Tw is nan):
    mask_tmp = np.isnan(var_in)

    mask_gap = mask_tmp.copy()
    mask_gap[:] = False

    var_out = var_in.copy()

    for im in range(1,mask_gap.size):

        if (im == 1) | (~mask_tmp[im-1]):
            # start new group
            sum_m = 0
            if ~mask_tmp[im]:
                sum_m = 0
            else:
                sum_m +=1
                istart = im

        else:
            if mask_tmp[im]:
                sum_m += 1
            else:
                # This is the end of the group of constant dTdt,
                # so count the total number of points in group,
                # and remove whole group if total is larger than
                # ndays
                iend = im
                if sum_m < ndays:
                    mask_gap[istart:iend] = True

                    if fill_type == 'linear':
                        # Fill small gap with linear interpolation
                        slope = (var_out[iend]-var_out[istart-1])/(sum_m+1)
                        var_out[istart:iend] = var_out[istart-1] + slope*(np.arange(sum_m)+1)

                    else:
                        print('Problem! ''fill_type'' not defined...')

                sum_m = 0 # Put back sum to zero


    return var_out, mask_gap

###
def get_window_vars(vars_in,names_in,end_dates,window_arr,years,time,month_start_day=1,date_ref=dt.date(1900,1,1)):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),len(window_arr),2))*np.nan
    window_size = window_arr[1]-window_arr[0]

    for iyr, year in enumerate(years):

        i0 = (dt.date(int(year),1,month_start_day)-date_ref).days
        i0 = np.where(time == i0)[0][0]

        i1 = (dt.date(int(year)+1,3,month_start_day)-date_ref).days
        try:
            i1 = np.where(time == i1)[0][0]
        except:
            i1 = len(time)-1

        doy0 = (dt.date(int(year),1,month_start_day)-(dt.date(int(year),1,1))).days + 1
        doy_arr = np.arange(doy0, doy0+(i1-i0))

        if ~np.isnan(end_dates[iyr]):
            for iw,w in enumerate(window_arr):
                # window_type == 'moving':
                ied = np.where(doy_arr == end_dates[iyr])[0][0]
                ifd = ied-(iw)*(window_size)
                iw0 = ied-(iw+1)*(window_size)

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,iw,0] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        if np.all(np.isnan(var_year[iw0:ifd])):
                            vars_out[ivar,iyr,iw,0] = np.nan
                        else:
                            vars_out[ivar,iyr,iw,0] = np.nansum(var_year[iw0:ifd])
                        # BELOW: TEST TO REPLACE ZERO SNOWFALL WITH NAN TO REMOVE THIS POINT FROM THE CORRELATION
                        # if np.nansum(var_year[iw0:ifd]) == 0:
                        #     vars_out[ivar,iyr,iw,0] = np.nan
                        # else:
                        #     vars_out[ivar,iyr,iw,0] = np.nansum(var_year[iw0:ifd])

                # window_type == 'increasing':
                ifd = np.where(doy_arr == end_dates[iyr])[0][0]
                iw0 = ifd-w

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,iw,1] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        if np.all(np.isnan(var_year[iw0:ifd])):
                            vars_out[ivar,iyr,iw,1] = np.nan
                        else:
                            vars_out[ivar,iyr,iw,1] = np.nansum(var_year[iw0:ifd])

                        # BELOW: TEST TO REPLACE ZERO SNOWFALL WITH NAN TO REMOVE THIS POINT FROM THE CORRELATION
                        # if np.nansum(var_year[iw0:ifd]) == 0:
                        #     vars_out[ivar,iyr,iw,1] = np.nan
                        # else:
                        #     vars_out[ivar,iyr,iw,1] = np.nansum(var_year[iw0:ifd])

    return vars_out


###
def get_window_monthly_vars(vars_in,names_in,end_dates,window_arr,years,time,month_start_day=1,date_ref=dt.date(1900,1,1)):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),len(window_arr),2))*np.nan

    for iyr, year in enumerate(years):

        i0 = (dt.date(int(year),1,month_start_day)-date_ref).days
        i0 = np.where(time == i0)[0][0]

        i1 = (dt.date(int(year)+1,3,month_start_day)-date_ref).days
        try:
            i1 = np.where(time == i1)[0][0]
        except:
            i1 = len(time)-1

        doy0 = (dt.date(int(year),1,month_start_day)-(dt.date(int(year),1,1))).days + 1
        doy_arr = np.arange(doy0, doy0+(i1-i0))

        if ~np.isnan(end_dates[iyr]):
            month_end = (dt.date(year,1,1)+dt.timedelta(days=int(end_dates[iyr]-1))).month
            iend = np.where(doy_arr == end_dates[iyr])[0][0]

            for imonth in range(len(window_arr)):
                month = month_end-(imonth+1)
                doy_month_1st = (dt.date(year,month,1)-dt.date(year,1,1)).days+1
                imonth_1st =  np.where(doy_arr == doy_month_1st)[0][0]
                doy_monthp1_1st = (dt.date(year,month+1,1)-dt.date(year,1,1)).days+1
                imonthp1_1st = np.where(doy_arr == doy_monthp1_1st)[0][0]

                # window_type == 'moving':
                ifd = imonthp1_1st
                iw0 = imonth_1st

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,imonth,0] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        if np.all(np.isnan(var_year[iw0:ifd])):
                            vars_out[ivar,iyr,imonth,0] = np.nan
                        else:
                            vars_out[ivar,iyr,imonth,0] = np.nansum(var_year[iw0:ifd])

                    if (varname[0:3] == 'Max'):
                        vars_out[ivar,iyr,imonth,0] = np.nanmax(var_year[iw0:ifd])

                    if (varname[0:3] == 'Min'):
                        vars_out[ivar,iyr,imonth,0] = np.nanmin(var_year[iw0:ifd])


                # window_type == 'increasing':
                ifd = iend
                iw0 = imonth_1st

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,imonth,1] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        if np.all(np.isnan(var_year[iw0:ifd])):
                            vars_out[ivar,iyr,imonth,1] = np.nan
                        else:
                            vars_out[ivar,iyr,imonth,1] = np.nansum(var_year[iw0:ifd])

                    if (varname[0:3] == 'Max'):
                        vars_out[ivar,iyr,imonth,1] = np.nanmax(var_year[iw0:ifd])

                    if (varname[0:3] == 'Min'):
                        vars_out[ivar,iyr,imonth,1] = np.nanmin(var_year[iw0:ifd])

    return vars_out


###
def deseasonalize_ts(Nwindow,vars_in,varnames,time_spec,time,years):
    vars_out = np.zeros(vars_in.shape)*np.nan

    for ivar in range(len(varnames)):
        var_mean, var_std, weather_window = rolling_climo(Nwindow,vars_in[:,ivar],time_spec,time,years)
        vars_out[:,ivar] = vars_in[:,ivar]-var_mean

    return vars_out


###
def detect_FUD_from_Tw(fp,loc_list,station_type,freezeup_opt,years,time,show=False,return_FUD_dates = False):

    if show:
        fig, ax = plt.subplots()

    # OPTION 1: Tw < 0.75 C for 1 day.
    if freezeup_opt == 1:
        def_opt = 1
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = False
        T_thresh = 0.75
        dTdt_thresh = 0.25
        d2Tdt2_thresh = 0.25
        nd = 1

    # OPTION 2: Tw and DoG below threshold for 30 days.
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

    # OPTION 3: TEST!!!!!!!!!
    if freezeup_opt == 3:
        def_opt = 3
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = True
        sig_dog = 3.5
        T_thresh = 30.
        dTdt_thresh = 0.15
        d2Tdt2_thresh = 0.15
        # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
        # d2Tdt2_thresh = 0.20
        nd = 30

    # OPTION 4: TEST!!!!!!!!!
    if freezeup_opt == 4:
        def_opt = 1
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = False
        T_thresh = 1.0
        dTdt_thresh = 0.25
        d2Tdt2_thresh = 0.25
        nd = 1

    freezeup_dates = np.zeros((len(years),3,len(loc_list)))*np.nan
    freezeup_doy = np.zeros((len(years),len(loc_list)))*np.nan

    Twater = np.zeros((len(time),len(loc_list)))*np.nan
    Twater_dTdt = np.zeros((len(time),len(loc_list)))*np.nan
    Twater_d2Tdt2 = np.zeros((len(time),len(loc_list)))*np.nan
    Twater_DoG1 = np.zeros((len(time),len(loc_list)))*np.nan
    Twater_DoG2 = np.zeros((len(time),len(loc_list)))*np.nan

    for iloc,loc in enumerate(loc_list):
        water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc+'_cleaned_filled.npz',allow_pickle='TRUE')
        Twater_tmp = water_loc_data['Twater'][:,1]

        # APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST
        Twater[:,iloc] = Twater_tmp
        if loc == 'Candiac':
            Twater[:,iloc] = Twater_tmp-0.8
        if (loc == 'Atwater'):
            Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7
        if (loc == 'Longueuil_updated'):
            Twater[14329:,iloc] = Twater_tmp[14329:]- 0.78

        # THEN FIND DTDt, D2TDt2, etc.
        Twater_tmp = Twater[:,iloc].copy()
        if round_T:
            if round_type == 'unit':
                Twater_tmp = np.round(Twater_tmp.copy())
            if round_type == 'half_unit':
                Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
        if smooth_T:
            Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

        dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

        dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
        dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
        dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

        Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
        Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

        if Gauss_filter:
            Twater_DoG1[:,iloc] = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
            Twater_DoG2[:,iloc] = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

        # THEN FIND FREEZEUP ACCORDING TO CHOSEN OPTION:
        if def_opt == 3:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_DoG1[:,iloc],Twater_DoG2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
        else:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd

        # FINALLY, TRANSFORM FREEZEUP FROM DATE FORMAT TO DOY FORMAT:
        for iyr,year in enumerate(years):
            if ~np.isnan(freezeup_dates[iyr,0,iloc]):
                fd_yy = int(freezeup_dates[iyr,0,iloc])
                fd_mm = int(freezeup_dates[iyr,1,iloc])
                fd_dd = int(freezeup_dates[iyr,2,iloc])

                if fd_mm < 3:
                    fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
                    if fd_doy < 60:
                        fd_doy += 365
                else:
                    fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1 -1*(calendar.isleap(fd_yy))

                freezeup_doy[iyr,iloc]=fd_doy

        if show:
            ax.plot(Twater_tmp)
            ax.plot(T_freezeup,'o')

    if return_FUD_dates:
        return  freezeup_doy, Twater, freezeup_dates
    else:
        return freezeup_doy, Twater


###
def detect_FUD_from_Tw_clim(Tw_clim,freezeup_opt,years,time,show=False):
    from functions import running_nanmean

    if show:
        fig, ax = plt.subplots()

    # OPTION 1: Tw < 0.75 C for 1 day.
    if freezeup_opt == 1:
        def_opt = 1
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = False
        T_thresh = 0.75
        dTdt_thresh = 0.25
        d2Tdt2_thresh = 0.25
        nd = 1

    # OPTION 2: Tw and DoG below threshold for 30 days.
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

    # OPTION 3: TEST!!!!!!!!!
    if freezeup_opt == 3:
        def_opt = 3
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = True
        sig_dog = 3.5
        T_thresh = 30.
        dTdt_thresh = 0.15
        d2Tdt2_thresh = 0.15
        # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
        # d2Tdt2_thresh = 0.20
        nd = 30

    # OPTION 4: TEST!!!!!!!!!
    if freezeup_opt == 4:
        def_opt = 1
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = False
        T_thresh = 1.0
        dTdt_thresh = 0.25
        d2Tdt2_thresh = 0.25
        nd = 1

    freezeup_dates = np.zeros((len(years),3,1))*np.nan
    freezeup_doy = np.zeros((len(years),1))*np.nan

    Twater_dTdt = np.zeros((len(time),1))*np.nan
    Twater_d2Tdt2 = np.zeros((len(time),1))*np.nan
    Twater_DoG1 = np.zeros((len(time),1))*np.nan
    Twater_DoG2 = np.zeros((len(time),1))*np.nan

    # FIND DTDt, D2TDt2, etc.
    iloc = 0
    Twater_tmp = Tw_clim.copy()
    if round_T:
        if round_type == 'unit':
            Twater_tmp = np.round(Twater_tmp.copy())
        if round_type == 'half_unit':
            Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
    if smooth_T:
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

    if Gauss_filter:
        Twater_DoG1[:,iloc] = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
        Twater_DoG2[:,iloc] = sp.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

    # THEN FIND FREEZEUP ACCORDING TO CHOSEN OPTION:
    if def_opt == 3:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_DoG1[:,iloc],Twater_DoG2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_dates[:,:,iloc] = fd
    else:
        fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
        freezeup_dates[:,:,iloc] = fd

    # FINALLY, TRANSFORM FREEZEUP FROM DATE FORMAT TO DOY FORMAT:
    for iyr,year in enumerate(years):
        if ~np.isnan(freezeup_dates[iyr,0,iloc]):
            fd_yy = int(freezeup_dates[iyr,0,iloc])
            fd_mm = int(freezeup_dates[iyr,1,iloc])
            fd_dd = int(freezeup_dates[iyr,2,iloc])

            if fd_mm < 3:
                fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
                if fd_doy < 60:
                    fd_doy += 365
            else:
                fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1

            freezeup_doy[iyr,iloc]=fd_doy

    if show:
        ax.plot(Twater_tmp)
        ax.plot(T_freezeup,'o')

    return freezeup_doy
