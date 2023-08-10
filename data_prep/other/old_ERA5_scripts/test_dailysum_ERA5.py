#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:02:07 2021

@author: Amelie
"""


import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import datetime as dt
from functions import haversine, ncdump
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir='/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/temp_files/') #python

#%%

# def compute_dailysum_var():
#     import time, sys
#     from datetime import datetime, timedelta

#     from netCDF4 import Dataset, date2num, num2date
#     import numpy as np

#     day = 20170101
#     d = datetime.strptime(str(day), '%Y%m%d')
#     f_in = 'tp_%d-%s.nc' % (day, (d + timedelta(days = 1)).strftime('%Y%m%d'))
#     f_out = 'daily-tp_%d.nc' % day

#     time_needed = []
#     for i in range(1, 25):
#         time_needed.append(d + timedelta(hours = i))

#     with Dataset(f_in) as ds_src:
#         var_time = ds_src.variables['time']
#         time_avail = num2date(var_time[:], var_time.units,
#                 calendar = var_time.calendar)

#         indices = []
#         for tm in time_needed:
#             a = np.where(time_avail == tm)[0]
#             if len(a) == 0:
#                 sys.stderr.write('Error: precipitation data is missing/incomplete - %s!\n'
#                         % tm.strftime('%Y%m%d %H:%M:%S'))
#                 sys.exit(200)
#             else:
#                 print('Found %s' % tm.strftime('%Y%m%d %H:%M:%S'))
#                 indices.append(a[0])

#         var_tp = ds_src.variables['tp']
#         tp_values_set = False
#         for idx in indices:
#             if not tp_values_set:
#                 data = var_tp[idx, :, :]
#                 tp_values_set = True
#             else:
#                 data += var_tp[idx, :, :]

#         with Dataset(f_out, mode = 'w', format = 'NETCDF3_64BIT_OFFSET') as ds_dest:
#             # Dimensions
#             for name in ['latitude', 'longitude']:
#                 dim_src = ds_src.dimensions[name]
#                 ds_dest.createDimension(name, dim_src.size)
#                 var_src = ds_src.variables[name]
#                 var_dest = ds_dest.createVariable(name, var_src.datatype, (name,))
#                 var_dest[:] = var_src[:]
#                 var_dest.setncattr('units', var_src.units)
#                 var_dest.setncattr('long_name', var_src.long_name)

#             ds_dest.createDimension('time', None)
#             var = ds_dest.createVariable('time', np.int32, ('time',))
#             time_units = 'hours since 1900-01-01 00:00:00'
#             time_cal = 'gregorian'
#             var[:] = date2num([d], units = time_units, calendar = time_cal)
#             var.setncattr('units', time_units)
#             var.setncattr('long_name', 'time')
#             var.setncattr('calendar', time_cal)

#             # Variables
#             var = ds_dest.createVariable(var_tp.name, np.double, var_tp.dimensions)
#             var[0, :, :] = data
#             var.setncattr('units', var_tp.units)
#             var.setncattr('long_name', var_tp.long_name)

#             # Attributes
#             ds_dest.setncattr('Conventions', 'CF-1.6')
#             ds_dest.setncattr('history', '%s %s'
#                     % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                     ' '.join(time.tzname)))

#             print('Done! Daily total precipitation saved in %s' % f_out)

#%%
# region = 'A' # ST_LAWRENCE
# region = 'B' # LAKE ONTARIO
# region = 'C' # OTTAWA RIVER + MONTREAL
region = 'D' # ALL
# region = 'E' # OTTAWA RIVER ONLY

start_year = 1991
end_year = 2022


path = '../../data/raw/ERA5_hourly/region'+region+'/'
save_path = '../../data/processed/ERA5_hourly/region'+region+'/'

verbose = False

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#%%
var_list      = ['u10',                    'v10',                    't2m',           't2m',           't2m',           'd2m',                    'licd',          'lict',                'ltlt',                        'msl',                    'ro',    'siconc',       'sst',                    'sf',      'smlt',    'tcc',              'tp']
savename_list = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_temperature','2m_temperature','2m_dewpoint_temperature','lake_ice_depth','lake_ice_temperature','lake_total_layer_temperature','mean_sea_level_pressure','runoff','sea_ice_cover','sea_surface_temperature','snowfall','snowmelt','total_cloud_cover','total_precipitation']
vartype_list  = ['mean',                   'mean',                   'mean',          'max',           'min',           'mean',                   'mean',          'mean',                'mean',                        'mean',                   'mean',  'mean',         'mean',                   'mean',    'mean',    'mean',             'mean']

# var_list      = ['t2m',           't2m',           't2m',           'd2m',                    'licd',          'lict']
# savename_list = ['2m_temperature','2m_temperature','2m_temperature','2m_dewpoint_temperature','lake_ice_depth','lake_ice_temperature']
# vartype_list  = ['mean',          'min',           'max',           'mean',                   'mean',          'mean']

#%%
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

lon = np.arange(-93,-58+0.25,0.25)
lat = np.arange(40,53+0.25,0.25)

rlon = np.arange(rlon1,rlon2+0.25,0.25)
rlat = np.arange(rlat1,rlat2+0.25,0.25)


#%%
path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/ERA5_hourly/'
yyyy = 1994
mm = 2
yyyy_str = str(yyyy)
mm_str = str(mm).rjust(2, '0')
fname = 'ERA5_snowfall_'+yyyy_str+mm_str+'.nc'
var = 'sf'

filename = path+'region'+region+'/'+yyyy_str+'-'+mm_str+'/'+fname
ncid = Dataset(filename, 'r')
var_nc2 = ncid.variables[var][:]
var_nc2[var_nc2<0] = 0

path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/ERA5_hourly/'
yyyy = 1994
mm = 3
yyyy_str = str(yyyy)
mm_str = str(mm).rjust(2, '0')
fname = 'ERA5_snowfall_'+yyyy_str+mm_str+'.nc'
var = 'sf'

filename = path+'region'+region+'/'+yyyy_str+'-'+mm_str+'/'+fname
ncid = Dataset(filename, 'r')
var_nc3 = ncid.variables[var][:]
var_nc3[var_nc3 < 0] = 0

var_dailysum = np.zeros((int(var_nc2.shape[0]/24),var_nc2.shape[1],var_nc2.shape[2]))
for iday in range(int(var_nc2.shape[0]/24)-1):
    var_dailysum[iday,:,:] = np.nansum(var_nc2[1+iday*24:1+(iday+1)*24,:,:],axis=0)
var_dailysum[-1,:,:] = np.squeeze(np.nansum(var_nc2[-23:,:,:],axis=0)+var_nc3[0,:,:])



#%%
# region = 'A' # ST_LAWRENCE
# region = 'B' # LAKE ONTARIO
# region = 'C' # OTTAWA RIVER + MONTREAL
region = 'D' # ALL
# region = 'E' # OTTAWA RIVER ONLY

start_year = 1991
end_year = 2002


path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/ERA5_hourly/region'+region+'/'
save_path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/ERA5_hourly/region'+region+'/test/'

verbose = False

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#%%
var_list      = [ 'ro',   'sf',      'smlt',    'tp']
savename_list = ['runoff','snowfall','snowmelt','total_precipitation']
vartype_list  = ['sum',   'sum',     'sum',     'sum']

var_list      = ['ro']
savename_list = ['runoff']
vartype_list  = ['sum']



#%%
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

lon = np.arange(-93,-58+0.25,0.25)
lat = np.arange(40,53+0.25,0.25)

rlon = np.arange(rlon1,rlon2+0.25,0.25)
rlat = np.arange(rlat1,rlat2+0.25,0.25)

#%%

for ivar,var in enumerate(var_list):

    var_out = np.zeros((len(time)))*np.nan

    for year in range(start_year, end_year+1):
        for month in range(1,13):

            month_str = str(month).rjust(2, '0') # '01' instead of '1'
            fdirectory = path+"{}-{}/".format(year, month_str)

            if os.path.isdir(fdirectory):

                filename = fdirectory + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month_str+'.nc'

                if os.path.isfile(filename):
                    print(savename_list[ivar]+'_'+str(year)+month_str)

                    ncid = Dataset(filename, 'r')
                    ncid.set_auto_mask(False)
                    time_tmp = ncid.variables['time'][:]
                    time_tmp = time_tmp[0:time_tmp.size:24]
                    ncid.close()

                    if vartype_list[ivar] == 'mean':
                        vdaily = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymean(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=filename)))), returnArray = var))
                        if len(vdaily.shape) > 1:
                            # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                            # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                            # equal to 1 for ERA5 and equal to 5 for ERA5T.
                            # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                            vdaily = np.nanmean(vdaily,axis=1)
                    elif vartype_list[ivar] == 'min':
                        vdaily = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymin(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=filename)))), returnArray = var))
                        if len(vdaily.shape) > 1:
                            # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                            # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                            # equal to 1 for ERA5 and equal to 5 for ERA5T.
                            # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                            vdaily = np.nanmean(vdaily,axis=1)
                    elif vartype_list[ivar] == 'max':
                        vdaily = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.daymax(input=cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,input=cdo.selname(var,input=filename)))), returnArray = var))
                        if len(vdaily.shape) > 1:
                            # This happens when the netcdf contains both ERA5 and ERA5T (near real time, for data of the last 3 months)
                            # The dimensions of the variables are then ['time', 'expver', 'latitude', 'longitude'], where 'expver' is
                            # equal to 1 for ERA5 and equal to 5 for ERA5T.
                            # We can then simply take the mean over the 'expver' dimension, since most of the time both  expver will not co-exist.
                            vdaily = np.nanmean(vdaily,axis=1)

                    elif vartype_list[ivar] == 'sum':
                        # Daily accumulated variables need to be summed from 01:00 to 00:00 the next day because the raw variables are accumulated at the end of the record period (i.e. accumulated rain at 00:00 is rain that fell between 23:00 and 00:00)
                        if month < 12:
                            month1 = str(month).rjust(2, '0') # '01' instead of '1'
                            fdirectory1 = path+"{}-{}/".format(year, month1)
                            filename1 = fdirectory1 + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month1+'.nc'
                            ncid1 = Dataset(filename1, 'r')
                            var_nc1 = ncid1.variables[var][:]
                            var_nc1[var_nc1 < 0] = 0
                            if len(var_nc1.shape) > 3:
                                var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))

                            month2 = str(month+1).rjust(2, '0') # '01' instead of '1'
                            fdirectory2 = path+"{}-{}/".format(year, month2)
                            filename2 = fdirectory2 + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month2+'.nc'
                            ncid2 = Dataset(filename2, 'r')
                            var_nc2 = ncid2.variables[var][:]
                            var_nc2[var_nc2 < 0] = 0
                            if len(var_nc2.shape) > 3:
                                var_nc2 = np.squeeze(np.nanmean(var_nc2,axis=1))

                            vdaily = np.zeros((int(var_nc1.shape[0]/24),var_nc1.shape[1],var_nc1.shape[2]))
                            for iday in range(int(var_nc1.shape[0]/24)-1):
                                vdaily[iday,:,:] = np.nansum(var_nc1[1+iday*24:1+(iday+1)*24,:,:],axis=0)
                            vdaily[-1,:,:] = np.squeeze(np.nansum(var_nc1[-23:,:,:],axis=0)+var_nc2[0,:,:])

                            vdaily = np.nanmean(np.nanmean(vdaily,axis=2), axis=1)

                        else:
                            if year == end_year:
                                # Last day sum will be missing one hour of accumulation...
                                month1 = str(month).rjust(2, '0') # '01' instead of '1'
                                fdirectory1 = path+"{}-{}/".format(year, month1)
                                filename1 = fdirectory1 + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month1+'.nc'
                                ncid1 = Dataset(filename1, 'r')
                                var_nc1 = ncid1.variables[var][:]
                                var_nc1[var_nc1 < 0] = 0
                                if len(var_nc1.shape) > 3:
                                    var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))

                                vdaily = np.zeros((int(var_nc1.shape[0]/24),var_nc1.shape[1],var_nc1.shape[2]))*np.nan
                                for iday in range(int(var_nc1.shape[0]/24)-1):
                                    vdaily[iday,:,:] = np.nansum(var_nc1[1+iday*24:1+(iday+1)*24,:,:],axis=0)
                                vdaily[-1,:,:] = np.nansum(var_nc1[1+(iday+1)*24:1+(iday+1+1)*24,:,:],axis=0)
                                vdaily = np.nanmean(np.nanmean(vdaily,axis=2), axis=1)

                            else:
                                # Add Jan. 1st 00:00 of next year to accumulated variable on Dec. 31st.
                                month1 = str(month).rjust(2, '0') # '01' instead of '1'
                                fdirectory1 = path+"{}-{}/".format(year, month1)
                                filename1 = fdirectory1 + 'ERA5_'+savename_list[ivar]+'_'+str(year)+month1+'.nc'
                                ncid1 = Dataset(filename1, 'r')
                                var_nc1 = ncid1.variables[var][:]
                                var_nc1[var_nc1 < 0] = 0
                                if len(var_nc1.shape) > 3:
                                    var_nc1 = np.squeeze(np.nanmean(var_nc1,axis=1))

                                month2 = str(1).rjust(2, '0') # '01' instead of '1'
                                fdirectory2 = path+"{}-{}/".format(year+1, month2)
                                filename2 = fdirectory2 + 'ERA5_'+savename_list[ivar]+'_'+str(year+1)+month2+'.nc'
                                ncid2 = Dataset(filename2, 'r')
                                var_nc2 = ncid2.variables[var][:]
                                var_nc2[var_nc2 < 0] = 0
                                if len(var_nc2.shape) > 3:
                                    var_nc2 = np.squeeze(np.nanmean(var_nc2,axis=1))

                                vdaily = np.zeros((int(var_nc1.shape[0]/24),var_nc1.shape[1],var_nc1.shape[2]))
                                for iday in range(int(var_nc1.shape[0]/24)-1):
                                    vdaily[iday,:,:] = np.nansum(var_nc1[1+iday*24:1+(iday+1)*24,:,:],axis=0)
                                vdaily[-1,:,:] = np.squeeze(np.nansum(var_nc1[-23:,:,:],axis=0)+var_nc2[0,:,:])

                                vdaily = np.nanmean(np.nanmean(vdaily,axis=2), axis=1)


                    # Correct units:
                    if var == 'msl':
                        vdaily = vdaily/1000. # Convert to kPa
                    if (var == 't2m') | (var == 'd2m') | (var == 'lict') | (var == 'ltlt'):
                        vdaily  = (vdaily-273.15)# Convert Kelvins to Celsius


                    # Then, arrange variables in the same format as weather from NCEI (i.e. [it,var])
                    for it in range(time_tmp.size):
                        date_it = date_ref+dt.timedelta(hours=int(time_tmp[it]))
                        new_time = (date_it - date_ref).days
                        if (new_time <= time[-1]) & (new_time >= time[0]):
                            itvar = np.where(time == int(new_time))[0][0]
                            var_out[itvar] = vdaily[it]

                    cdo.cleanTempDir()

    # Finally, save as npy file
    savename ='ERA5_daily'+vartype_list[ivar]+'_'+savename_list[ivar]
    np.savez(save_path+savename,data=var_out)


# # MEAKE NEW VARIABLES FROM THE COMBINATION OF ORIGINAL ONES

# Load original variables
u10 = np.load(save_path+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
v10 = np.load(save_path+'ERA5_dailymean_10m_u_component_of_wind.npz')['data']
avg_Ta = np.load(save_path+'ERA5_dailymean_2m_temperature.npz')['data']
avg_Td = np.load(save_path+'ERA5_dailymean_2m_dewpoint_temperature.npz')['data']
slp = np.load(save_path+'ERA5_dailymean_mean_sea_level_pressure.npz')['data']

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

# Save new variables
np.savez(save_path+'ERA5_dailymean_windspeed',data=windspeed)
np.savez(save_path+'ERA5_dailymean_RH',data=avg_RH)
np.savez(save_path+'ERA5_dailymean_SH',data=avg_SH)
np.savez(save_path+'ERA5_dailymean_FDD',data=FDD)
np.savez(save_path+'ERA5_dailymean_TDD',data=TDD)


#%%

data_24hr = np.load(save_path+'ERA5_dailymean_snowfall.npz')['data']
data_4xday = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/ERA5/regionD/'+'ERA5_dailymean_snowfall.npz')['data']
data_sum = np.load(save_path+'ERA5_dailysum_snowfall.npz')['data']

plt.figure();plt.plot(data_24hr);plt.plot(data_4xday)

plt.figure();plt.plot(data_sum)


