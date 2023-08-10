#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:29:32 2022

@author: amelie
"""

local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import calendar
from netCDF4 import Dataset
import netCDF4

import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

from functions import ncdump
from functions import K_to_C

import warnings

from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir=local_path +'slice/prog/temp_files/') #python


#%%
class SEAS5frcst(object):
    import warnings

    def __init__(self, filename):
        self.ds = netCDF4.Dataset(filename)
        self.vars = [v for v in self.ds.variables]
        self.var = [v for v in self.ds.variables][-1]
        self.time_units = self.ds['time'].units
        self.units = self.ds[self.var].units
        if self.var == 'sf':
            self.long_name = self.ds[self.var].long_name
            self.std_name = self.ds[self.var].standard_name
        self.filepath = self.ds.filepath()
        self.base_dir = '/'.join(self.ds.filepath().split('/')[:-2]+[''])
        self.fname = self.ds.filepath().split('/')[-1]
        self.feature = '_'.join(self.ds.filepath().split('/')[-1].split('_')[1:-1])
        self.year = int(self.ds.filepath().split('/')[-1].split('_')[-1][-9:-3][0:4])
        self.month = int(self.ds.filepath().split('/')[-1].split('_')[-1][-9:-3][4:])
        self.region = self.ds.filepath().split('/')[-3][-1]

    def get_time(self,time_rep='daily',time_format=None):
        import datetime as dt
        if time_rep == 'monthly':
            if ( self.time_units == 'hours since 1900-01-01 00:00:00.0'):
                date_ref = dt.date(1900,1,1)
                if time_format == 'ordinal':
                    t_out = None
                    raise Exception('ERROR... CANNOT CONVERT MONTHLY TIME VECTOR TO ORDINAL FORMAT! CHANGE TIME FORMAT TO <None> or <array>')

                elif time_format == 'array':
                    month = [(date_ref + dt.timedelta(hours=int(self.ds['time'][i]))).month for i in range(len(self.ds['time'][:]))]
                    nmonths = len(np.unique(month))
                    t_out = np.zeros((nmonths,2))

                    m = 0
                    for i in range(len(self.ds['time'][:])):
                        date_i = (date_ref + dt.timedelta(hours=int(self.ds['time'][i])))
                        if m == 0:
                            t_out[m,0] = date_i.year
                            t_out[m,1] = date_i.month
                            m += 1
                        else:
                            if t_out[m-1,1] != date_i.month :
                                t_out[m,0] = date_i.year
                                t_out[m,1] = date_i.month
                                m += 1
                elif time_format == 'plot':
                    month = [(date_ref + dt.timedelta(hours=int(self.ds['time'][i]))).month for i in range(len(self.ds['time'][:]))]
                    nmonths = len(np.unique(month))
                    t_out = np.zeros((nmonths,2))

                    m = 0
                    for i in range(len(self.ds['time'][:])):
                        date_i = (date_ref + dt.timedelta(hours=int(self.ds['time'][i])))
                        if m == 0:
                            t_out[m,0] = date_i.year
                            t_out[m,1] = date_i.month
                            m += 1
                        else:
                            if t_out[m-1,1] != date_i.month :
                                t_out[m,0] = date_i.year
                                t_out[m,1] = date_i.month
                                m += 1

                    t_out = np.array([dt.date(int(t_out[i,0]),int(t_out[i,1]),1) for i in range(nmonths)])
                else:
                    t_out = []
                    t_out.append((date_ref + dt.timedelta(hours=int(self.ds['time'][0]))).month)
                    for i in range(len(self.ds['time'][:])):
                        month = (date_ref + dt.timedelta(hours=int(self.ds['time'][i]))).month
                        if month not in t_out:
                            t_out.append(month)
                    t_out = np.array(t_out)
            else:
                t_out = None
                raise Exception('ERROR WITH TIME AXIS UNITS.... CANNOT COMPUTE MONTHLY-AVERAGED TIME STEPS.')


        if time_rep == 'daily':
            if ( self.time_units == 'hours since 1900-01-01 00:00:00.0'):
                date_ref = dt.date(1900,1,1)
                if time_format == 'ordinal':
                    first_day = (date_ref+dt.timedelta(hours=int(self.ds['time'][0]))).toordinal()
                    last_day = (date_ref+dt.timedelta(hours=int(self.ds['time'][-1]))).toordinal()
                    t_out = np.arange(first_day, last_day + 1)
                elif time_format == 'array':
                    t_out = np.zeros((len(self.ds['time'][:]),3))
                    for i in range(len(self.ds['time'][:])):
                        date_i = (date_ref + dt.timedelta(hours=int(self.ds['time'][i])))
                        t_out[i,0] = date_i.year
                        t_out[i,1] = date_i.month
                        t_out[i,2] = date_i.day
                elif time_format == 'plot':
                    t_out = np.zeros((len(self.ds['time'][:]),3))
                    for i in range(len(self.ds['time'][:])):
                        date_i = (date_ref + dt.timedelta(hours=int(self.ds['time'][i])))
                        t_out[i,0] = date_i.year
                        t_out[i,1] = date_i.month
                        t_out[i,2] = date_i.day

                    t_out = np.array([(dt.date(int(t_out[i,0]),int(t_out[i,1]),int(t_out[i,2]))) for i in range(len(t_out))])
                else:
                    t_out = self.ds['time'][:]
            else:
                t_out = None
                raise Exception('ERROR WITH TIME AXIS UNITS.... CANNOT COMPUTE MONTHLY-AVERAGED TIME STEPS.')


        return t_out

    def read_vars(self,vnames,
                  time_rep = 'daily',
                  time_format = None,
                  lead = 'all',
                  spatial_avg=False,
                  latmin=None,latmax=None,lonmin=None,lonmax=None,
                  ensemble_avg=False):

        from cdo import Cdo
        cdo = Cdo()

        try:
            vnames = vnames.split()
        except:
            pass

        vout = []
        for v in vnames:
            # 1) Select variable from netcdf and perform spatio-temporal average:
            if (v == 'time'):
                vtmp = self.get_time(time_rep,time_format)

            elif (v == 'longitude') | (v == 'latitude') :
                vtmp = self.ds[v][:]

            else:
                if spatial_avg:
                    if ('longitude' in self.ds[v].dimensions) & ('latitude' in self.ds[v].dimensions):
                        # Average on all lat/lon available if no bounds are specified...
                        if ((latmin == None) | (latmax == None) | (lonmin == None) | (lonmax == None)):
                            # Also perform monthly average at the same time if specified:
                            if (time_rep == 'monthly'):

                                if ('time' in self.ds[v].dimensions):

                                    if (v == 'sf') | ( v == 'tp') | ( v == 'ro') :
                                        if (v == 'sf'):
                                            # Cannot use CDO here because it doesn't recognize the new netcdf format that was made for the daily snowfall increment.
                                            vall = np.nanmean(self.ds[v], axis=(2,3))
                                            ttmp = self.get_time(time_rep='daily',time_format='array')
                                            month_tmp = ttmp[:,1]
                                            vtmp = np.zeros((8,51))*np.nan
                                            for il in range(8):
                                                l = self.month-1+il-12*((self.month-1+il)//12)
                                                y = np.nansum(vall[np.where(month_tmp == (l+1))[0],:],axis=0)
                                                vtmp[il,0:y.shape[0]] = y
                                        else:
                                            vtmp = np.squeeze(cdo.monsum(input=cdo.zonmean(input=cdo.mermean(input=cdo.selname(v,input=self.ds.filepath()))), returnArray = v, options = "-b 64"))
                                    else:
                                        vtmp = np.squeeze(cdo.monmean(input=cdo.zonmean(input=cdo.mermean(input=cdo.selname(v,input=self.ds.filepath()))), returnArray = v))
                                else:
                                    warnings.warn("*** WARNING: Monthly mean of variable "+ v +" could not be perfomed. The output is the raw data.")
                                    vtmp = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.selname(v,input=self.ds.filepath())), returnArray = v))
                            else:
                                if v == 'sf':
                                    # Cannot use CDO here because it doesn't recognize the new netcdf format that was made for the daily snowfall increment.
                                    vtmp = np.nanmean(self.ds[v], axis=(2,3))
                                else:
                                    vtmp = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.selname(v,input=self.ds.filepath())), returnArray = v))

                        # ... otherwise, select a lat/lon box to average in:
                        else:
                            if (v == 'sf'):
                                raise Exception('Lat-lon selection on snowfall data is not yet implemented... Sorry!')
                            else:
                                # and again, perform the monthly average at the same time if specified:
                                if (time_rep == 'monthly'):
                                    if ('time' in self.ds[v].dimensions):
                                        if (v == 'sf') | ( v == 'tp') | ( v == 'ro') :
                                            vtmp = np.squeeze(cdo.monsum(input=cdo.zonmean(input=cdo.mermean(input=cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,input=cdo.selname(v,input=self.ds.filepath())))), returnArray = v, options = "-b 64"))
                                        else:
                                            vtmp = np.squeeze(cdo.monmean(input=cdo.zonmean(input=cdo.mermean(input=cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,input=cdo.selname(v,input=self.ds.filepath())))), returnArray = v))
                                    else:
                                        warnings.warn("*** WARNING: Monthly mean of variable "+ v +" could not be perfomed. The output is the raw data.")
                                        vtmp = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,input=cdo.selname(v,input=self.ds.filepath()))), returnArray = v))
                                else:
                                    vtmp = np.squeeze(cdo.zonmean(input=cdo.mermean(input=cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,input=cdo.selname(v,input=self.ds.filepath()))), returnArray = v))

                    else: #(No spatial averaging can be done on variables that do not have a latitude and longitude dimension)
                        warnings.warn("*** WARNING: Spatial averaging of variable "+ v +" could not be perfomed. The output is the raw data.")
                        if (time_rep == 'monthly'):
                            if ('time' in self.ds[v].dimensions):
                                if (v == 'sf') | ( v == 'tp') | ( v == 'ro') :
                                    if (v == 'sf'):
                                        # Cannot use CDO here because it doesn't recognize the new netcdf format that was made for the daily snowfall increment.
                                        vall = self.ds[v]
                                        ttmp = self.get_time(time_rep='daily',time_format='array')
                                        month_tmp = ttmp[:,1]
                                        vtmp = np.zeros((8,51,vall.shape[2],vall.shape[3]))*np.nan
                                        for il in range(8):
                                            l = self.month-1+il-12*((self.month-1+il)//12)
                                            y = np.nansum(vall[np.where(month_tmp == (l+1))[0],:,:,:],axis=0)
                                            vtmp[il,0:y.shape[0],:,:] = y
                                    else:
                                        vtmp = np.squeeze(cdo.monsum(input=cdo.selname(v,input=self.ds.filepath()), returnArray = v, options = "-b 64"))
                                else:
                                    vtmp = np.squeeze(cdo.monmean(input=cdo.selname(v,input=self.ds.filepath()), returnArray = v))
                            else:
                                warnings.warn("*** WARNING: Monthly mean of variable "+ v +" could not be perfomed. The output is the raw data.")
                                vtmp = np.squeeze(self.ds[v][:])
                        # Or simply return the selected variable from the netcdf:
                        else:
                            vtmp = np.squeeze(self.ds[v][:])

                # If no spatial averaging is done,
                else:
                    # then temporally average separately if specified...
                    if (time_rep == 'monthly'):
                        if ('time' in self.ds[v].dimensions):
                            if (v == 'sf') | ( v == 'tp') | ( v == 'ro') :
                                if (v == 'sf'):
                                    # Cannot use CDO here because it doesn't recognize the new netcdf format that was made for the daily snowfall increment.
                                    vall = self.ds[v]
                                    ttmp = self.get_time(time_rep='daily',time_format='array')
                                    month_tmp = ttmp[:,1]
                                    vtmp = np.zeros((8,51,vall.shape[2],vall.shape[3]))*np.nan
                                    for il in range(8):
                                        l = self.month-1+il-12*((self.month-1+il)//12)
                                        y = np.nansum(vall[np.where(month_tmp == (l+1))[0],:,:,:],axis=0)
                                        vtmp[il,0:y.shape[0],:,:] = y
                                else:
                                    vtmp = np.squeeze(cdo.monsum(input=cdo.selname(v,input=self.ds.filepath()), returnArray = v, options = "-b 64"))
                            else:
                                vtmp = np.squeeze(cdo.monmean(input=cdo.selname(v,input=self.ds.filepath()), returnArray = v))
                        else:
                            warnings.warn("*** WARNING: Monthly mean of variable "+ v +" could not be perfomed. The output is the raw data.")
                            vtmp = np.squeeze(self.ds[v][:])
                    # Or simply return the selected variable from the netcdf:
                    else:
                        vtmp = np.squeeze(self.ds[v][:])

                cdo.cleanTempDir()

            # 2) Then average all ensemble members, if specified:
            if ensemble_avg:
                if ('number' in self.ds[v].dimensions):
                    idim = np.where(np.array(self.ds[v].dimensions) == 'number')[0][0]
                    vtmp = np.nanmean(vtmp, axis=idim)

            # 3) Finally, select which lead day/month to keep:
            if lead != 'all':
                if ('time' in self.ds[v].dimensions):
                    if len(np.array(lead)) > 1:
                        l0 = lead[0]
                        l1 = lead[-1]
                        if (l0 > 0) & (l1 > 0):
                            idim = np.where(np.array(self.ds[v].dimensions) == 'time')[0][0]
                            vtmp = vtmp.take(indices=np.arange(l0-1,l1), axis=idim)
                        else:
                            raise Exception('Lead time error - Lead needs to be larger than zero.')
                    else:
                        l = lead
                        if (l > 0):
                            idim = np.where(np.array(self.ds[v].dimensions) == 'time')[0][0]
                            vtmp = vtmp.take(indices=l-1, axis=idim)
                        else:
                            raise Exception('Lead time error - Lead needs to be larger than zero.')



            # 4) Append final variable to the output list
            vout.append(vtmp)

        # Keep only the first variable of the list if only one variable was asked:
        if len(vnames) == 1:
            vout = vout[0]

        return vout


    def get_all_years(self,
                      yrs = None,
                      time_rep = 'daily',
                      lead = 'all',
                      spatial_avg=False,
                      latmin=None,latmax=None,lonmin=None,lonmax=None,
                      ):

        # If no years were provided to use for computing climatology, set it to
        # all possible hindcast years
        if yrs is None:
            yr_start = 1981
            yr_end = 2022
            yrs = np.arange(yr_start,yr_end+1)

        nyrs = len(yrs)
        if lead == 'all':
            if time_rep == 'monthly':
                nleads = 8
            if time_rep == 'daily':
                nleads = 215
        else:
            nleads = len(lead)

        v_arr = np.zeros((nyrs,nleads,51))*np.nan
        for iyr, year in enumerate(yrs):
            # Get ensemble mean forecast for that year for the same start month
            fname_yr = self.base_dir + "{}-{}/".format(year, str(self.month).rjust(2, '0'))+self.fname[:-9]+"{}{}.nc".format(year, str(self.month).rjust(2, '0'))

            if os.path.isfile(fname_yr):
                s_yr = SEAS5frcst(fname_yr)
                v_em_yr = s_yr.read_vars(s_yr.var,
                                      spatial_avg=spatial_avg,
                                      lead = lead,
                                      latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax,
                                      ensemble_avg=False,
                                      time_rep=time_rep
                                      )
                # Add to clim_arr
                v_arr[iyr,:,0:v_em_yr.shape[1]] = v_em_yr

        return v_arr



    def get_climatology(self,
                        clim_yrs = None,
                        time_rep = 'daily',
                        lead = 'all',
                        spatial_avg=False,
                        latmin=None,latmax=None,lonmin=None,lonmax=None,
                        ):

        # If no years were provided to use for computing climatology, set it to
        # hindcast years
        if clim_yrs is None:
            if self.year >= 2022:
                climyr_start = 1981
                climyr_end = 2016
            else:
                climyr_start = 1993
                climyr_end = 2016
            clim_yrs = np.arange(climyr_start,climyr_end+1)

        clim_arr = self.get_all_years(yrs = clim_yrs,
                                  time_rep = time_rep,
                                  lead = lead,
                                  spatial_avg=spatial_avg,
                                  latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax,
                                  )

        clim_mean = np.nanmean(clim_arr,axis=(0,2))
        clim_std = np.nanstd(clim_arr,axis=(0,2))
        clim_p33 = np.squeeze(np.nanpercentile(clim_arr,100/3.,axis=(0,2)))
        clim_p66 = np.squeeze(np.nanpercentile(clim_arr,200/3.,axis=(0,2)))

        return clim_mean, clim_std, clim_p33, clim_p66, clim_arr


        # v = self.read_vars(self.var,
        #               time_rep = time_rep,
        #               lead = lead,
        #               spatial_avg=spatial_avg,
        #               latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax,
        #               ensemble_avg=False
        #               )

        # # If no years were provided to use for computing climatology, set it to
        # # hindcast years
        # if clim_yrs is None:
        #     if self.year >= 2022:
        #         climyr_start = 1981
        #         climyr_end = 2016
        #     else:
        #         climyr_start = 1993
        #         climyr_end = 2019
        #     clim_yrs = np.arange(climyr_start,climyr_end+1)


        # nclimyrs = len(clim_yrs)
        # clim_arr = np.zeros((nclimyrs,v.shape[0],51))*np.nan
        # # print(self.base_dir,self.fname,self.year,self.month,v.shape,clim_arr.shape)

        # for iyr, year in enumerate(clim_yrs):
        #     # Get ensemble mean forecast for that year for the same start month
        #     fname_yr = self.base_dir + "{}-{}/".format(year, str(self.month).rjust(2, '0'))+self.fname[:-9]+"{}{}.nc".format(year, str(self.month).rjust(2, '0'))
        #     s_yr = SEAS5frcst(fname_yr)
        #     v_em_yr = s_yr.read_vars(s_yr.var,
        #                           spatial_avg=spatial_avg,
        #                           lead = lead,
        #                           latmin=latmin,latmax=latmax,lonmin=lonmin,lonmax=lonmax,
        #                           ensemble_avg=False,
        #                           time_rep=time_rep
        #                           )
        #     # Add to clim_arr
        #     clim_arr[iyr,:,0:v_em_yr.shape[1]] = v_em_yr

        # # Now find mean and std daily values over all years and all members:
        # clim_mean = np.nanmean(clim_arr,axis=(0,2))
        # clim_std = np.nanstd(clim_arr,axis=(0,2))
        # clim_p33 = np.squeeze(np.nanpercentile(clim_arr,100/3.,axis=(0,2)))
        # clim_p66 = np.squeeze(np.nanpercentile(clim_arr,200/3.,axis=(0,2)))


        # return clim_mean, clim_std, clim_p33, clim_p66, clim_arr



#%%
if __name__ == "__main__":
    r_dir = local_path + 'slice/data/raw/SEAS5/'
    region = 'D'
    base = 'SEAS5'

    # feature = 'mean_sea_level_pressure'
    # feature = 'minimum_2m_temperature_in_the_last_24_hours'
    # feature = 'maximum_2m_temperature_in_the_last_24_hours'
    # feature = '2m_dewpoint_temperature'
    feature = '2m_temperature'
    # feature = '10m_u_component_of_wind'
    # feature = '10m_v_component_of_wind'
    # feature = 'runoff'
    # feature = 'snowfall'
    # feature = 'total_cloud_cover'
    # feature = 'total_precipitation'

    # feature = 'sea_ice_cover'
    # feature = 'sea_surface_temperature'

    year = 2000
    month = 11

    extension = "{}{}.nc".format(year, str(month).rjust(2, '0'))

    path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month).rjust(2, '0'))
    fname = base + '_' + feature + '_' + extension

    # Initialize forecast class
    s = SEAS5frcst(path+fname)
    spatial_avg = True
    ensemble_avg = True
    time_rep='monthly'
    # time_rep='daily'
    time_format='plot'
    # lonmin =
    # lonmax =
    # latmin =
    # latmax =

    # Get ensemble mean
    var_em, time = s.read_vars([s.var,'time'],
                                 spatial_avg=spatial_avg,
                                 ensemble_avg=True,
                                 time_rep=time_rep,
                                 time_format=time_format
                               )
    var_em = K_to_C(var_em)

    # Get all members
    var, time = s.read_vars([s.var,'time'],
                              spatial_avg=spatial_avg,
                              ensemble_avg=False,
                              time_rep=time_rep,
                              time_format=time_format
                            )
    var = K_to_C(var)

    # Plot all members + ensemble mean on top
    fig,ax = plt.subplots()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
    for n in range(25):
        ax.plot(time,var[:,n], color='gray',linewidth=0.75,alpha=0.25)
        # ax.plot(time,var[n], color='gray',linewidth=0.75,alpha=0.25)
    ax.plot(time,var_em,'-',color='black',linewidth=2)

    # Plot climatology for the given start month
    ax.plot(time,var_em,'-',color=plt.get_cmap('tab20')(1))
    v_yr = s.get_climatology(spatial_avg=spatial_avg,time_rep=time_rep)
    var_clim_mean = K_to_C(v_yr[0])
    # var_clim_mean = v_yr[0]
    ax.plot(time,var_clim_mean,'--',color=plt.get_cmap('tab20')(0))

    # Make new figure with anomalies w/r to the climatology
    var_anom = var - var_clim_mean[:, np.newaxis]
    var_em_anom = var_em - var_clim_mean

    fig_anom,ax_anom = plt.subplots()
    ax_anom.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
    ax_anom.axhline(y=0, color='gray', linestyle=':',linewidth=1)
    for n in range(25):
        ax_anom.plot(time,var_anom[:,n], color='gray',linewidth=0.75,alpha=0.25)
    ax_anom.plot(time,var_em_anom,'-',color='black',linewidth=2)
    # ax_anom.grid(visible='on',axis='y')

    #%%

    r_dir = local_path + 'slice/data/raw/SEAS5/'
    region = 'D'
    base = 'SEAS5'

    # feature = 'mean_sea_level_pressure'
    # feature = 'minimum_2m_temperature_in_the_last_24_hours'
    # feature = 'maximum_2m_temperature_in_the_last_24_hours'
    # feature = '2m_dewpoint_temperature'
    # feature = '2m_temperature'
    # feature = '10m_u_component_of_wind'
    # feature = '10m_v_component_of_wind'
    # feature = 'runoff'
    feature = 'snowfall'
    # feature = 'total_cloud_cover'
    # feature = 'total_precipitation'

    # feature = 'sea_ice_cover'
    # feature = 'sea_surface_temperature'


    year = 1999
    fig,ax = plt.subplots()
    for month in range(1,13):

        extension = "{}{}.nc".format(year, str(month).rjust(2, '0'))

        path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month).rjust(2, '0'))
        fname = base + '_' + feature + '_' + extension

        # Initialize forecast class
        s = SEAS5frcst(path+fname)

        # Get ensemble mean
        var_em, time = s.read_vars([s.var,'time'],
                                    spatial_avg=True,
                                    ensemble_avg=True,
                                    lead = [1,31],
                                    # time_rep='monthly',
                                    time_format='plot'
                                    )
        if 'temperature' in feature: var_em = K_to_C(var_em)

        # Get all members
        var, time = s.read_vars([s.var,'time'],
                                  spatial_avg=True,
                                  lead = [1,31],
                                  # time_rep='monthly',
                                  time_format='plot'
                                )
        if 'temperature' in feature: var = K_to_C(var)


        # Get climatology
        v_yr = s.get_climatology(1993,2019,
                                 spatial_avg=True,
                                 lead = [1,31],
                                 # time_rep='monthly'
                                 )
        var_clim_mean = v_yr[0]

        if 'temperature' in feature: var_clim_mean = K_to_C(var_clim_mean)


        # Plot all members + ensemble mean on top
        if month < 11:
            month_plot = month
        else:
            month_plot = month-10
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
        for n in range(var.shape[1]):
            ax.plot(time,var[:,n], color=plt.get_cmap('tab20')((2*(month_plot-1))+1),linewidth=0.75,alpha=0.25)
        ax.plot(time,var_em,'-',color=plt.get_cmap('tab20')((2*(month_plot-1))),linewidth=2)
        ax.plot(time,var_clim_mean,'--',color='gray',linewidth=1)




    #%%
    # CHECK SOME VARIABLES SNAPSHOTS
    # l = Dataset(r_dir+'landmask.nc','r')
    # landmask = np.nanmean(np.squeeze( l['lsm'][:]),axis=0)
    # lat_mask = np.squeeze( l['latitude'][:])
    # lon_mask = np.squeeze( l['longitude'][:])
    # plt.figure()
    # plt.pcolormesh(lon_mask,lat_mask,landmask)
    # #%%
    # si = Dataset(r_dir+'seaicecover_200201.nc','r')
    # sic = np.nanmean(np.squeeze( si['siconc'][:]),axis=1)
    # lat_si = np.squeeze( si['latitude'][:])
    # lon_si = np.squeeze( si['longitude'][:])
    # plt.figure()
    # plt.pcolormesh(lon_si,lat_si,sic[30,:,:])
    # #%%
    # st = Dataset(r_dir+'sst_200201.nc','r')
    # sst = np.nanmean(np.squeeze( st['sst'][:]),axis=1)
    # lat_sst = np.squeeze( st['latitude'][:])
    # lon_sst = np.squeeze( si['longitude'][:])
    # plt.figure()
    # plt.pcolormesh(lon_sst,lat_sst,sst[50,:,:])

    #%%
    import pandas as pd
    filepath = 'slice/data/colab/'
    df = pd.read_excel(local_path+filepath+'predictor_data_daily_timeseries.xlsx')

    # The dates column is not needed
    time = df['Days since 1900-01-01'].values
    date_ref = dt.date(1900,1,1)
    #%%
    r_dir = local_path + 'slice/data/raw/SEAS5/'
    region = 'D'
    base = 'SEAS5'
    feature = '2m_temperature'

    batch_size = 512
    input_len = 128
    pred_len = 60
    predictor_vars = ['Avg. Ta_max',
                      'Avg. Ta_min',
                      'Tot. snowfall',
                      'NAO',
                      'Avg. Twater'
                      ]
    forecast_vars = ['Avg. Ta_mean']
    # forecast_vars = []

    target_var = ['Avg. Twater']

    train_yr_start = 1993 # Training dataset: 1992 - 2010
    valid_yr_start = 2011 # Validation dataset: 2011 - 2015
    test_yr_start = 2016 # Testing dataset: 2016 - 2021


    #%%
    fSEAS5 = np.zeros((len(time),12))*np.nan
    SEAS5_years = np.arange(1993,2021+1)

    for year in SEAS5_years:
        for imonth in range(12):
            month = imonth + 1
            print(year, month)
            extension = "{}{}.nc".format(year, str(month).rjust(2, '0'))

            path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month).rjust(2, '0'))
            fname = base + '_' + feature + '_' + extension

            # Initialize forecast class
            s = SEAS5frcst(path+fname)
            spatial_avg = True
            ensemble_avg = True
            time_rep='daily'
            time_format='plot'

            # Get ensemble mean
            var_em, _ = s.read_vars([s.var,'time'],
                                         spatial_avg=spatial_avg,
                                         ensemble_avg=True,
                                         time_rep=time_rep,
                                         time_format=time_format
                                       )
            var_em = K_to_C(var_em)


            it_month_start = np.where(time == (dt.date(year,month,1)-date_ref).days)[0][0]
            it_month_end = np.where(time == (dt.date(year,month,+1)-date_ref).days)[0][0]

            if (year == SEAS5_years[-1]) and (month == 11):
                fSEAS5[it_month_start:it_month_start+61,imonth] = var_em[0:61]
            elif (year == SEAS5_years[-1]) and (month == 12):
                fSEAS5[it_month_start:it_month_start+31,imonth] = var_em[0:31]
            else:
                fSEAS5[it_month_start:it_month_start+31+pred_len,imonth] = var_em[0:31+pred_len]

    #%%
    fSEAS5_monthly = np.zeros((len(time),12))*np.nan
    SEAS5_years = np.arange(1993,2021+1)

    for year in SEAS5_years[28:]:
        for imonth in range(12):
            month = imonth + 1
            print(year, month)
            extension = "{}{}.nc".format(year, str(month).rjust(2, '0'))

            path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month).rjust(2, '0'))
            fname = base + '_' + feature + '_' + extension

            # Initialize forecast class
            s = SEAS5frcst(path+fname)
            spatial_avg = True
            ensemble_avg = True
            time_rep='daily'
            time_format='plot'

            # Get ensemble mean
            var_em, _ = s.read_vars([s.var,'time'],
                                         spatial_avg=spatial_avg,
                                         ensemble_avg=True,
                                         time_rep=time_rep,
                                         time_format=time_format
                                       )
            var_em = K_to_C(var_em)


            it_month_start = np.where(time == (dt.date(year,month,1)-date_ref).days)[0][0]
            it_month_end = np.where(time == (dt.date(year,month,+1)-date_ref).days)[0][0]

            fSEAS5_monthly[it_month_start:it_month_start+31,imonth] = var_em[0:31]



    #%%
    import tensorflow as tf

    df = pd.read_excel(local_path+filepath+'predictor_data_daily_timeseries.xlsx')

    # The dates column is not needed
    time = df['Days since 1900-01-01'].values
    date_ref = dt.date(1900,1,1)
    df.drop(columns='Days since 1900-01-01', inplace=True)
    #%%
    # Keep only data for 1992-2020.
    yr_start = 1993
    yr_end = 2020
    date_ref = dt.date(1900,1,1)
    it_start = np.where(time == (dt.date(yr_start,1,1)-date_ref).days)[0][0]
    it_end = np.where(time == (dt.date(yr_end+1,1,1)-date_ref).days)[0][0]

    df = df.iloc[it_start:it_end,:]
    time = time[it_start:it_end]
    first_day = (date_ref+dt.timedelta(days=int(time[0]))).toordinal()
    last_day = (date_ref+dt.timedelta(days=int(time[-1]))).toordinal()
    time_plot = np.arange(first_day, last_day + 1)

    df['Avg. Twater'][9886:9946] = 0
    df['Avg. Twater'][df['Avg. Twater'] < 0] = 0
    df['Tot. FDD'][np.isnan(df['Tot. FDD'])] = 0
    df['Tot. TDD'][np.isnan(df['Tot. TDD'])] = 0
    df = df[target_var+predictor_vars+forecast_vars]

    # GET TRAINING AND VALIDATION DATA SETS
    train_years = np.arange(train_yr_start,valid_yr_start)
    valid_years = np.arange(valid_yr_start,test_yr_start)
    test_years = np.arange(test_yr_start,yr_end+1)

    istart_train = np.where(time_plot == dt.date(train_yr_start, 4, 1).toordinal())[0][0]
    istart_valid = np.where(time_plot == dt.date(valid_yr_start, 4, 1).toordinal())[0][0]
    istart_test = np.where(time_plot == dt.date(test_yr_start, 4, 1).toordinal())[0][0]
    ind_train = np.arange(istart_train,istart_valid)
    ind_valid = np.arange(istart_valid,istart_test)
    ind_test = np.arange(istart_test,len(time))

    time_train = time[ind_train]
    time_valid = time[ind_valid]
    time_test = time[ind_test]
    time_train_plot = time_plot[ind_train]
    time_valid_plot = time_plot[ind_valid]
    time_test_plot = time_plot[ind_test]

    df_train = df.iloc[ind_train]
    df_valid = df.iloc[ind_valid]
    df_test = df.iloc[ind_test]

    df_SEAS5 = pd.DataFrame(fSEAS5,columns=['SEAS5 - Jan.','SEAS5 - Feb.','SEAS5 - Mar.','SEAS5 - Apr.','SEAS5 - May','SEAS5 - Jun.','SEAS5 - Jul.','SEAS5 - Aug.','SEAS5 - Sep.','SEAS5 - Oct.','SEAS5 - Nov.','SEAS5 - Dec.'])
    df_SEAS5 = df_SEAS5.iloc[it_start:it_end,:]
    df_SEAS5_train = df_SEAS5.iloc[ind_train]
    df_SEAS5_valid = df_SEAS5.iloc[ind_valid]
    df_SEAS5_test = df_SEAS5.iloc[ind_test]


    from functions_encoderdecoder import get_predictor_clim
    df_train_clim, df_valid_clim, df_test_clim = get_predictor_clim(df_train,df_valid,df_test,
                                                                    ind_train,ind_valid,ind_test,
                                                                    time_train,train_years,time,nw=1,verbose = True)

    # df_SEAS5_train_clim, df_SEAS5_valid_clim, df_SEAS5_test_clim = get_predictor_clim(df_SEAS5_train,df_SEAS5_valid,df_SEAS5_test,
    #                                                                 ind_train,ind_valid,ind_test,
    #                                                                 time_train,train_years,time,nw=1,verbose = True)


    #%%
    df = df_train
    df_clim = df_train_clim
    df_fcst = df_SEAS5_train
    time_in = time_train
    n_forecasts = len(forecast_vars)
    window_size = input_len
    forecast_size = pred_len
    batch_size = batch_size
    shuffle=False


    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    # Selecting windows
    data = tf.data.Dataset.from_tensor_slices(df.values)
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    data_clim = tf.data.Dataset.from_tensor_slices(df_clim.values)
    data_clim = data_clim.window(total_size, shift=1, drop_remainder=True)
    data_clim = data_clim.flat_map(lambda k: k.batch(total_size))

    time_tf = tf.data.Dataset.from_tensor_slices(time_in)
    time_tf = time_tf.window(total_size, shift=1, drop_remainder=True)
    time_tf = time_tf.flat_map(lambda k: k.batch(total_size))

    data_fcst = tf.data.Dataset.from_tensor_slices(df_fcst.values)
    data_fcst = data_fcst.window(total_size, shift=1, drop_remainder=True)
    data_fcst = data_fcst.flat_map(lambda k: k.batch(total_size))


    # Zip all datasets together so that we can filter out the samples
    # that are discontinuous in time due to cross-validation splits.
    all_ds = tf.data.Dataset.zip((data, data_clim, data_fcst, time_tf))
    all_ds_filtered =  all_ds.filter(lambda d,dc,dfcst,t: tf.math.equal(t[-1]-t[0]+1,total_size))

    #%%
    # Then extract the separate data sets
    def tf_month(tf_in,it):
        return (dt.timedelta(days=int(tf_in[it].numpy()))+date_ref).month

    data_filtered = all_ds_filtered.map(lambda d,dc,dfcst,t: tf.concat([d,dfcst[:,tf.py_function(tf_month,[t,window_size],Tout=tf.int32)-1:tf.py_function(tf_month,[t,window_size],Tout=tf.int32)] ],1))
    data_clim_filtered =  all_ds_filtered.map(lambda d,dc,dfcst,t: dc)
    time_filtered =  all_ds_filtered.map(lambda d,dc,dfcst,t: t)
    # data_filtered = all_ds_filtered.map(lambda d,dc,dfcst,t: d)
    # data_clim_filtered =  all_ds_filtered.map(lambda d,dc,dfcst,t: dc)
    # data_fcst_filtered =  all_ds_filtered.map(lambda d,dc,dfcst,t: dfcst)
    # time_filtered =  all_ds_filtered.map(lambda d,dc,dfcst,t: t)



    # TEST TO SEE THAT THE FORECAST VARIABLE IS SET PROPERLY:
    # for it,test in enumerate(test_filtered):
    #     print(it,test[window_size:].shape)

    #     if it ==200:
    #         print(time_train[it],date_ref+dt.timedelta(days=int(time_train[it]+window_size)))
    #         print(df_SEAS5_train.iloc[it+window_size:it+window_size+60,(date_ref+dt.timedelta(days=int(time_train[it]+window_size))).month-1])
    #         plt.figure()
    #         plt.plot(np.arange(60),test[window_size:,-1])
    #         plt.plot(np.arange(60),df_SEAS5_train.iloc[it+window_size:it+window_size+60,(date_ref+dt.timedelta(days=int(time_train[it]+window_size))).month-1])
    #%%

    for it,test in enumerate(time_filtered):
        print(it, test.shape)

    #%%
    # Shuffling data
    # !!!!! NOT SURE HOW TO DEAL WITH SHUFFLE AND RECONSTRUCT THE SHUFFLED TIME SERIES...
    # so we keep shuffle to False for now...
    shuffle = False
    if shuffle:
        shuffle_buffer_size = len(data_filtered) # This number can be changed
    #     data = data.shuffle(shuffle_buffer_size, seed=42)
        data_filtered = data_filtered.shuffle(shuffle_buffer_size, seed=42)
        data_clim_filtered =  data_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
        time_filtered =  time_filtered.shuffle(shuffle_buffer_size, seed=42)

    # Extracting (past features, forecasts, decoder initial recurrent input) + targets
    # NOTE : the initial decoder input is set as the last value of the target.
    if n_forecasts > 0:
        data_filtered = data_filtered.map(lambda k: ((k[:-forecast_size,1:-n_forecasts], # Past predictors samples
                                                      k[-forecast_size:, -n_forecasts:], # Future forecasts samples
                                                      k[-forecast_size-1:-forecast_size,0:1] # Decoder input: last time step of target before prediction time starts
                                                     ),
                                                 k[-forecast_size:, 0:1])) # Target samples during prediction time

        data_clim_filtered = data_clim_filtered.map(lambda k: (k[:-forecast_size,1:-n_forecasts], # Past predictor climatology samples
                                                               k[-forecast_size:,0:1])) # Target climatology samples during prediction time



    else:
        data_filtered = data_filtered.map(lambda k: ((k[:-forecast_size,1:],  # Past predictors samples
                                                      k[-forecast_size-1:-forecast_size,0:1] # Decoder input: last time step of target before prediction time starts
                                                     ),
                                                 k[-forecast_size:, 0:1])) # Target samples during prediction time

        data_clim_filtered = data_clim_filtered.map(lambda k: (k[:-forecast_size,1:], # Past predictor climatology samples
                                                               k[-forecast_size:,0:1])) # Target climatology samples during prediction time

    time_filtered = time_filtered.map(lambda k: (k[:-forecast_size], # Time for past predictors samples
                                                 k[-forecast_size:]))    # Time for prediction samples



    # return data_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), data_clim_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), time_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)




    #%%
    for v1,v2 in data_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE):
        print(v1[0].shape,v1[1].shape,v1[2].shape,v2.shape)


