#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:23:06 2021

@author: Amelie
"""
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


def get_doy(date_arr, out_arr, flip_date_new_year = True):

    for iy in range(nyears):
        yr = date_start.year + iy

        for i in range(len(date_arr)):
            dt_fi = (date_ref+dt.timedelta(days=date_arr[i]))
            yr_fi = dt_fi.year
            mo_fi = dt_fi.month

            if ((yr_fi == yr) & (mo_fi > 10)) | (yr_fi == yr+1) & (mo_fi < 4):

                out_arr[iy] = (dt_fi - dt.date(yr_fi, 1, 1)).days + 1
                if calendar.isleap(yr_fi) & (mo_fi > 9):
                    out_arr[iy] -= 1 # Remove 1 for leap years so that
                                 # e.g. Dec 1st is always DOY = 335
                # print(iy,yr,years[iy],yr_fi,mo_fi,dy_fi,out_arr[iy])

    if flip_date_new_year: out_arr[out_arr < 200] = out_arr[out_arr < 200]  + 365

    return out_arr


def corr_freezeup(loc1,loc2,comp_fi,comp_si,comp_ci):
    ice_data1 = np.load(fp+'freezeup_dates_'+loc1[1]+'/freezeup_'+loc1[1]+'_'+loc1[0]+'.npz',allow_pickle='TRUE')
    ice_data2 = np.load(fp+'freezeup_dates_'+loc2[1]+'/freezeup_'+loc2[1]+'_'+loc2[0]+'.npz',allow_pickle='TRUE')

    if comp_fi:
        fi1 = ice_data1['freezeup_fi']
        fi1 = fi1[~np.isnan(fi1)]
        fi2 = ice_data2['freezeup_fi']
        fi2 = fi2[~np.isnan(fi2)]

        doy_fi_arr = np.zeros((nyears,2))*np.nan
        doy_fi_arr[:,0] = get_doy(fi1,doy_fi_arr[:,0])
        doy_fi_arr[:,1] = get_doy(fi2,doy_fi_arr[:,1])

        xfi = doy_fi_arr[:,0]
        yfi = doy_fi_arr[:,1]
        mask1 = ~np.isnan(xfi)
        mask2 = ~np.isnan(yfi)
        xfi = xfi[mask1 & mask2]
        yfi = yfi[mask1 & mask2]
        linfit_fi, rsqr_fi = linear_fit(xfi,yfi)

    if comp_si:
        si1 = ice_data1['freezeup_si']
        si1 = si1[~np.isnan(si1)]
        si2 = ice_data2['freezeup_si']
        si2 = si2[~np.isnan(si2)]

        doy_si_arr = np.zeros((nyears,2))*np.nan
        doy_si_arr[:,0] = get_doy(si1,doy_si_arr[:,0])
        doy_si_arr[:,1] = get_doy(si2,doy_si_arr[:,1])

        xsi = doy_si_arr[:,0]
        ysi = doy_si_arr[:,1]
        mask1 = ~np.isnan(xsi)
        mask2 = ~np.isnan(ysi)
        xsi = xsi[mask1 & mask2]
        ysi = ysi[mask1 & mask2]
        linfit_si, rsqr_si = linear_fit(xsi,ysi)

    if comp_ci:
        ci1 = ice_data1['freezeup_ci']
        ci1 = ci1[~np.isnan(ci1)]
        ci2 = ice_data2['freezeup_ci']
        ci2 = ci2[~np.isnan(ci2)]

        doy_ci_arr = np.zeros((nyears,2))*np.nan
        doy_ci_arr[:,0] = get_doy(ci1,doy_ci_arr[:,0])
        doy_ci_arr[:,1] = get_doy(ci2,doy_ci_arr[:,1])

        xci = doy_ci_arr[:,0]
        yci = doy_ci_arr[:,1]
        mask1 = ~np.isnan(xci)
        mask2 = ~np.isnan(yci)
        xci = xci[mask1 & mask2]
        yci = yci[mask1 & mask2]
        linfit_ci, rsqr_ci = linear_fit(xci,yci)


    plt.figure()
    xplot=np.arange(330,410)
    plt.plot(np.arange(330,410),np.arange(330,410),'-',color='black',linewidth=0.5)
    if comp_fi:
        plt.plot(doy_fi_arr[:,0],doy_fi_arr[:,1],'o', label='First ice')
        plt.plot(xplot,xplot*linfit_fi[0]+linfit_fi[1],':',color=plt.get_cmap('tab20')(1))
        plt.text(327,392,'Rsqr = %3.2f'%(rsqr_fi), color=plt.get_cmap('tab20')(1))
        print('Linfit, fi:'+ str(linfit_fi))
    if comp_si:
        plt.plot(doy_si_arr[:,0],doy_si_arr[:,1],'*', label='Stable ice')
        plt.plot(xplot,xplot*linfit_si[0]+linfit_si[1],':',color=plt.get_cmap('tab20')(3))
        plt.text(327,388,'Rsqr = %3.2f'%(rsqr_si), color=plt.get_cmap('tab20')(3))
        print('Linfit, si:'+ str(linfit_si))
    if comp_ci:
        plt.plot(doy_ci_arr[:,0],doy_ci_arr[:,1],'+', label='Chart ice')
        plt.plot(xplot,xplot*linfit_ci[0]+linfit_ci[1],':',color=plt.get_cmap('tab20')(5))
        plt.text(327,383,'Rsqr = %3.2f'%(rsqr_ci), color=plt.get_cmap('tab20')(5))
        print('Linfit, ci:'+ str(linfit_ci))
    plt.xlabel('DOY - '+loc1[0]+', '+loc1[1])
    plt.ylabel('DOY - '+loc2[0]+', '+loc2[1])
    plt.xlim(320,420)
    plt.legend()


    # plt.figure()
    # if comp_fi:
    #     plt.hist(doy_fi_arr[:,0]-doy_fi_arr[:,1],10)
    #     plt.boxplot(xfi-yfi,vert=False,positions=[5],widths=[1])


# ==========================================================================

years = [1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,
         1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,
         2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,
         2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
         2020]

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
nyears = len(years)

freezeup_SLSMC_list = ['BeauharnoisCanal','MontrealPort','SouthShoreCanal', 'LakeStLouis',
                      'Summerstown','Iroquois','LakeStLawrence','LakeStFrancisEAST']
freezeup_HQ_list = ['BeauharnoisCanal']

fp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'

# loc1 = ['SouthShoreCanal','SLSMC']
# loc2 = ['BeauharnoisCanal','SLSMC']
# corr_freezeup(loc1,loc2,comp_fi=True,comp_si=True,comp_ci=True)

loc1 = ['BeauharnoisCanal','HQ']
loc2 = ['BeauharnoisCanal','SLSMC']
corr_freezeup(loc1,loc2,comp_fi=True,comp_si=True,comp_ci=False)


# loc1 = ['LakeStFrancisEAST','SLSMC']
# loc2 = ['BeauharnoisCanal','HQ']
# corr_freezeup(loc1,loc2,comp_fi=True,comp_si=True,comp_ci=False)
#
# loc1 = ['LakeStFrancisEAST','SLSMC']
# loc2 = ['BeauharnoisCanal','SLSMC']
# corr_freezeup(loc1,loc2,comp_fi=True,comp_si=True,comp_ci=False)


# loc1 = ['SouthShoreCanal','SLSMC']
# loc2 = ['MontrealPort','SLSMC']
# corr_freezeup(loc1,loc2,comp_fi=False,comp_si=False,comp_ci=True)

# loc1 = ['LakeStLawrence','SLSMC']
# loc2 = ['Iroquois','SLSMC']
# corr_freezeup(loc1,loc2,comp_fi=True,comp_si=True,comp_ci=False)

loc1 = ['SouthShoreCanal','SLSMC']
loc2 = ['LakeStLouis','SLSMC']
corr_freezeup(loc1,loc2,comp_fi=True,comp_si=True,comp_ci=True)


