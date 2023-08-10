#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 09:19:34 2022

@author: Amelie
"""
import argparse
import os
import calendar
# import urllib2
import urllib.request
import datetime as dt

def download_CMC_GHRSST(output_dir = os.getcwd(), start_year=1991, end_year=2022):
    """
    Download CMC Global SST Reanalysis

    :param path: String. Full directory to download files to
    :param start_year: Integer. Start year in YYYY.
    :param end_year: Integer. End year in YYYY.
    :return: None
    """

    for year in range(start_year, end_year):
        print(year)
        os.chdir(output_dir)

        # 0.2 deg. 1991-2015
        if (year >= 1991) & (year <= 2015) :
            res = '0.2deg'
            version = 'v2'
            end = '-fv02.0'
        # 0.1 deg 2016-2022
        if (year >= 2016) :
            res = '0.1deg'
            version = 'v3'
            end = '-fv03.0'

        for doy in range(365+calendar.isleap(year)):
            print(doy)
            os.chdir(output_dir)

            subdirectory = "{}".format(year)
            if not os.path.isdir(subdirectory):
                os.mkdir(subdirectory)

            os.chdir(subdirectory)

            doydate = dt.date(year,1,1)+dt.timedelta(days=doy)
            month = doydate.month
            day = doydate.day
            filename = str(year)+str(month).rjust(2, '0')+str(day).rjust(2, '0')+'120000-CMC-L4_GHRSST-SSTfnd-CMC'+res+'-GLOB-v02.0'+end+'.nc'
            url = 'https://opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/CMC/CMC'+ res+'/'+version+'/'+str(year)+'/'+str(doy+1).rjust(3, '0')+'/'+str(year)+str(month).rjust(2, '0')+str(day).rjust(2, '0')+'120000-CMC-L4_GHRSST-SSTfnd-CMC'+res+'-GLOB-v02.0'+end+'.nc?time[0:1:0],lat[0:1:900],lon[0:1:1799],analysed_sst[0:1:0][0:1:900][0:1:1799],analysis_error[0:1:0][0:1:900][0:1:1799],sea_ice_fraction[0:1:0][0:1:900][0:1:1799],mask[0:1:0][0:1:900][0:1:1799]'
            if not os.path.isfile(filename):
                print("Downloading file {}".format(filename))
                downloaded = False

                while not downloaded:
                    try:
                        try:
                            urllib.request.urlretrieve(url,filename)
                            # filedata = urllib.request.urlopen(url)
                            # datatowrite = filedata.read()

                            # with open(filename,'wb') as f:
                            #     f.write(datatowrite)

                        except Exception as e:
                            print(e)
                    except Exception as e:
                        print(e)

                        # Delete the partially downloaded file.
                        if os.path.isfile(filename):
                            os.remove(filename)
                    else:
                        # no exception implies download was complete
                        downloaded = True



save_dir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CMC_GHRSST/'
download_CMC_GHRSST(save_dir, start_year=1992, end_year=2022)
