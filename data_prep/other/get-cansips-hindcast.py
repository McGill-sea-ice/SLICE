#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:44:18 2022

@author: Amelie
"""
import numpy as np
import urllib
import os

#%%
###
def download_cansips_hindcast(features, output_dir = os.getcwd(), start_year=1980, end_year=2020, start_month=1, end_month=12):

    base = "cansips_hindcast_raw_"
    res = "latlon1.0x1.0_"
    url_base = "https://dd.meteo.gc.ca/ensemble/cansips/grib2/hindcast/raw/"

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

            # _1997-01_allmembers.grib2
            extension = "_{}-{}_allmembers.grib2".format(year, str(month).rjust(2, '0'))
            url = url_base +"{}/{}/".format(year, str(month).rjust(2, '0'))

            for feature in features:
                print(feature)

                # eg. cansips_hindcast_raw_latlon1.0x1.0_UGRD_ISBL_0200_1997-01_allmembers.grib2
                filename = base + res + feature + extension

                if not os.path.isfile(filename):
                    print("Downloading file {}".format(filename))

                    downloaded = False

                    while not downloaded:
                        try:
                            urllib.request.urlretrieve(url+filename, filename)

                        except Exception as e:
                            print(e)

                            # Delete the partially downloaded file.
                            if os.path.isfile(filename):
                                os.remove(filename)

                        else:
                            # no exception implies download was complete
                            downloaded = True



# url = "https://dd.meteo.gc.ca/ensemble/cansips/grib2/hindcast/raw/1980/01/cansips_hindcast_raw_latlon1.0x1.0_HGT_ISBL_0500_1980-01_allmembers.grib2"
# fout = "./cansips_hindcast_raw_latlon1.0x1.0_HGT_ISBL_0500_1980-01_allmembers.grib2"
# urllib.request.urlretrieve(url, fout)



feature_list = ['WTMP_SFC_0','PRATE_SFC_0','TMP_TGL_2m','TMP_ISBL_0850',
                'PRMSL_MSL_0','HGT_ISBL_0500','UGRD_ISBL_0200','UGRD_ISBL_0850',
                'VGRD_ISBL_0200','VGRD_ISBL_0850']

out_dir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CanSIPS/hindcast/raw/'
start_year = 1983
end_year = 2020
download_cansips_hindcast(features=feature_list, output_dir=out_dir, start_year=start_year, end_year=end_year)

