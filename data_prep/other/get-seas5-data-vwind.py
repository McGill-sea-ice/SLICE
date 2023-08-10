#!/home/burcherr/anaconda3/bin/python
"""
Download historic SEAS5 data from the C3S Copernicus 

Notes
-----

Get your UID and API key from the CDS portal at the address
https://cds.climate.copernicus.eu/user and write it into
the configuration file, so it looks like:

$ cat ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API key>
verify: 0

References
----------

https://pypi.org/project/cdsapi/

"""

import argparse
import cdsapi
import os

# CDS creditionals key
key = "68986:cd4929b1-ca5d-4f2b-b884-4d89b243703c"

def download_seas5(features, region, output_dir = os.getcwd(), start_year=1979, end_year=2018):
    """
    Download SEAS5 files

    :param path: String. Full directory to download files to
    :param start_year: Integer. Start year in YYYY.
    :param end_year: Integer. End year in YYYY.
    :return: None
    """

    base = "SEAS5_"
    url = "https://cds.climate.copernicus.eu/api/v2"

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

        for month in range(1, 13):  # up to 12
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

                # eg. SEAS5_10m_u_component_of_wind_197901.nc
                filename = base + feature + extension

                if not os.path.isfile(filename):
                    print("Downloading file {}".format(filename))

                    downloaded = False

                    while not downloaded:
                        try:
                            client = cdsapi.Client(url=url, key=key, retry_max=5)
                            client.retrieve(
                                'seasonal-original-single-levels',
                                {
                                    'format': 'netcdf',
                                    'originating_centre': 'ecmwf',
                                    'system': '5',
                                    'variable': feature,
                                    'area': [
                                        toplat, leftlon, downlat,
                                        rightlon,
                                    ],
                                    'day': '01',
                                    # API ignores cases where there are less than 31 days
                                    'month': month,
                                    'year': str(year),
                                    'leadtime_hour': [
                                         '24','48','72','96',
                                         '120','144','168','192','216','240',
                                         '264','288','312','336','360','384',
                                         '408','432','456','480','504','528',
                                         '552','576','600','624','648','672',
                                         '696','720','744','768','792','816',
                                         '840','864','888','912','936','960',
                                         '984','1008','1032','1056','1080','1104',
                                         '1128','1152','1176','1200','1224','1248',
                                         '1272','1296','1320','1344', '1368','1392',
                                         '1416','1440','1464','1488','1512','1536',
                                         '1560','1584','1608','1632','1656','1680',
                                         '1704','1728','1752','1776','1800','1824',
                                         '1848','1872','1896','1920','1944','1968',
                                         '1992','2016','2040','2064','2088','2112',
                                         '2136','2160','2184','2208','2232','2256','2280',
                                         '2304','2328','2352','2376','2400','2424',
                                         '2448','2472','2496','2520','2544','2568',
                                         '2592','2616','2640','2664','2688','2712',
                                         '2736','2760','2784','2808','2832','2856',
                                         '2880','2904','2928','2952','2976','3000',
                                         '3024','3048','3072','3096','3120','3144',
                                         '3168','3192','3216','3240','3264','3288',
                                         '3312','3336','3360','3384','3408','3432',
                                         '3456','3480','3504','3528','3552','3576',
                                         '3600','3624','3648','3672','3696','3720',
                                         '3744','3768','3792','3816','3840','3864',
                                         '3888','3912','3936','3960','3984','4008',
                                         '4032','4056','4080','4104','4128','4152',
                                         '4176','4200','4224','4248','4272','4296',
                                         '4320','4344','4368','4392','4416','4440',
                                         '4464','4488','4512','4536','4560','4584',
                                         '4608','4632','4656','4680','4704','4728',
                                         '4752','4776','4800','4824','4848','4872',
                                         '4896','4920','4944','4968','4992','5016',
                                         '5040','5064','5088','5112','5136','5160'
                                    ],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEAS5 files to current working directory.")
    # parser.add_argument("--start_year", type=int, default=1991,
    #                     help="Year to start in YYYY format.")
    # parser.add_argument("--end_year", type=int, default=2022,
    #                     help="Year to end in YYYY format.")
    parser.add_argument("--start_year", type=int, default=1993,
                        help="Year to start in YYYY format.")
    parser.add_argument("--end_year", type=int, default=2022,
                        help="Year to end in YYYY format.")
    args = parser.parse_args()

    feature_list = [
                    # '2m_temperature', 
                    # 'maximum_2m_temperature_in_the_last_24_hours', 
                    # 'minimum_2m_temperature_in_the_last_24_hours', 
                    # 'snowfall', 
                    # 'total_precipitation',
                    # 'total_cloud_cover',
                    # 'mean_sea_level_pressure',
                    # '10m_u_component_of_wind', 
                    '10m_v_component_of_wind', 
                    # '2m_dewpoint_temperature',
                    # 'runoff', 'sea_ice_cover',
                    # 'sea_surface_temperature', 
                    # 'surface_latent_heat_flux',
                    # 'surface_net_solar_radiation', 
                    # 'surface_net_thermal_radiation', 
                    # 'surface_sensible_heat_flux',
                    # 'surface_solar_radiation_downwards', 
                    # 'surface_thermal_radiation_downwards', 
                    ]

    reg = 'D'
    out_dir = '/storage/amelie/slice/data/raw/SEAS5/region'+reg

    download_seas5(features=feature_list, region=reg, output_dir=out_dir, start_year=args.start_year, end_year=args.end_year)
