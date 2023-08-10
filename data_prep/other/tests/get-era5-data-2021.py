#!/home/burcherr/anaconda3/bin/python
"""
Download historic ERA5 files

Initial created by Matthew King 2019 @ NRC

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

def download_era5(features, region, output_dir = os.getcwd(), start_year=1991, end_year=2021, start_month=1, end_month=13, start_day=1, end_day = 31):
    """
    Download ERA5 files

    :param path: String. Full directory to download files to
    :param start_year: Integer. Start year in YYYY.
    :param end_year: Integer. End year in YYYY.
    :return: None
    """

    base = "ERA5_"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 files to current working directory.")
    parser.add_argument("--start_year", type=int, default=2021,
                        help="Year to start in YYYY format.")
    parser.add_argument("--end_year", type=int, default=2022,
                        help="Year to end in YYYY format.")
    parser.add_argument("--start_month", type=int, default=1,
                        help="Month to start in MM format.")
    parser.add_argument("--end_month", type=int, default=13,
                        help="Month+1 to end in MM format.")

    args = parser.parse_args()

    feature_list = ['2m_temperature',
                    '10m_u_component_of_wind','10m_v_component_of_wind',
                    'total_precipitation','total_cloud_cover',
                    'snowfall','mean_sea_level_pressure',
                    '2m_dewpoint_temperature','evaporation',
                    'snowmelt', 'runoff',
                    'sea_ice_cover','lake_ice_depth', 'lake_ice_temperature',
                    'lake_total_layer_temperature','surface_solar_radiation_downwards',
                    'surface_latent_heat_flux','surface_sensible_heat_flux',
                    'sea_surface_temperature',
                    ]

    reg = 'D'
    out_dir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/ERA5/region'+reg

    download_era5(features=feature_list, region=reg, output_dir=out_dir, start_year=args.start_year, end_year=args.end_year, start_month=args.start_month, end_month=args.end_month)
