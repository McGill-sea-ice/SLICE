#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:23:26 2022

@author: Amelie
"""

import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(9, 13),
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax

extent = [-75,-73,45.2,45.9]
# extent = [-39, -38.25, -13.25, -12.5]
spath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/openstreetmap/'


# fig, ax = make_map(projection=ccrs.PlateCarree())
# ax.set_extent(extent)

# shp = shapereader.Reader(spath+'coastlines-split-4326/lines')
# for record, geometry in zip(shp.records(), shp.geometries()):
#     ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='w',
#                       edgecolor='black')

fig, ax = make_map(projection=ccrs.PlateCarree())
ax.set_extent(extent)
land_10m = cartopy.feature.GSHHSFeature(scale='h', levels=[1],facecolor=cartopy.feature.COLORS['land'])
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(land_10m)
# shp = shapereader.Reader(spath+'land-polygons-complete-4326/land_polygons')
# for record, geometry in zip(shp.records(), shp.geometries()):
#     ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',
#                       edgecolor='black')

#%%
# FROM MATT:


# %%
# from matplotlib.colors import cnames

# fig, ax = make_map(projection=ccrs.PlateCarree())
# extent = [-74,-73.4,45.2,45.9]

# ax.set_extent(extent)
# shp = shapereader.Reader(spath+'gadm40_CAN_shp/gadm40_CAN_0')

# k = 0
# colors = list(cnames.keys())
# for record, geometry in zip(shp.records(), shp.geometries()):
#     if record.attributes['NAME_1'].decode('latin-1') == u'Bahia':
#         if k+1 == len(colors):
#             k = 0
#         else:
#             k += 1
#         color = colors[k]
#         ax.add_geometries([geometry], ccrs.PlateCarree(),
#                         facecolor=color, edgecolor='black')