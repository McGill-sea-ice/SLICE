#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:44:59 2020

@author: Amelie
"""




import numpy as np
import matplotlib.pyplot as plt
from owslib.wcs import WebCoverageService
from owslib.wfs import WebFeatureService
from owslib.wms import WebMapService
import cartopy


#%%
web_link = "https://geoportal.gc.ca/arcgis/rest/services/FGP/500m_Gridded_Bathymetry/MapServer"
# web_link = "https://data.chs-shc.ca/geoserver/wms?request=GetCapabilities&service=WMS&layers=caris:NONNA 10&legend_format=image/png&feature_info_type=text/plain"
# web_link = "http://geoegl.msp.gouv.qc.ca/ws/radarsat.fcgi?VERSION=1.0.0&SERVICE=WCS&REQUEST=GETCAPABILITIES"

# wcs = WebCoverageService(web_link, version='1.0.0')
# wfs = WebFeatureService(web_link, version='1.0.0')
wms = WebMapService(web_link, version='1.1.1')

# print('WCS: ', list(wcs.contents))
# print('WFS: ', list(wfs.contents))
print('WMS: ', list(wms.contents))
print("Title: ", wms.identification.title)
print("Type: ", wms.identification.type)
print("Operations: ", [op.name for op in wms.operations])
print("GetMap options: ", wms.getOperationByName('GetMap').formatOptions)
wms.contents.keys()



k=list(wms.contents)
# for key in k[0:5]:
#     print(wms.contents[key].title)

#%%

# name = 'R2_MTL_14janv09'
name = 'caris:NONNA 10'
layer = wms.contents[name]
print("Abstract: ", layer.abstract)
print("BBox: ", layer.boundingBoxWGS84)
print("CRS: ", layer.crsOptions)
print("Styles: ", layer.styles)
print("Timestamps: ", layer.timepositions)
# HTML(layer.parent.abstract)

#%%
response = wms.getmap(layers=[name],
                      srs='EPSG:32198',
                      bbox=(-313236.0, 172455.0, -230099.0, 317005.0),
                      size=(500,500),
                      format='image/tiff')



import io
image = io.BytesIO(response.read())


data = plt.imread(image, format='tiff')
plt.figure();plt.imshow(data)

#%%
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_axes([0,0,1,1], projection=cartopy.crs.Mollweide())
# ax.imshow(data, origin="upper", extent=(-72.6091, 45.4689, -71.4421, 46.8048),
#           transform=cartopy.crs.PlateCarree())
# ax.coastlines()
# plt.show()



