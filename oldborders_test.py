# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:43:11 2019

@author: qx911590
"""

from __future__ import division
import pylab as pl
import glob
from mpl_toolkits.basemap import Basemap
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


pl.close('all')
homedir = '/home/users/qx911590/'

#reader = shpreader.Reader(homedir+'cntry1880')

ax = pl.axes(projection=ccrs.PlateCarree())
ax.set_extent([-31,26,31,67])
shape_feature = ShapelyFeature(Reader(homedir+'cntry1914').geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
ax.add_feature(shape_feature,facecolor='w',linewidth=0.5)