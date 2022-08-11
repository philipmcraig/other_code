#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:39:12 2018

@author: qx911590
"""

from __future__ import division
import pylab as pl
import glob
from mpl_toolkits.basemap import Basemap
import pyproj
from hf_func import OSGB36toWGS84
from rt_func import WGS84toOSGB36
import pcraig_funcs as pc
#import cartopy
#import cartopy.crs as ccrs

pl.close('all')
resdir = '/home/users/qx911590/weatherrescue/'

filenames = glob.glob(resdir+'UK_daily_rainfall/*')
filenames = pl.asarray(filenames)
filenames = pl.sort(filenames)

oct1903 = filenames[273:304]
rain = pl.zeros([len(oct1903),290,180])

for name in range(len(oct1903)):
    x = pl.genfromtxt(oct1903[name],skip_header=6)
    #y = pl.where(x!=-9999)
    z = pl.where(x==-9999)
    x[z] = pl.float32('nan')
    rain[name] = x

rainmean = pl.mean(rain,axis=0)
rainsum  = pl.sum(rain,axis=0)
#ax = pl.axes(projection=ccrs.OSGB)
#ax.coastlines()

#bng = pyproj.Proj(init='epsg:27700')
#wgs84 = pyproj.Proj(init='epsg:4326')#'epsg:4326'

#llcrnrlon,llcrnrlat = pyproj.transform(bng,wgs84,-200000,-200000)
#llcrnrlon,llcrnrlat = pyproj.transform(bng,wgs84,0,0)
#llcrnrlon = -7.5600; llcrnrlat = 49.9600
#llcrnrlon = -12; llcrnrlat = 45
#lon_0,lat_0 = pyproj.transform(bng,wgs84,250000,525000)
#lon_0,lat_0 = pyproj.transform(bng,wgs84,350000,625000)
#urcrnrlon,urcrnrlat = pyproj.transform(bng,wgs84,700000,1250000)
#urcrnrlon = 1.7800; urcrnrlat = 60.8400

lat_in = pl.linspace(45,65,x.shape[0])
lon_in = pl.linspace(-15,10,x.shape[1])

x_coords = pl.arange(-200000+2500,700001-2500,5000)
y_coords = pl.arange(-200000+2500,1250001-2500,5000)

#lonlat = pl.zeros([x_coords.size,y_coords.size],dtype=object)
lons = pl.zeros([x_coords.size,y_coords.size])
lats = pl.zeros([x_coords.size,y_coords.size])

for i in range(x_coords.size):
    for j in range(y_coords.size):
        a,b = OSGB36toWGS84(x_coords[i],y_coords[j])
        #lonlat[i,j] = (a,b)
        lons[i,j] = a; lats[i,j] = b

pl.figure(figsize=(10,10))
m = Basemap(llcrnrlon=-9,llcrnrlat=49,urcrnrlon=5,
            urcrnrlat=61,resolution='i',projection='tmerc',
                                            lon_0=-2.5,lat_0=55)

#lons, lats = pl.meshgrid(lon_in,lat_in)
X, Y = m(lons,lats)

m.drawcoastlines(); m.drawcountries()
cmap = pl.get_cmap('viridis')
cmap.set_under('white')
#m.pcolormesh(X.T,Y.T,pl.flipud(rainmean),norm=pl.Normalize(0,25),cmap=cmap)
cf = m.contourf(X.T,Y.T,pl.flipud(rainsum),norm=pl.Normalize(0,300),cmap=cmap,
           levels=pl.linspace(0,300,13),extend='max')
#m.plot(-2.456,50.514,marker='x',color='r',latlon=True)
cb = m.colorbar(cf,extend='max')
cb.set_label('mm',fontsize=18)
pl.title('Total UK rainfall October 1903')
#pl.savefig(resdir+'oct1903_total_rain_300.png')

#for i in range(lonlat.shape[0]):
#    for j in range(lonlat.shape[1]):
#        a = lonlat[i,j][0]; b = lonlat[i,j][1] 
#        m.plot(a,b,latlon=True,marker='x',color='r')

#ax = pl.axes(projection=ccrs.TransverseMercator())
#ax.coastlines(resolution='50m')
#ax.contourf(lons,lats,pl.flipud(x),transform=ccrs.PlateCarree())