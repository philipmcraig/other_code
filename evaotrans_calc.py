# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:51:15 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import xarray as xr

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'

rtifile = xr.open_dataset(ncdir+'rti'+'_season.nc')
rtidata = rtifile.variables['rti'][:]
rtidata = xr.DataArray(rtidata)
lat = xr.DataArray(rtifile.variables['latitude'][:])
lon = xr.DataArray(rtifile.variables['longitude'][:])
time = xr.DataArray(rtifile.variables['time'][:])
rtifile.close()

epfile = xr.open_dataset(ncdir+'ep'+'_season.nc')
epdata = epfile.variables['ep'][:]
epdata = xr.DataArray(epdata)
#lat = xr.DataArray(ncfile.variables['latitude'][:])
#lon = xr.DataArray(ncfile.variables['longitude'][:])
#time = xr.DataArray(ncfile.variables['time'][:])
epfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])
months = pl.asarray([i[5:7] for i in time])

ind = pl.where((years=='1979') & (months=='04'))
ind = ind[0][0]

evp = rtidata - epdata

ds = xr.Dataset({'evp': (('time','longitude','latitude'),evp.values)},
            coords={'time': time,
                    'longitude': lon.values,
                    'latitude': lat.values})

ds.to_netcdf(ncdir + 'evp_season.nc')