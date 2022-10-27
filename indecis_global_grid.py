# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:03:07 2021

@author: pmcraig
"""

import pylab as pl
import xarray as xr
import pcraig_funcs as pc

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
era5dir = ncasdir + 'ERA5/'
indecis = ncasdir + 'INDECIS/'

ncfile = xr.open_dataset(indecis+'ncfiles/gtg_season.nc')
indlon = xr.DataArray(ncfile.longitude)
indlat = xr.DataArray(ncfile.latitude)
indgtg = xr.DataArray(ncfile.gtg)
indtime = xr.DataArray(ncfile.time)
ncfile.close()

dlon = 0.25
dlat = 0.25

glon = pl.arange(-179.875,180,0.25)
glat = pl.arange(-89.875,90,0.25)

gtg_global = pl.zeros([indgtg.shape[0],glon.size,glat.size])
gtg_global[:,:,:] = pl.float32('nan')

lonmin = pl.where(glon==indlon[0].data)[0][0]
lonmax = pl.where(glon==indlon[-1].data)[0][0]

latmin = pl.where(glat==indlat[0].data)[0][0]
latmax = pl.where(glat==indlat[-1].data)[0][0]

gtg_global[:,lonmin:lonmax+1,latmin:latmax+1] = indgtg.data

gtg_global = xr.DataArray(gtg_global,
                          coords={"time": indtime.data,
                                  "longitude": glon,
                                  "latitude": glat},
                            dims=["time","longitude","latitude"],
                            name='gtg',
                            attrs={"units": "gtg"})

glon = xr.DataArray(glon,
                      coords={"longitude": glon},
                        dims=["longitude"],
                        name='longitude',
                        attrs={"units": "degrees_east"})

glat = xr.DataArray(glat,
                    coords={"latitude": glat},
                    dims=["latitude"],
                    name='latitude',
                    attrs={"units": "degrees_north"})

gtg_global.to_netcdf(indecis+'ncfiles/gtg_season_global.nc',mode='w')
                     #encoding={"gtg":})
#glon.to_netcdf(indecis+'ncfiles/gtg_season_global.nc',mode='a')
#glat.to_netcdf(indecis+'ncfiles/gtg_season_global.nc',mode='a')